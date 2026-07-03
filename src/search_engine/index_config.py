"""OpenSearch index schema + settings.

Schema version is stamped onto every document via `INDEX_SCHEMA_VERSION`
so a reindex after a mapping change can be reconciled at query time.
Bump the suffix (`v1` → `v2` → ...) every time you change `build_mappings`
in a way that affects retrieval (added/removed fields, analyzer
changes, dim change on the embedding) — then run:

    python manage.py opensearch_setup --recreate
    python manage.py opensearch_reindex

`--recreate` deletes the physical index and resets the RagChunk
tracking table so every chunk is treated as new and re-embedded
against the new mapping.

v2 (2026-05) added:
  * Author identity on chat chunks (`author_id`, `author_name`,
    `chat_message_id`) — unlocks "what did Bob say about X" without a
    DB round-trip.
  * Task metadata on task chunks (`task_status`, `task_priority`,
    `task_assignee_id`, `task_milestone_id`, `task_sprint_id`) —
    enables status-aware ranking and "my open tasks about X" overlays
    in hybrid search.
  * Note ownership (`note_owner_id`, `note_parent_id`).
  * Retrieval-quality subfields:
      - `title.prefix` (edge n-gram) — fast typeahead prefix match
      - `search_text.en` (English analyzer) — stemming/synonym recall
  * Configurable index settings (`OPENSEARCH_SHARDS`,
    `OPENSEARCH_REPLICAS`, `OPENSEARCH_REFRESH_INTERVAL`) — production
    can tune via env without code changes.
  * Dropped `source_version` (never written by any chunker).

v3 (2026-06) — multilingual BM25 (7 UI languages: en/ja/es/fr/zh/ar/hi):
  * Requires the OpenSearch `analysis-icu` + `analysis-kuromoji`
    plugins (baked into docker/opensearch/Dockerfile; bundled free on
    AWS managed OpenSearch).
  * New analyzers in `_build_analysis`:
      - `multilingual_icu` (icu_tokenizer + icu_folding) — dictionary
        word-segmentation for ja/zh and accent/Unicode folding for
        es/fr/ar/hi. The default `standard` analyzer shatters CJK into
        per-character unigrams; ICU segments into words.
      - `icu_prefix_index` / `icu_prefix_search` — CJK/JA-aware
        typeahead (edge n-gram over ICU tokens).
      - `japanese_kuromoji` — Japanese morphology ICU can't do:
        base-form/inflection match (走った/走って → 走る), cjk_width,
        ja_stop particle removal.
  * New ADDITIVE subfields (base fields + `.en`/`.prefix` untouched, so
    English/exact-phrase behaviour is byte-for-byte preserved):
      - `search_text.icu`, `snippet_text.icu`, `title.icu` (multilingual_icu)
      - `search_text.ja`, `snippet_text.ja`, `title.ja` (japanese_kuromoji)
      - `title.icu_prefix` (CJK/JA typeahead)
  * Rollout — two equivalent paths (both reach the same retrieval
    quality; embeddings are NOT affected by analyzer changes):
      A) recreate (simplest; re-embeds the corpus):
           opensearch_setup --recreate && opensearch_reindex
      B) zero-re-embed (large prod corpus): _close → PUT _settings
         (add the analyzers) → _open → PUT _mapping (add subfields) →
         POST _update_by_query with an inline script stamping
         index_schema_version="v3". Rebuilds each doc from `_source`,
         preserving the stored embedding vectors.
    NOTE: `--update-mapping` alone is INSUFFICIENT — analyzers live in
    index settings (need close/open) and put_mapping does not
    re-analyze existing documents.
"""

import os

from origin.search_engine.embeddings import get_active_embedding_dimensions

INDEX_SCHEMA_VERSION = "v3"


def build_index_settings():
    # Per-mode HNSW tuning (`m`, `ef_construction`) intentionally left
    # at OpenSearch/Lucene defaults — they're index-time params and
    # changing them requires a full reindex. Query-time `ef_search` is
    # the cheap lever and lives in `search.py` per-call.
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": int(os.environ.get("OPENSEARCH_SHARDS", "1")),
                "number_of_replicas": int(os.environ.get("OPENSEARCH_REPLICAS", "0")),
                "refresh_interval": os.environ.get("OPENSEARCH_REFRESH_INTERVAL", "1s"),
                "analysis": _build_analysis(),
            }
        },
        "mappings": build_mappings(),
    }


def _build_analysis():
    """Custom analyzers used by v2 retrieval-quality subfields.

    `title_prefix` is an edge n-gram tokenizer for fast typeahead
    prefix matching ("Cmd-K" → "C", "Cm", "Cmd", ...). Min 2 chars
    avoids exploding the postings list with single-char shards;
    max 12 chars caps the per-token expansion at a reasonable size.

    `english_basic` is the standard English analyzer with stemming +
    a small custom stopword list. Used as a `.en` subfield on
    `search_text` so the BM25 lane has stemming/synonym recall
    without losing the exact-phrase match path (the base field stays
    on the default standard analyzer).

    v3 multilingual analyzers (require analysis-icu + analysis-kuromoji):
      * `multilingual_icu` — icu_tokenizer does dictionary-based word
        segmentation (Japanese/Chinese have no spaces; the default
        `standard` analyzer would emit one token per CJK character).
        `icu_folding` normalises Unicode + folds accents (café→cafe),
        helping es/fr/ar/hi. `icu_normalizer` (NFKC, char_filter) runs
        first so full-width/compatibility forms are unified.
      * `icu_prefix_index` / `icu_prefix_search` — the typeahead pair
        rebuilt on icu_tokenizer so CJK/JA titles produce real prefix
        tokens. BOTH sides must use ICU or CJK typeahead matches
        nothing. icu_tokenizer is a superset of `standard` on Latin
        text, so English typeahead is unchanged.
      * `japanese_kuromoji` — morphological analysis ICU can't do:
        `kuromoji_baseform` collapses inflections (走った/走って→走る),
        `cjk_width` normalises half/full-width kana, `ja_stop` drops
        particles. `mode: search` decompounds long compounds for
        recall. Degrades gracefully on non-Japanese text (splits Latin
        on whitespace), so it can join the multi_match fan-out without
        any per-request language signal.
    """
    return {
        "char_filter": {
            # NFKC normalisation + case-fold BEFORE tokenization, so the
            # tokenizer sees unified forms (full-width →half-width, etc.).
            "icu_normalizer": {
                "type": "icu_normalizer",
                "name": "nfkc_cf",
                "mode": "compose",
            },
        },
        "filter": {
            "edge_ngram_2_12": {
                "type": "edge_ngram",
                "min_gram": 2,
                "max_gram": 12,
            },
            "english_stop": {"type": "stop", "stopwords": "_english_"},
            "english_stemmer": {"type": "stemmer", "language": "english"},
            # Unicode case-fold + accent/diacritic folding (café→cafe,
            # Arabic presentation-form normalisation, etc.).
            "icu_folding": {"type": "icu_folding"},
        },
        "analyzer": {
            "title_prefix_index": {
                "tokenizer": "standard",
                "filter": ["lowercase", "edge_ngram_2_12"],
            },
            # Search-side analyzer for the prefix subfield: lowercase
            # only, NO edge-ngram. Otherwise the query "fra" would also
            # n-gram itself and match every doc containing "f" / "fr".
            "title_prefix_search": {
                "tokenizer": "standard",
                "filter": ["lowercase"],
            },
            "english_basic": {
                "tokenizer": "standard",
                "filter": ["lowercase", "english_stop", "english_stemmer"],
            },
            # v3 — universal multilingual analyzer (all 7 languages).
            "multilingual_icu": {
                "char_filter": ["icu_normalizer"],
                "tokenizer": "icu_tokenizer",
                "filter": ["lowercase", "icu_folding"],
            },
            # v3 — CJK/JA-aware typeahead. Index side edge-ngrams over
            # ICU word tokens; search side must mirror it WITHOUT the
            # edge-ngram (same rule as title_prefix_search).
            "icu_prefix_index": {
                "char_filter": ["icu_normalizer"],
                "tokenizer": "icu_tokenizer",
                "filter": ["lowercase", "icu_folding", "edge_ngram_2_12"],
            },
            "icu_prefix_search": {
                "char_filter": ["icu_normalizer"],
                "tokenizer": "icu_tokenizer",
                "filter": ["lowercase", "icu_folding"],
            },
            # v3 — dedicated Japanese morphological analyzer.
            "japanese_kuromoji": {
                "char_filter": ["icu_normalizer"],
                "tokenizer": "kuromoji_tokenizer",
                "filter": [
                    "kuromoji_baseform",
                    "kuromoji_part_of_speech",
                    "cjk_width",
                    "ja_stop",
                    "kuromoji_stemmer",
                    "lowercase",
                ],
            },
        },
    }


def build_mappings():
    dims = get_active_embedding_dimensions()
    return {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "entity_type": {"type": "keyword"},
            "entity_id": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},
            # Tenant + access control
            "team_id": {"type": "keyword"},
            "acl_user_ids": {"type": "keyword"},
            # Chat-specific identifiers (nullable for non-chat chunks).
            # Stored as keywords so the API can filter/group by them.
            "chat_type": {"type": "keyword"},
            "chat_id": {"type": "keyword"},
            "thread_id": {"type": "keyword"},
            # v2: chat-message identity. `author_id` enables an exact-
            # filter "what did Bob say" path without scanning bodies;
            # `author_name` is denormalized so source chips can render
            # the sender's name without a DB lookup. `chat_message_id`
            # is the per-message PK (DM/GM/PM/MDM messages or thread
            # messages depending on chunk_type) so citation chips can
            # deep-link to a specific bubble.
            "author_id": {"type": "keyword"},
            "author_name": {"type": "keyword"},
            "chat_message_id": {"type": "keyword"},
            # Task / Note identifiers (nullable per chunk type).
            "task_id": {"type": "keyword"},
            "note_id": {"type": "keyword"},
            "note_type": {"type": "keyword"},
            "project_id": {"type": "keyword"},
            # v2: task overlays — enable status/assignee filters and
            # priority-aware ranking without round-tripping to the DB.
            "task_status": {"type": "keyword"},
            "task_priority": {"type": "keyword"},
            "task_assignee_id": {"type": "keyword"},
            "task_milestone_id": {"type": "keyword"},
            "task_sprint_id": {"type": "keyword"},
            # v2: note overlays — "my notes about X" and parent-note
            # traversal without DB lookups.
            "note_owner_id": {"type": "keyword"},
            "note_parent_id": {"type": "keyword"},
            # Searchable text fields. `title` and `search_text` carry
            # v2 multi-fields for retrieval-quality tuning:
            #   title.prefix    — edge n-gram for fast Cmd-K typeahead
            #   search_text.en  — English analyzer for stemming/recall
            # The default subfield on each stays the standard analyzer,
            # so existing `multi_match` queries against the bare field
            # name keep their exact-phrase semantics.
            "title": {
                "type": "text",
                "fields": {
                    "prefix": {
                        "type": "text",
                        "analyzer": "title_prefix_index",
                        "search_analyzer": "title_prefix_search",
                    },
                    # v3 multilingual subfields (additive).
                    "icu": {"type": "text", "analyzer": "multilingual_icu"},
                    "ja": {"type": "text", "analyzer": "japanese_kuromoji"},
                    "icu_prefix": {
                        "type": "text",
                        "analyzer": "icu_prefix_index",
                        "search_analyzer": "icu_prefix_search",
                    },
                },
            },
            "search_text": {
                "type": "text",
                "fields": {
                    "en": {
                        "type": "text",
                        "analyzer": "english_basic",
                    },
                    # v3 multilingual subfields (additive).
                    "icu": {"type": "text", "analyzer": "multilingual_icu"},
                    "ja": {"type": "text", "analyzer": "japanese_kuromoji"},
                },
            },
            # snippet_text is boosted ^2 (highest-weight body field) — it
            # must not stay on the default standard analyzer for CJK.
            "snippet_text": {
                "type": "text",
                "fields": {
                    "icu": {"type": "text", "analyzer": "multilingual_icu"},
                    "ja": {"type": "text", "analyzer": "japanese_kuromoji"},
                },
            },
            # spotlight_answer lane only — stored-only provenance for the
            # "Previous answer" card. Not analyzed (search_text already carries
            # the Q+A); `index: false` keeps them out of the inverted index,
            # and `enabled: false` stops OpenSearch from parsing the nested
            # source objects. Additive fields: applied to a live index via
            # `opensearch_setup --update-mapping` (no recreate / re-embed).
            "answer_text": {"type": "text", "index": False},
            "answer_sources": {"type": "object", "enabled": False},
            # Vector for k-NN
            "embedding": {
                "type": "knn_vector",
                "dimension": dims,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                },
            },
            # Cross-entity relations. Set by the chunkers (task→chat,
            # note→parent-note, etc.) and projected into search results
            # as a fallback the frontend reads when chunk-side ids
            # don't carry enough context to build a deep link.
            "related_entity_ids": {"type": "keyword"},
            # Bookkeeping
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
            "text_hash": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "index_schema_version": {"type": "keyword"},
        }
    }
