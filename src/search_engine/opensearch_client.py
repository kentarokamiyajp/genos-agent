from django.conf import settings
from opensearchpy import OpenSearch

_client = None


def get_client():
    """Singleton OpenSearch client built from Django settings."""
    global _client
    if _client is not None:
        return _client

    cfg = settings.SEARCH_ENGINE
    _client = OpenSearch(
        hosts=[{"host": cfg["OPENSEARCH_HOST"], "port": cfg["OPENSEARCH_PORT"]}],
        http_compress=True,
        use_ssl=cfg["OPENSEARCH_USE_SSL"],
        verify_certs=cfg["OPENSEARCH_USE_SSL"],
        ssl_show_warn=False,
    )
    return _client


def get_index_alias():
    return settings.SEARCH_ENGINE["OPENSEARCH_ALIAS"]


def get_physical_index():
    return settings.SEARCH_ENGINE["OPENSEARCH_INDEX"]
