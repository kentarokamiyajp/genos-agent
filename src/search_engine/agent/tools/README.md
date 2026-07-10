# Agent tools — adding a new tool (Definition of Done)

The contract lives in [`base.py`](base.py) (`Tool`, `ToolContext`,
`ToolError`, `wrap_workspace_content`); registration in
[`__init__.py`](__init__.py) → `REGISTRY`. Generic declaration gates run
automatically over every registered tool in
`origin/tests/test_agent_tool_registry.py` — the checklist below is what
you must add **per tool**. Rationale and the broader release-safety plan:
genos-docs `spotlight/SPOTLIGHT_AGENT_CHANGE_SAFETY.md` (§4.2).

Every new-tool PR ships with:

- [ ] **ACL test** — call `run()` with a foreign `team_id` / non-member
  `user_id` in `ToolContext`; assert `ToolError` (or an empty result), never
  another team's rows. Authorization comes from `ctx`, **never** from LLM
  args — the model must not be able to escalate by passing different ids.
  Pattern to copy: `origin/tests/test_agent_tools_membership.py`.

- [ ] **Injection boundary** — if the tool returns user-authored free text
  (chat messages, note bodies, task descriptions), wrap each piece with
  `wrap_workspace_content(...)` and assert the wrap in a test. Unwrapped
  user text reopens the prompt-injection hole the boundary exists to close
  (the system prompt treats `<workspace_content>` as data, not instructions).

- [ ] **`requires_approval=True` if it writes** — the controller's
  pause/approve protocol keys on this one flag; a mis-flagged write tool
  mutates data with **no user approval step**. Enforced generically for
  `create_` / `update_` / `delete_` / `assign_` / `add_`-prefixed names;
  a new write *verb* must be added to `WRITE_PREFIXES` in
  `origin/tests/test_agent_tool_registry.py` (that test tells you when).

- [ ] **Quota / tier accounting** — external or per-call-expensive tools
  (network APIs, LLM-heavy work) are wired into the tier quotas
  (`SEARCH_ENGINE["TIER_QUOTAS"]` in `apis/settings.py`), not left
  uncounted. The web tool (`search_web`, [`web_search.py`](web_search.py) /
  `web_search_daily`) is the reference.

- [ ] **At least one eval case** — a behavior case in
  [`../evals/cases.yaml`](../evals/cases.yaml) exercising the tool
  (eval-first is the house rule — genos-docs
  `spotlight/SPOTLIGHT_EVALS_CI.md` §6.1).

- [ ] **Description says *when*, not just what** — the model chooses among
  ~50 tools by description alone (see the `Tool.description` docstring in
  [`base.py`](base.py)).

Rollout / rollback: a risky tool can ship dark and be switched off without
a redeploy-of-code via the fail-open `AGENT_DISABLED_TOOLS` kill-switch —
see `SPOTLIGHT_AGENT_CHANGE_SAFETY.md` §4.4.
