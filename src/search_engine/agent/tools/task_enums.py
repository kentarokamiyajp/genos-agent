"""Shared task field enums for agent tools.

Canonical enum lives on the frontend in `taskMeta.ts`. Keep these in
sync — the priority and effort-level sets are read by chip colour
lookups, and an out-of-set value renders without a colour.

`create_task.py` / `update_task.py` predate this module and carry their
own private copies; new tools import from here so a future enum change
is a one-file edit. (Existing tools are left untouched to avoid churn —
consolidate opportunistically.)
"""

VALID_PRIORITIES = {"Minimal", "Low", "Normal", "High", "Critical"}
VALID_EFFORTS = {"Minimal", "Low", "Moderate", "High", "Extensive"}
# "Deleted" is deliberately absent: bulk/plan tools must not soft-delete
# tasks as a side effect of an organize pass. `update_task` still allows
# it for a single, explicitly-named task.
VALID_STATUSES = {"Open", "WIP", "Pending", "Closed"}

PRIORITY_ENUM = sorted(VALID_PRIORITIES)
EFFORT_ENUM = sorted(VALID_EFFORTS)
STATUS_ENUM = sorted(VALID_STATUSES)
