"""Agent tool registry.

The tools the agent can call. Each tool module exports a single `Tool`
instance; `REGISTRY` aggregates them by name so the controller can
dispatch a function-call to the right `run(...)`.

Phase 11 — write-tool surface expansion. Write tools flagged
`requires_approval=True` route through the pause/resume protocol from
Phase 7. Read-only tools execute inline.

Phase 13 — internal tool expansion. Seven new tools covering structured
queries and write operations that were previously impossible or required
fragile semantic search workarounds:
  Read (inline):
    list_projects, list_tasks, get_team_members,
    get_current_user, get_project_summary
  Write (requires_approval):
    assign_task, update_note

Phase 15 — aggregation/analytics surface. Five read-only tools that
return cross-project, time-ranged statistics so the model can answer
PM-style questions ("throughput last week", "top contributors", "which
project has the most notes") without enumerating individual records:
  Read (inline):
    get_task_throughput_stats, get_top_task_closers,
    get_project_activity_ranking, get_workload_distribution,
    get_stale_tasks

Phase 18 — me-scoped (user-specific) aggregation. Eight read-only
tools that always operate on `ctx.user_id`, so the agent can answer
"what kind of WIP tasks do I have?", "what should I handle first?",
"my schedule this week", "what milestones am I on?", "what's blocking
me?", "what did I close this week?", "what's in my inbox?", and "who
@mentioned me?" with a single call:
  Read (inline):
    get_my_task_summary, get_my_focus_tasks, get_my_schedule,
    list_my_milestones, list_my_inbox, list_my_mentions,
    get_my_blockers, get_my_throughput
Plus `list_milestones` gains an `assignee_user_id` filter for the
general "which milestones is Alice on?" case.
"""

from origin.search_engine.agent.tools.add_comment import ADD_COMMENT
from origin.search_engine.agent.tools.assign_task import ASSIGN_TASK
from origin.search_engine.agent.tools.base import REGISTRY, Tool, ToolContext, ToolError
from origin.search_engine.agent.tools.create_calendar_event import CREATE_CALENDAR_EVENT
from origin.search_engine.agent.tools.create_note import CREATE_NOTE
from origin.search_engine.agent.tools.create_task import CREATE_TASK
from origin.search_engine.agent.tools.create_task_plan import CREATE_TASK_PLAN
from origin.search_engine.agent.tools.create_todo_item import CREATE_TODO_ITEM
from origin.search_engine.agent.tools.delete_calendar_event import DELETE_CALENDAR_EVENT
from origin.search_engine.agent.tools.fetch_chat_thread import FETCH_CHAT_THREAD
from origin.search_engine.agent.tools.fetch_note import FETCH_NOTE
from origin.search_engine.agent.tools.fetch_pr import FETCH_PR
from origin.search_engine.agent.tools.fetch_task import FETCH_TASK
from origin.search_engine.agent.tools.get_current_user import GET_CURRENT_USER
from origin.search_engine.agent.tools.get_milestone_assignee_counts import (
    GET_MILESTONE_ASSIGNEE_COUNTS,
)
from origin.search_engine.agent.tools.get_milestone_summary import GET_MILESTONE_SUMMARY
from origin.search_engine.agent.tools.get_my_blockers import GET_MY_BLOCKERS
from origin.search_engine.agent.tools.get_my_focus_tasks import GET_MY_FOCUS_TASKS
from origin.search_engine.agent.tools.get_my_schedule import GET_MY_SCHEDULE
from origin.search_engine.agent.tools.get_my_task_summary import GET_MY_TASK_SUMMARY
from origin.search_engine.agent.tools.get_my_throughput import GET_MY_THROUGHPUT
from origin.search_engine.agent.tools.get_project_activity_ranking import (
    GET_PROJECT_ACTIVITY_RANKING,
)
from origin.search_engine.agent.tools.get_project_summary import GET_PROJECT_SUMMARY
from origin.search_engine.agent.tools.get_sprint_summary import GET_SPRINT_SUMMARY
from origin.search_engine.agent.tools.get_stale_tasks import GET_STALE_TASKS
from origin.search_engine.agent.tools.get_task_blockers import GET_TASK_BLOCKERS
from origin.search_engine.agent.tools.get_task_throughput_stats import (
    GET_TASK_THROUGHPUT_STATS,
)
from origin.search_engine.agent.tools.get_team_members import GET_TEAM_MEMBERS
from origin.search_engine.agent.tools.get_team_task_summary import GET_TEAM_TASK_SUMMARY
from origin.search_engine.agent.tools.get_top_task_closers import GET_TOP_TASK_CLOSERS
from origin.search_engine.agent.tools.get_workload_distribution import (
    GET_WORKLOAD_DISTRIBUTION,
)
from origin.search_engine.agent.tools.list_calendar_events import LIST_CALENDAR_EVENTS
from origin.search_engine.agent.tools.list_calendars import LIST_CALENDARS
from origin.search_engine.agent.tools.list_channel_members import LIST_CHANNEL_MEMBERS
from origin.search_engine.agent.tools.list_milestones import LIST_MILESTONES
from origin.search_engine.agent.tools.list_my_inbox import LIST_MY_INBOX
from origin.search_engine.agent.tools.list_my_mentions import LIST_MY_MENTIONS
from origin.search_engine.agent.tools.list_my_milestones import LIST_MY_MILESTONES
from origin.search_engine.agent.tools.list_note_folders import LIST_NOTE_FOLDERS
from origin.search_engine.agent.tools.list_pr_comments import LIST_PR_COMMENTS
from origin.search_engine.agent.tools.list_pr_commits import LIST_PR_COMMITS
from origin.search_engine.agent.tools.list_pr_files import LIST_PR_FILES
from origin.search_engine.agent.tools.list_pr_reviews import LIST_PR_REVIEWS
from origin.search_engine.agent.tools.list_project_members import LIST_PROJECT_MEMBERS
from origin.search_engine.agent.tools.list_projects import LIST_PROJECTS
from origin.search_engine.agent.tools.list_sprints import LIST_SPRINTS
from origin.search_engine.agent.tools.list_task_dependencies import LIST_TASK_DEPENDENCIES
from origin.search_engine.agent.tools.list_tasks import LIST_TASKS
from origin.search_engine.agent.tools.list_today_todos import LIST_TODAY_TODOS
from origin.search_engine.agent.tools.list_uncompleted_todos import LIST_UNCOMPLETED_TODOS
from origin.search_engine.agent.tools.search_kb import SEARCH_KNOWLEDGE_BASE
from origin.search_engine.agent.tools.search_past_conversations import (
    SEARCH_PAST_CONVERSATIONS,
)
from origin.search_engine.agent.tools.update_calendar_event import UPDATE_CALENDAR_EVENT
from origin.search_engine.agent.tools.update_note import UPDATE_NOTE
from origin.search_engine.agent.tools.update_task import UPDATE_TASK
from origin.search_engine.agent.tools.update_tasks_bulk import UPDATE_TASKS_BULK
from origin.search_engine.agent.tools.update_todo_item import UPDATE_TODO_ITEM
from origin.search_engine.agent.tools.web_search import SEARCH_WEB

# Register at import time so REGISTRY is populated by the time the
# controller asks for a tool by name. Read tools first, then write
# tools — the order only matters for any future iteration of REGISTRY
# (we'd want reads to surface first in tool-list dumps).
for _t in (
    # --- Read tools (Phase 1–11) ---
    SEARCH_KNOWLEDGE_BASE,
    # --- Conversation memory (Q2.3 / C1) ---
    SEARCH_PAST_CONVERSATIONS,
    FETCH_TASK,
    FETCH_CHAT_THREAD,
    FETCH_NOTE,
    LIST_NOTE_FOLDERS,
    # --- Read tools (Phase 13) ---
    LIST_PROJECTS,
    LIST_TASKS,
    GET_TEAM_MEMBERS,
    GET_CURRENT_USER,
    GET_PROJECT_SUMMARY,
    # --- Read tools — membership rosters (A1-ext) ---
    LIST_PROJECT_MEMBERS,
    LIST_CHANNEL_MEMBERS,
    # --- Write tools (Phase 11) ---
    CREATE_TASK,
    UPDATE_TASK,
    ADD_COMMENT,
    CREATE_NOTE,
    # --- Composite write tools (BAU Wave 1) — one approval per plan ---
    CREATE_TASK_PLAN,
    UPDATE_TASKS_BULK,
    # --- Write tools (Phase 13) ---
    ASSIGN_TASK,
    UPDATE_NOTE,
    # --- Read tools (Phase 14) ---
    SEARCH_WEB,
    # --- Read tools (Phase 15) — analytics/aggregation ---
    GET_TASK_THROUGHPUT_STATS,
    GET_TOP_TASK_CLOSERS,
    GET_PROJECT_ACTIVITY_RANKING,
    GET_WORKLOAD_DISTRIBUTION,
    GET_STALE_TASKS,
    # --- Read tools (Phase 17) — milestone, sprint & dependency aggregation ---
    GET_TEAM_TASK_SUMMARY,
    LIST_MILESTONES,
    GET_MILESTONE_SUMMARY,
    GET_MILESTONE_ASSIGNEE_COUNTS,
    LIST_SPRINTS,
    GET_SPRINT_SUMMARY,
    GET_TASK_BLOCKERS,
    # --- Read tools (BAU Wave 1) — whole-scope dependency graph ---
    LIST_TASK_DEPENDENCIES,
    # --- Read tools (Phase 16) — GitHub PR introspection ---
    FETCH_PR,
    LIST_PR_COMMENTS,
    LIST_PR_FILES,
    LIST_PR_REVIEWS,
    LIST_PR_COMMITS,
    # --- Read tools — Google Calendar ---
    LIST_CALENDARS,
    LIST_CALENDAR_EVENTS,
    # --- Write tools — Google Calendar (requires_approval) ---
    CREATE_CALENDAR_EVENT,
    UPDATE_CALENDAR_EVENT,
    DELETE_CALENDAR_EVENT,
    # --- Read tools (Phase 18) — me-scoped (user-specific) aggregation ---
    GET_MY_TASK_SUMMARY,
    GET_MY_FOCUS_TASKS,
    GET_MY_SCHEDULE,
    LIST_MY_MILESTONES,
    LIST_MY_INBOX,
    LIST_MY_MENTIONS,
    GET_MY_BLOCKERS,
    GET_MY_THROUGHPUT,
    # --- Todo tools ---
    LIST_TODAY_TODOS,
    LIST_UNCOMPLETED_TODOS,
    # --- Todo write tools (requires_approval) ---
    CREATE_TODO_ITEM,
    UPDATE_TODO_ITEM,
):
    REGISTRY[_t.name] = _t


__all__ = [
    "ADD_COMMENT",
    "ASSIGN_TASK",
    "CREATE_CALENDAR_EVENT",
    "CREATE_NOTE",
    "CREATE_TASK",
    "CREATE_TASK_PLAN",
    "CREATE_TODO_ITEM",
    "DELETE_CALENDAR_EVENT",
    "FETCH_CHAT_THREAD",
    "FETCH_NOTE",
    "LIST_NOTE_FOLDERS",
    "FETCH_PR",
    "FETCH_TASK",
    "GET_CURRENT_USER",
    "GET_MILESTONE_ASSIGNEE_COUNTS",
    "GET_MILESTONE_SUMMARY",
    "GET_MY_BLOCKERS",
    "GET_MY_FOCUS_TASKS",
    "GET_MY_SCHEDULE",
    "GET_MY_TASK_SUMMARY",
    "GET_MY_THROUGHPUT",
    "GET_PROJECT_ACTIVITY_RANKING",
    "GET_PROJECT_SUMMARY",
    "GET_SPRINT_SUMMARY",
    "GET_STALE_TASKS",
    "GET_TASK_BLOCKERS",
    "GET_TASK_THROUGHPUT_STATS",
    "GET_TEAM_MEMBERS",
    "GET_TEAM_TASK_SUMMARY",
    "GET_TOP_TASK_CLOSERS",
    "GET_WORKLOAD_DISTRIBUTION",
    "LIST_CALENDAR_EVENTS",
    "LIST_CALENDARS",
    "LIST_MILESTONES",
    "LIST_MY_INBOX",
    "LIST_MY_MENTIONS",
    "LIST_MY_MILESTONES",
    "LIST_PR_COMMENTS",
    "LIST_PR_COMMITS",
    "LIST_PR_FILES",
    "LIST_PR_REVIEWS",
    "LIST_PROJECTS",
    "LIST_SPRINTS",
    "LIST_TASK_DEPENDENCIES",
    "LIST_TASKS",
    "LIST_TODAY_TODOS",
    "LIST_UNCOMPLETED_TODOS",
    "REGISTRY",
    "SEARCH_KNOWLEDGE_BASE",
    "Tool",
    "ToolContext",
    "ToolError",
    "UPDATE_CALENDAR_EVENT",
    "UPDATE_NOTE",
    "UPDATE_TASK",
    "UPDATE_TASKS_BULK",
    "UPDATE_TODO_ITEM",
    "SEARCH_WEB",
]
