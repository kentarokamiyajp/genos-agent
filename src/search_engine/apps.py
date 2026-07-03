from django.apps import AppConfig


class SearchEngineConfig(AppConfig):
    name = "origin.search_engine"
    label = "search_engine"
    default_auto_field = "django.db.models.BigAutoField"
    verbose_name = "Search Engine"
