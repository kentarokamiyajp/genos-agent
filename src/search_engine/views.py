"""REST API for the chunk-based hybrid search engine.

Single endpoint:

    POST /api/v2/search/

Request body (JSON):
    {
      "query":         "payment retry failure",          # required
      "team_id":       "<team uuid>",                     # required
      "entity_types":  ["chat","task","note"],            # optional, default all
      "date_from":     "2026-01-01T00:00:00Z",            # optional
      "date_to":       "2026-05-15T00:00:00Z",            # optional
      "limit":         20,                                # optional, default 20
      "use_vector":    true                               # optional, default true
    }

Authenticated `request.user.id` is used as the ACL filter — clients
do not need to pass user_id explicitly.
"""

from rest_framework import status
from rest_framework.response import Response

from origin.search_engine.search import search
from origin.views.common.base_auth_api_view import AuthenticatedAPIView


class SearchView(AuthenticatedAPIView):
    def post(self, request):
        data = request.data or {}

        query = (data.get("query") or "").strip()
        team_id = data.get("team_id")

        if not query:
            return Response(
                {"error": "query is required and must be non-empty."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not team_id:
            return Response(
                {"error": "team_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user_id = str(getattr(request.user, "id", "")) or data.get("user_id")
        if not user_id:
            return Response(
                {"error": "Could not determine user_id from the auth token."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        entity_types = data.get("entity_types") or None
        if entity_types is not None and not isinstance(entity_types, list):
            return Response(
                {"error": "entity_types must be a list of strings."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            limit = int(data.get("limit", 20))
        except (TypeError, ValueError):
            return Response(
                {"error": "limit must be an integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        use_vector = bool(data.get("use_vector", True))

        # Optional relevance filters. Frontend can override the
        # backend defaults per call (e.g. set min_score_ratio=0 to
        # disable when an admin debug UI wants to see the long tail).
        min_score_ratio = data.get("min_score_ratio")
        min_score = data.get("min_score")
        extra_kwargs: dict = {}
        if min_score_ratio is not None:
            try:
                extra_kwargs["min_score_ratio"] = float(min_score_ratio)
            except (TypeError, ValueError):
                return Response(
                    {"error": "min_score_ratio must be a number."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        if min_score is not None:
            try:
                extra_kwargs["min_score"] = float(min_score)
            except (TypeError, ValueError):
                return Response(
                    {"error": "min_score must be a number."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        result = search(
            query=query,
            team_id=str(team_id),
            user_id=user_id,
            entity_types=entity_types,
            date_from=data.get("date_from"),
            date_to=data.get("date_to"),
            limit=limit,
            use_vector=use_vector,
            **extra_kwargs,
        )
        return Response(result, status=status.HTTP_200_OK)
