"""Optional FastAPI application for a self-hosted Hive aggregator."""

from __future__ import annotations

from typing import Any, Protocol

from .aggregator import HiveAggregator, HiveAggregatorError
from .types import ContributionEnvelope


class HiveRequestAuthorizer(Protocol):
    def __call__(self, authorization: str | None, operation: str) -> str | None: ...


def create_hive_app(
    aggregator: HiveAggregator,
    *,
    api_keys: set[str] | None = None,
    authorizer: HiveRequestAuthorizer | None = None,
) -> Any:
    """Create the reference service without importing FastAPI at SDK import time.

    ``api_keys=None`` is intended only for a loopback deployment.  Supplying an
    empty set denies every request; supplying keys enables bearer auth.  A
    custom authorizer can map a request to a team actor and enforce richer
    tenancy rules before the aggregator's domain authorizer runs.
    """

    try:
        from fastapi import FastAPI, Header, HTTPException, Query
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Hive service requires `pip install synapsekit[hive]`") from exc

    app = FastAPI(title="SynapseKit Hive", version="1.0")

    def actor_for(authorization: str | None, operation: str) -> str | None:
        if authorizer is not None:
            return authorizer(authorization, operation)
        if api_keys is None:
            return "local"
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer authentication required")
        token = authorization.removeprefix("Bearer ").strip()
        if token not in api_keys:
            raise HTTPException(status_code=403, detail="invalid Hive API key")
        return token

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "schema_version": "1.0"}

    @app.post("/v1/contributions")
    def contribute(
        envelope: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, str]:
        actor_for(authorization, "contribute")
        try:
            contribution = ContributionEnvelope.from_dict(envelope)
            contribution_id = aggregator.submit(
                contribution, actor=actor_for(authorization, "contribute")
            )
        except (HiveAggregatorError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"contribution_id": contribution_id}

    @app.get("/v1/suggestions")
    def suggestions(
        scope_id: str = Query(min_length=1, max_length=128),
        query: str | None = Query(default=None, max_length=128),
        minimum_cohort: int = Query(default=3, ge=1, le=100_000),
        limit: int = Query(default=20, ge=1, le=100),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        actor_for(authorization, "suggestions")
        try:
            values = aggregator.suggestions(
                scope_id=scope_id, query=query, minimum_cohort=minimum_cohort, limit=limit
            )
        except (HiveAggregatorError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"scope_id": scope_id, "suggestions": [item.to_dict() for item in values]}

    @app.post("/v1/withdraw")
    def withdraw(
        body: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, int]:
        actor = actor_for(authorization, "withdraw")
        try:
            count = aggregator.withdraw(
                contributor_id=str(body["contributor_id"]),
                scope_id=str(body["scope_id"]),
                actor=actor,
            )
        except (HiveAggregatorError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"revoked": count}

    @app.get("/v1/transparency")
    def transparency(
        contributor_id: str = Query(min_length=1, max_length=128),
        scope_id: str = Query(min_length=1, max_length=128),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        actor_for(authorization, "transparency")
        try:
            return aggregator.transparency(
                contributor_id=contributor_id, scope_id=scope_id
            ).to_dict()
        except (HiveAggregatorError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/dashboard")
    def dashboard(authorization: str | None = Header(default=None)) -> dict[str, str]:
        actor_for(authorization, "dashboard")
        return {
            "name": "SynapseKit Hive transparency dashboard",
            "purpose": "Inspect aggregate-only, DP-processed contribution reports through /v1/transparency.",
            "privacy": "Raw memory content is never accepted by the service.",
        }

    return app
