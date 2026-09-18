from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from nexus.api.main import create_app
from nexus.config.settings import get_settings


@pytest.fixture
async def client(_memory_db_env: str) -> AsyncIterator[AsyncClient]:
    get_settings(refresh=True)
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_record_signal_returns_updated_state(client: AsyncClient) -> None:
    response = await client.post(
        "/api/personal/u1/signal",
        json={"dimension": "physical.energy", "value": 0.8, "source": "explicit"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "u1"
    assert body["dimensions"]["physical.energy"]["value"] == pytest.approx(0.8)
    assert body["dimensions"]["physical.energy"]["sample_count"] == 1


@pytest.mark.asyncio
async def test_record_signal_rejects_invalid_dimension(client: AsyncClient) -> None:
    response = await client.post(
        "/api/personal/u1/signal",
        json={"dimension": "not.a.real.dimension", "value": 0.5, "source": "explicit"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_state_matches_what_was_recorded(client: AsyncClient) -> None:
    await client.post(
        "/api/personal/u2/signal",
        json={"dimension": "mental.focus", "value": 0.6, "source": "explicit"},
    )

    response = await client.get("/api/personal/u2/state")

    assert response.status_code == 200
    assert response.json()["dimensions"]["mental.focus"]["value"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_full_round_trip_signal_state_baselines_weaknesses(client: AsyncClient) -> None:
    await client.post(
        "/api/personal/u3/signal",
        json={"dimension": "physical.energy", "value": 0.3, "source": "explicit"},
    )

    state_resp = await client.get("/api/personal/u3/state")
    assert state_resp.status_code == 200
    assert "physical.energy" in state_resp.json()["dimensions"]

    # The signal just recorded falls inside the recent window, which
    # baselines deliberately exclude — so no baseline exists yet for a
    # brand-new user, and consequently no weakness can be computed either.
    baselines_resp = await client.get("/api/personal/u3/baselines")
    assert baselines_resp.status_code == 200
    assert baselines_resp.json()["baselines"] == {}

    weaknesses_resp = await client.get("/api/personal/u3/weaknesses")
    assert weaknesses_resp.status_code == 200
    assert weaknesses_resp.json()["weaknesses"] == []


@pytest.mark.asyncio
async def test_profile_set_and_get_round_trip(client: AsyncClient) -> None:
    put_resp = await client.put(
        "/api/personal/u4/profile", json={"profile": {"primary_goal": "improve endurance"}}
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["profile"] == {"primary_goal": "improve endurance"}

    get_resp = await client.get("/api/personal/u4/profile")
    assert get_resp.status_code == 200
    assert get_resp.json()["profile"] == {"primary_goal": "improve endurance"}


@pytest.mark.asyncio
async def test_delete_removes_all_signals_and_profile_for_the_user(client: AsyncClient) -> None:
    await client.post(
        "/api/personal/u5/signal",
        json={"dimension": "physical.energy", "value": 0.5, "source": "explicit"},
    )
    await client.put("/api/personal/u5/profile", json={"profile": {"goal": "x"}})

    delete_resp = await client.delete("/api/personal/u5")
    assert delete_resp.status_code == 204

    state_resp = await client.get("/api/personal/u5/state")
    assert state_resp.json()["dimensions"] == {}

    profile_resp = await client.get("/api/personal/u5/profile")
    assert profile_resp.json()["profile"] == {}


@pytest.mark.asyncio
async def test_delete_does_not_affect_other_users(client: AsyncClient) -> None:
    await client.post(
        "/api/personal/u6/signal",
        json={"dimension": "physical.energy", "value": 0.5, "source": "explicit"},
    )
    await client.post(
        "/api/personal/u7/signal",
        json={"dimension": "physical.energy", "value": 0.7, "source": "explicit"},
    )

    await client.delete("/api/personal/u6")

    u7_state = await client.get("/api/personal/u7/state")
    assert "physical.energy" in u7_state.json()["dimensions"]
