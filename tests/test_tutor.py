"""Tests for Tutor: skill tree, forgetting-curve decay, gap detection."""
import pytest
import asyncio
from conftest import post, get

USER_ID = "tutor_test_user"


@pytest.fixture(autouse=True)
async def seed_skills(client):
    await post(client, "/tutor/skills", skill_name="math_basics")
    await post(client, "/tutor/skills", skill_name="algebra", parent_skill_name="math_basics")
    await post(client, "/tutor/skills", skill_name="calculus", parent_skill_name="algebra")
    await post(client, "/tutor/skills", skill_name="statistics", parent_skill_name="math_basics")
    await post(client, "/tutor/skills", skill_name="python_basics")
    await post(client, "/tutor/progress",
        user_id=USER_ID, skill_name="math_basics", proficiency_score=0.9)
    await post(client, "/tutor/progress",
        user_id=USER_ID, skill_name="algebra", proficiency_score=0.4)
    await post(client, "/tutor/progress",
        user_id=USER_ID, skill_name="calculus", proficiency_score=0.1)
    await post(client, "/tutor/progress",
        user_id=USER_ID, skill_name="python_basics", proficiency_score=0.75)


@pytest.mark.asyncio
async def test_all_skills_present(client):
    data = await get(client, f"/tutor/gaps/{USER_ID}")
    names = {r["skill_name"] for r in data}
    expected = {"math_basics", "algebra", "calculus", "statistics", "python_basics"}
    assert expected.issubset(names), f"Missing: {expected - names}"


@pytest.mark.asyncio
async def test_decayed_scores_below_original(client):
    data = await get(client, f"/tutor/gaps/{USER_ID}")
    score_map = {r["skill_name"]: r["decayed_score"] for r in data}
    assert score_map["math_basics"] <= 0.9
    assert score_map["algebra"] == pytest.approx(0.4, rel=1e-5)
    assert score_map["calculus"] == pytest.approx(0.1, rel=1e-5)


@pytest.mark.asyncio
async def test_gap_threshold_classification(client):
    data = await get(client, f"/tutor/gaps/{USER_ID}")
    for r in data:
        if r["skill_name"] in ("algebra", "calculus"):
            assert r["status"] == "gap"
        if r["skill_name"] == "math_basics":
            assert r["status"] == "mastered"


@pytest.mark.asyncio
async def test_unattempted_skill_is_gap(client):
    data = await get(client, f"/tutor/gaps/{USER_ID}")
    stats = next(r for r in data if r["skill_name"] == "statistics")
    assert stats["decayed_score"] == 0.0
    assert stats["status"] == "gap"


@pytest.mark.asyncio
async def test_progress_update_increases_score(client):
    await post(client, "/tutor/progress",
        user_id=USER_ID, skill_name="calculus", proficiency_score=0.8)
    data = await get(client, f"/tutor/gaps/{USER_ID}")
    calc = next(r for r in data if r["skill_name"] == "calculus")
    assert calc["decayed_score"] >= 0.7
    assert calc["status"] == "mastered"


@pytest.mark.asyncio
async def test_decay_after_delay(client):
    initial = (await get(client, f"/tutor/gaps/{USER_ID}"))
    py_initial = next(r["decayed_score"] for r in initial if r["skill_name"] == "python_basics")
    await asyncio.sleep(2)
    later = (await get(client, f"/tutor/gaps/{USER_ID}"))
    py_later = next(r["decayed_score"] for r in later if r["skill_name"] == "python_basics")
    assert py_later <= py_initial, f"Score should decay: {py_initial} -> {py_later}"
