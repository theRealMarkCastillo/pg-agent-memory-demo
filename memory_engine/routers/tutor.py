from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter()


class SkillCreate(BaseModel):
    skill_name: str
    parent_skill_name: Optional[str] = None


class UserProgressUpdate(BaseModel):
    user_id: str
    skill_name: str
    proficiency_score: float = Field(ge=0, le=1)


class SkillGapSearch(BaseModel):
    user_id: str


@router.post("/skills")
async def create_skill(skill: SkillCreate, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        parent_id = None
        if skill.parent_skill_name:
            row = await conn.fetchrow(
                "SELECT skill_id FROM tutor_skills WHERE skill_name = $1",
                skill.parent_skill_name,
            )
            if not row:
                raise HTTPException(status_code=404, detail="Parent skill not found")
            parent_id = row["skill_id"]

        await conn.execute(
            "INSERT INTO tutor_skills (skill_name, parent_skill_id) VALUES ($1, $2) "
            "ON CONFLICT (skill_name) DO NOTHING",
            skill.skill_name,
            parent_id,
        )

    return {"status": "created"}


@router.post("/progress")
async def update_progress(progress: UserProgressUpdate, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        skill = await conn.fetchrow(
            "SELECT skill_id FROM tutor_skills WHERE skill_name = $1",
            progress.skill_name,
        )
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")

        await conn.execute(
            """
            INSERT INTO tutor_user_progress (user_id, skill_id, proficiency_score, last_reviewed_at)
            VALUES ($1, $2, $3, clock_timestamp())
            ON CONFLICT (user_id, skill_id)
            DO UPDATE SET proficiency_score = $3, last_reviewed_at = clock_timestamp()
            """,
            progress.user_id,
            skill["skill_id"],
            progress.proficiency_score,
        )

    return {"status": "updated"}


@router.get("/gaps/{user_id}")
async def find_skill_gaps(user_id: str, request: Request):
    pool = request.app.state.pool

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH RECURSIVE skill_with_decay AS (
                SELECT s.skill_id, s.skill_name, s.parent_skill_id,
                       COALESCE(up.proficiency_score, 0.0) *
                       POWER(0.95, EXTRACT(EPOCH FROM (clock_timestamp() - COALESCE(up.last_reviewed_at, clock_timestamp()))) / 86400.0)
                       AS decayed_score
                FROM tutor_skills s
                LEFT JOIN tutor_user_progress up ON s.skill_id = up.skill_id AND up.user_id = $1
            )
            , ancestors AS (
                SELECT skill_id AS child_id,parent_skill_id AS ancestor_id FROM tutor_skills WHERE parent_skill_id IS NOT NULL
                UNION
                SELECT a.child_id,s.parent_skill_id FROM ancestors a JOIN tutor_skills s ON s.skill_id=a.ancestor_id WHERE s.parent_skill_id IS NOT NULL
            )
            SELECT d.skill_name,d.decayed_score,p.skill_name AS prerequisite,
                   CASE WHEN d.decayed_score < 0.5 THEN 'gap' ELSE 'mastered' END AS status,
                   NOT EXISTS(SELECT 1 FROM ancestors a JOIN skill_with_decay x ON x.skill_id=a.ancestor_id WHERE a.child_id=d.skill_id AND x.decayed_score<0.5) AS ready
            FROM skill_with_decay d LEFT JOIN tutor_skills p ON p.skill_id=d.parent_skill_id
            ORDER BY ready DESC,d.decayed_score ASC,d.skill_name
            """,
            user_id,
        )

    return [dict(r) for r in rows]
