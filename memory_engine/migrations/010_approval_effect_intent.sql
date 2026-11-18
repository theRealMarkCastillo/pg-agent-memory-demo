-- Freeze a reviewed revision before crossing the non-transactional file boundary.
ALTER TABLE ext_proposals DROP CONSTRAINT ext_proposals_status_check;
ALTER TABLE ext_proposals ADD CONSTRAINT ext_proposals_status_check
    CHECK(status IN ('pending','approved','applying','rejected','cancelled','applied'));
ALTER TABLE ext_proposals ADD COLUMN effect_started_at timestamptz;
ALTER TABLE ext_proposals ADD CONSTRAINT ext_proposals_effect_intent_check
    CHECK(status <> 'applying' OR effect_started_at IS NOT NULL);
