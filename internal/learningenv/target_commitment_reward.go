package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"strings"
)

// Offline only. The budget belongs to an episode/life, not a target or run.
type commitmentState struct {
	world           policy.Identity
	next            policy.Identity
	worker, episode string
	index           int
	key             targetKey
	frames          int
	spent           float64
	visible         map[targetKey]bool
}

type TargetCommitmentEvidence struct {
	Entity               int     `json:"entity"`
	Track                int     `json:"track"`
	ConsecutiveFrames    int     `json:"consecutive_frames"`
	SelectedHealthDamage int     `json:"selected_health_damage"`
	Bonus                float64 `json:"bonus"`
	BudgetSpent          float64 `json:"budget_spent"`
	Reason               string  `json:"reason"`
}

func (q *SequenceReward) commitmentReward(s *Step, o ServerOutcome, r Reward) Reward {
	state := &q.commitment
	world := s.Observation.Identity
	world.Frame = 0
	if state.world != world || state.worker != s.Worker || state.episode != s.Episode {
		*state = commitmentState{world: world, worker: s.Worker, episode: s.Episode}
	}
	if !r.Available {
		state.frames = 0
		state.next = policy.Identity{}
		return r
	}
	evidence := &TargetCommitmentEvidence{Entity: s.Action.TargetEntity, Track: s.Action.TargetTrack, BudgetSpent: state.spent, Reason: "no_productive_hold"}
	r.TargetCommitment = evidence
	r.Components["target_commitment"] = 0
	key := targetKey{s.Action.TargetEntity, s.Action.TargetTrack}
	enemy, exists := matchingTarget(s.Observation, key.entity, key.track)
	valid := s.Owner == "provider" && key.entity > 0 && key.track > 0 && exists && enemy.ClearShot != nil && *enemy.ClearShot
	if !valid || s.Truncated || (s.Terminal && s.Reason != "combat_goal_complete") {
		state.frames = 0
		state.next = policy.Identity{}
		return r
	}
	previous := s.Observation.PreviousTarget
	previousIdentity := s.Observation.Identity
	previousIdentity.Frame--
	continuous := state.next == s.Observation.Identity && state.index+1 == s.Index && state.key == key && previous != nil && previous.Entity == key.entity && previous.Track == key.track && previous.Identity == previousIdentity
	current := visibleTargets(s.Observation)
	if len(current) != len(state.visible) {
		continuous = false
	}
	for k := range current {
		if !state.visible[k] {
			continuous = false
		}
	}
	state.visible = current
	if continuous {
		state.frames++
	} else {
		state.frames = 1
	}
	state.key, state.next, state.index = key, s.Next.Identity, s.Index
	evidence.ConsecutiveFrames = state.frames
	if q.Config.TargetCommitmentDamage == 0 {
		evidence.Reason = "bonus_disabled"
		return r
	}
	total, selected := 0, 0
	for _, e := range o.Events {
		if e.Attacker != s.Observation.Identity.Actor || !strings.HasPrefix(e.TargetClass, "monster_") {
			continue
		}
		if e.Target <= 0 || e.Target == e.Attacker || e.Map != world.Map || e.Spawncount != world.Spawncount || e.Frame < s.Native.BeginFrame || e.Frame > s.Native.EndFrame || e.Take < 0 || e.HealthBefore-e.Take != e.HealthAfter {
			r.Available, r.Score, r.Components, r.Reason = false, nil, nil, "invalid_commitment_damage_evidence"
			state.frames = 0
			state.next = policy.Identity{}
			return r
		}
		damage := min(max(e.HealthBefore, 0), e.Take)
		total += damage
		if e.Target == key.entity {
			// The client model decoder exposes all soldier skins as soldier.
			// Keep native identity/window checks; only reconcile this known alias.
			sameClass := enemy.Class == e.TargetClass || enemy.Class == "monster_soldier" &&
				(e.TargetClass == "monster_soldier_light" || e.TargetClass == "monster_soldier_ss")
			if !sameClass {
				r.Available, r.Score, r.Components, r.Reason = false, nil, nil, "commitment_target_class_mismatch"
				state.frames = 0
				state.next = policy.Identity{}
				return r
			}
			selected += damage
		}
	}
	if total != o.MonsterHealthDamage {
		r.Available, r.Score, r.Components, r.Reason = false, nil, nil, "commitment_damage_total_mismatch"
		state.frames = 0
		state.next = policy.Identity{}
		return r
	}
	evidence.SelectedHealthDamage = selected
	if state.frames < 3 || selected == 0 {
		return r
	}
	bonus := math.Min(float64(selected)*q.Config.TargetCommitmentDamage, math.Max(0, q.Config.TargetCommitmentBudget-state.spent))
	state.spent += bonus
	evidence.Bonus, evidence.BudgetSpent = bonus, state.spent
	evidence.Reason = "productive_observed_hold"
	if bonus == 0 {
		evidence.Reason = "episode_budget_exhausted"
	}
	r.Components["target_commitment"] = bonus
	*r.Score += bonus
	return r
}
