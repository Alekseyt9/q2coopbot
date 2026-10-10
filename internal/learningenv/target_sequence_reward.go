package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
)

// SequenceReward owns only offline reward history. It changes no policy inputs,
// chooses no target, and must be scoped to one monotonically exported episode.
type SequenceReward struct {
	Config          RewardConfig
	runs            []targetRun
	next            policy.Identity
	index           int
	worker, episode string
	visible         map[targetKey]bool
}

type targetKey struct{ entity, track int }
type targetRun struct {
	key        targetKey
	start, end int
	turn       float64
}

type TargetCycleEvidence struct {
	Entity             int     `json:"entity"`
	Track              int     `json:"track"`
	IntermediateEntity int     `json:"intermediate_entity"`
	IntermediateTrack  int     `json:"intermediate_track"`
	LeaveFrame         int     `json:"leave_frame"`
	ReturnFrame        int     `json:"return_frame"`
	Scope              string  `json:"scope"`
	IntermediateTurn   float64 `json:"intermediate_turn_degrees"`
	ReturnTurn         float64 `json:"return_turn_degrees"`
}

func (q *SequenceReward) reset() { q.runs = nil; q.visible = nil; q.next = policy.Identity{} }

func visibleTargets(o policy.Observation) map[targetKey]bool {
	m := map[targetKey]bool{}
	for _, e := range o.Enemies {
		if e.Track != nil && *e.Track > 0 && e.ClearShot != nil && *e.ClearShot {
			m[targetKey{e.ID, *e.Track}] = true
		}
	}
	return m
}

func (q *SequenceReward) Evaluate(s *Step, outcome ServerOutcome) Reward {
	r := q.Config.evaluate(s, outcome)
	if q.Config.Version != TargetSequenceRewardVersion {
		return r
	}
	if !r.Available {
		q.reset()
		return r
	}
	r.Components["target_churn"] = 0
	if s.Truncated || s.Terminal || s.Owner != "provider" || s.Action.TargetEntity <= 0 || s.Action.TargetTrack <= 0 {
		q.reset()
		return r
	}
	if q.next != s.Observation.Identity || q.index+1 != s.Index || q.worker != s.Worker || q.episode != s.Episode {
		q.reset()
	}
	current := visibleTargets(s.Observation)
	after := visibleTargets(*s.Next)
	key := targetKey{s.Action.TargetEntity, s.Action.TargetTrack}
	if len(q.runs) > 0 && (s.Observation.PreviousTarget == nil ||
		targetKey{s.Observation.PreviousTarget.Entity, s.Observation.PreviousTarget.Track} != q.runs[len(q.runs)-1].key) {
		q.runs = nil
	}
	if !current[key] || !after[key] {
		q.reset()
		return r
	}
	// Progress, injury, a newly available threat, or a disappearing target can
	// justify a switch. Do not label a cycle across any of these boundaries.
	changed := false
	if len(current) != len(after) {
		changed = true
	}
	for k := range after {
		if !current[k] {
			changed = true
		}
	}
	if q.visible != nil {
		if len(current) != len(q.visible) {
			changed = true
		}
		for k := range current {
			if !q.visible[k] {
				changed = true
			}
		}
	}
	if changed || outcome.MonsterHealthDamage > 0 || outcome.MonsterKills > 0 ||
		outcome.ReceivedHealthDamage > 0 || s.Next.Health < s.Observation.Health || s.Next.Armor < s.Observation.Armor {
		q.runs = nil
	}
	for _, run := range q.runs {
		if !current[run.key] || !after[run.key] {
			q.runs = nil
			break
		}
	}
	frame := s.Observation.Identity.Frame
	turn := math.Hypot(s.AppliedAction.YawDelta, s.AppliedAction.PitchDelta)
	if len(q.runs) == 0 {
		q.runs = []targetRun{{key, frame, frame, turn}}
	} else if q.runs[len(q.runs)-1].key == key {
		q.runs[len(q.runs)-1].end = frame
		q.runs[len(q.runs)-1].turn += turn
	} else {
		if len(q.runs) == 2 && q.runs[0].key == key && frame-q.runs[0].end <= 10 && q.runs[1].turn >= 30 && turn >= 30 {
			r.Components["target_churn"] = q.Config.TargetChurn
			*r.Score += q.Config.TargetChurn
			r.TargetCycle = &TargetCycleEvidence{Entity: key.entity, Track: key.track,
				IntermediateEntity: q.runs[1].key.entity, IntermediateTrack: q.runs[1].key.track,
				LeaveFrame: q.runs[0].end, ReturnFrame: frame,
				IntermediateTurn: q.runs[1].turn, ReturnTurn: turn,
				Scope: "Observed A-B-A within10 ticks; both targets remained clear, no intervening native health damage or kill. Soft consistency cost, not proof that the switch was useless; delayed projectile effects may follow."}
		}
		q.runs = append(q.runs, targetRun{key, frame, frame, turn})
		if len(q.runs) > 2 {
			q.runs = q.runs[1:]
		}
	}
	q.next, q.index, q.worker, q.episode, q.visible = s.Next.Identity, s.Index, s.Worker, s.Episode, after
	return r
}
