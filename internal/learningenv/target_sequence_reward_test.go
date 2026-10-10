package learningenv

import (
	"q2coopbot/internal/policy"
	"testing"
)

func sequenceFixture(index, target int) (RewardConfig, Step, ServerOutcome) {
	c, s, o := qualityFixture()
	c.Version, c.TargetChurn = TargetSequenceRewardVersion, -.02
	s.Owner = "provider"
	s.Index = index + 1
	s.Observation.Identity.Frame = 10 + index
	s.Next.Identity.Frame = 11 + index
	s.Action.Identity, s.AppliedAction.Identity = s.Observation.Identity, s.Observation.Identity
	s.Native.BeginFrame, s.Native.EndFrame = 10+index, 11+index
	s.Observation.Health, s.Next.Health = 100, 100
	o.Step, o.MonsterHealthDamage, o.ReceivedHealthDamage = index+1, 0, 0
	e := s.Observation.Enemies[0]
	e.ID = 8
	s.Observation.Enemies = append(s.Observation.Enemies, e)
	s.Next.Enemies = s.Observation.Enemies
	s.Action.TargetEntity = target
	if index == 0 {
		s.Observation.PreviousTarget = nil
	} else {
		previous := 7
		if index%2 == 0 {
			previous = 8
		}
		s.Observation.PreviousTarget = &policy.TargetIntent{Entity: previous, Track: 1}
		s.AppliedAction.YawDelta = 50
	}
	return c, s, o
}

func TestTargetSequenceRewardCycleAndEvidence(t *testing.T) {
	var q SequenceReward
	for i, target := range []int{7, 8, 7} {
		c, s, o := sequenceFixture(i, target)
		q.Config = c
		r := q.Evaluate(&s, o)
		if !r.Available {
			t.Fatal(r)
		}
		if i < 2 && r.Components["target_churn"] != 0 {
			t.Fatal("first switch charged", r)
		}
		if i == 2 && (r.Components["target_churn"] != -.02 || r.TargetCycle == nil || r.TargetCycle.Entity != 7 || r.TargetCycle.IntermediateEntity != 8) {
			t.Fatal(r)
		}
	}
}

func TestTargetSequenceRewardMasksUsefulOrUnprovenCycles(t *testing.T) {
	for _, reason := range []string{"damage", "injury", "new_threat", "next_threat", "lost_target", "handoff", "gap", "new_world", "slow_return", "tiny_turn", "intent_mismatch"} {
		t.Run(reason, func(t *testing.T) {
			var q SequenceReward
			for i, target := range []int{7, 8, 7} {
				c, s, o := sequenceFixture(i, target)
				q.Config = c
				if i == 1 {
					switch reason {
					case "damage":
						o.MonsterHealthDamage = 1
					case "injury":
						o.ReceivedHealthDamage = 1
					case "new_threat":
						e := s.Observation.Enemies[0]
						e.ID = 9
						s.Observation.Enemies = append(s.Observation.Enemies, e)
					case "lost_target":
						s.Next.Enemies = s.Next.Enemies[1:]
					case "next_threat":
						e := s.Next.Enemies[0]
						e.ID = 9
						s.Next.Enemies = append(s.Next.Enemies, e)
					case "handoff":
						s.Truncated, s.Reason = true, "control_handoff"
					case "gap":
						s.Execution.Matched = false
					case "new_world":
						s.Observation.Identity.Connection++
						s.Next.Identity.Connection++
						s.Action.Identity = s.Observation.Identity
						s.AppliedAction.Identity = s.Observation.Identity
					case "slow_return":
						q.runs[0].end = -100
					case "tiny_turn":
						s.AppliedAction.YawDelta = 1
					case "intent_mismatch":
						s.Observation.PreviousTarget = nil
					}
				}
				if r := q.Evaluate(&s, o); r.Components["target_churn"] != 0 || r.TargetCycle != nil {
					t.Fatal(reason, r)
				}
			}
		})
	}
}

func TestTargetSequenceRewardRequiresContextAndKeepsLegacy(t *testing.T) {
	c, s, o := sequenceFixture(0, 7)
	if r := c.Evaluate(&s, o); r.Available || r.Reason != "sequence_context_required" {
		t.Fatal(r)
	}
	c.TargetChurn = -.2
	if c.Validate() == nil {
		t.Fatal("unbounded cost accepted")
	}
	c.Version, c.TargetChurn = ActionQualityRewardVersion, 0
	q := SequenceReward{Config: c}
	if r := q.Evaluate(&s, o); !r.Available || r.Components["target_churn"] != 0 || r.TargetCycle != nil {
		t.Fatal(r)
	}
	// A missing current target is not an A-B-A event.
	s.Action.TargetEntity = 0
	q.Config.Version, q.Config.TargetChurn = TargetSequenceRewardVersion, -.02
	if r := q.Evaluate(&s, o); !r.Available || r.TargetCycle != nil {
		t.Fatal(r)
	}
}
