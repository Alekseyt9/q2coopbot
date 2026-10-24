package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"testing"
)

func commitmentFixture(index, target, damage int) (RewardConfig, Step, ServerOutcome) {
	c, s, o := sequenceFixture(index, target)
	c.Version = TargetCommitmentRewardVersion
	c.TargetCommitmentDamage, c.TargetCommitmentBudget = .002, .05
	for i := range s.Observation.Enemies {
		s.Observation.Enemies[i].Class = "monster_parasite"
	}
	s.Next.Enemies = s.Observation.Enemies
	id := s.Observation.Identity
	id.Frame--
	s.Observation.PreviousTarget = &policy.TargetIntent{Identity: id, Entity: target, Track: 1}
	s.Next.PreviousTarget = &policy.TargetIntent{Identity: s.Observation.Identity, Entity: target, Track: 1}
	o.MonsterHealthDamage = damage
	if damage > 0 {
		o.Events = []DamageEvent{{Map: s.Observation.Identity.Map, Spawncount: s.Native.Spawncount, Frame: s.Native.EndFrame, Attacker: s.Native.Actor, Target: target, TargetClass: "monster_parasite", HealthBefore: 100, HealthAfter: 100 - damage, Take: damage}}
	}
	return c, s, o
}

func TestTargetCommitmentRewardProductiveHoldAndBudget(t *testing.T) {
	var q SequenceReward
	sum := 0.0
	for i := 0; i < 12; i++ {
		c, s, o := commitmentFixture(i, 7, 10)
		q.Config = c
		r := q.Evaluate(&s, o)
		if !r.Available {
			t.Fatal(r)
		}
		bonus := r.Components["target_commitment"]
		if i < 2 && bonus != 0 {
			t.Fatal("bonus before hold", r)
		}
		sum += bonus
	}
	if math.Abs(sum-.05) > 1e-12 {
		t.Fatal(sum)
	}
	// Switching does not replenish the life budget.
	for i := 12; i < 18; i++ {
		c, s, o := commitmentFixture(i, 8, 10)
		q.Config = c
		r := q.Evaluate(&s, o)
		if !r.Available || r.Components["target_commitment"] != 0 {
			t.Fatal(r)
		}
	}
}

func TestTargetCommitmentObservedSoldierAliases(t *testing.T) {
	for _, nativeClass := range []string{"monster_soldier", "monster_soldier_light", "monster_soldier_ss"} {
		t.Run(nativeClass, func(t *testing.T) {
			var q SequenceReward
			for i := 0; i < 3; i++ {
				c, s, o := commitmentFixture(i, 7, 10)
				q.Config = c
				for j := range s.Observation.Enemies {
					s.Observation.Enemies[j].Class = "monster_soldier"
				}
				o.Events[0].TargetClass = nativeClass
				o.Events[0].HealthBefore, o.Events[0].HealthAfter = 20, 10
				r := q.Evaluate(&s, o)
				if !r.Available || r.TargetCommitment.SelectedHealthDamage != 10 {
					t.Fatal("known observable alias dropped real damage", r)
				}
				if i == 2 && r.Components["target_commitment"] != .02 {
					t.Fatal("productive hold bonus changed", r)
				}
			}
		})
	}
	for _, nativeClass := range []string{"monster_gunner", "monster_soldier_light_unknown"} {
		c, s, o := commitmentFixture(0, 7, 10)
		for j := range s.Observation.Enemies {
			s.Observation.Enemies[j].Class = "monster_soldier"
		}
		o.Events[0].TargetClass = nativeClass
		q := SequenceReward{Config: c}
		if r := q.Evaluate(&s, o); r.Available || r.Reason != "commitment_target_class_mismatch" {
			t.Fatal("unrelated native class accepted", nativeClass, r)
		}
	}
}

func TestTargetCommitmentRewardNoIdleOrOtherTargetBonus(t *testing.T) {
	for _, mode := range []string{"idle", "other", "corpse", "gap", "track"} {
		t.Run(mode, func(t *testing.T) {
			var q SequenceReward
			for i := 0; i < 4; i++ {
				c, s, o := commitmentFixture(i, 7, 10)
				q.Config = c
				switch mode {
				case "idle":
					o.MonsterHealthDamage = 0
					o.Events = nil
				case "other":
					o.Events[0].Target = 8
				case "corpse":
					o.MonsterHealthDamage = 0
					o.Events[0].HealthBefore = -5
					o.Events[0].HealthAfter = -15
				case "gap":
					s.Execution.Matched = false
				case "track":
					s.Observation.PreviousTarget.Track = 99
				}
				r := q.Evaluate(&s, o)
				if mode != "gap" && !r.Available {
					t.Fatal(r)
				}
				if r.Components["target_commitment"] != 0 {
					t.Fatal(r)
				}
			}
		})
	}
}

func TestTargetCommitmentRewardVersionAndDamageEvidence(t *testing.T) {
	c, s, o := commitmentFixture(0, 7, 10)
	if c.Evaluate(&s, o).Reason != "sequence_context_required" {
		t.Fatal("stateless v12 accepted")
	}
	q := SequenceReward{Config: c}
	o.Events[0].Target = 0
	if r := q.Evaluate(&s, o); r.Available {
		t.Fatal("bad damage accepted", r)
	}
	c.Version = SelectedAimRewardVersion
	c.TargetChurn = 0
	if c.Validate() == nil {
		t.Fatal("bonus accepted in legacy reward")
	}
	c.Version = TargetCommitmentRewardVersion
	c.TargetChurn = -.02
	c.TargetCommitmentBudget = 0
	if c.Validate() == nil {
		t.Fatal("unbounded bonus accepted")
	}
}

func TestTargetCommitmentRewardNewThreatAndBudgetSurviveGap(t *testing.T) {
	var q SequenceReward
	for i := 0; i < 4; i++ {
		c, s, o := commitmentFixture(i, 7, 10)
		q.Config = c
		if i == 3 {
			e := s.Observation.Enemies[0]
			e.ID = 9
			s.Observation.Enemies = append(s.Observation.Enemies, e)
		}
		r := q.Evaluate(&s, o)
		if !r.Available {
			t.Fatal(r)
		}
		if i == 3 && r.Components["target_commitment"] != 0 {
			t.Fatal("new threat earned hold bonus", r)
		}
	}
	spent := q.commitment.spent
	c, s, o := commitmentFixture(4, 7, 10)
	q.Config = c
	s.Execution.Matched = false
	q.Evaluate(&s, o)
	if q.commitment.spent != spent {
		t.Fatal("gap replenished budget")
	}
}
