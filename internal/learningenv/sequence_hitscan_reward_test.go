package learningenv

import (
	"math"
	"reflect"
	"testing"
)

func TestSequenceHitscanV13ZeroControlRetainsV12Sequence(t *testing.T) {
	var legacy, control SequenceReward
	for i := 0; i < 12; i++ {
		c, s, o := commitmentFixture(i, 7, 10)
		legacy.Config = c
		before := legacy.Evaluate(&s, o)
		c.Version = SequenceHitscanRewardVersion
		control.Config = c
		o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1", Misses: []HitscanMiss{{Shot: 1, Outcome: "geometry", Window: *s.Native}}}
		after := control.Evaluate(&s, o)
		if !before.Available || !after.Available || *before.Score != *after.Score {
			t.Fatal("zero control changed objective", i, before, after)
		}
		delete(after.Components, "machinegun_miss")
		if !reflect.DeepEqual(before.Components, after.Components) || !reflect.DeepEqual(before.TargetCommitment, after.TargetCommitment) || !reflect.DeepEqual(before.TargetCycle, after.TargetCycle) || !reflect.DeepEqual(before.AimReference, after.AimReference) {
			t.Fatal("sequence semantics changed", i)
		}
	}
}

func TestSequenceHitscanV13ActualMissWithoutKnownTarget(t *testing.T) {
	for _, shots := range []int{0, 1, 2} {
		c, s, o := commitmentFixture(0, 7, 0)
		s.Observation.Enemies, s.Next.Enemies = nil, nil
		s.Action.TargetEntity, s.Action.TargetTrack = 0, 0
		s.AppliedAction.TargetEntity, s.AppliedAction.TargetTrack = 0, 0
		q := SequenceReward{Config: c}
		before := q.Evaluate(&s, o)
		c.Version, c.MachinegunMiss = SequenceHitscanRewardVersion, -.02
		q = SequenceReward{Config: c}
		o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1"}
		for n := 0; n < shots; n++ {
			o.HitscanMisses.Misses = append(o.HitscanMisses.Misses, HitscanMiss{Shot: uint32(n + 1), Outcome: "geometry", Window: *s.Native})
		}
		after := q.Evaluate(&s, o)
		if !before.Available || !after.Available || math.Abs(*after.Score-*before.Score-float64(shots)*c.MachinegunMiss) > 1e-12 || after.Components["off_target_attack"] != 0 {
			t.Fatal("wrong actual-shot delta", shots, before, after)
		}
	}
}

func TestSequenceHitscanV13RetainsTargetCycle(t *testing.T) {
	var legacy, control SequenceReward
	for i, target := range []int{7, 8, 7} {
		c, s, o := sequenceFixture(i, target)
		c.Version = TargetCommitmentRewardVersion
		legacy.Config = c
		before := legacy.Evaluate(&s, o)
		c.Version = SequenceHitscanRewardVersion
		control.Config = c
		o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1"}
		after := control.Evaluate(&s, o)
		if !before.Available || !after.Available || *before.Score != *after.Score || !reflect.DeepEqual(before.TargetCycle, after.TargetCycle) {
			t.Fatal("cycle semantics changed", before, after)
		}
		if i == 2 && after.Components["target_churn"] != -.02 {
			t.Fatal("target churn lost", after)
		}
	}
}

func TestSequenceHitscanV13EvidenceAndConfig(t *testing.T) {
	c, s, o := commitmentFixture(0, 7, 0)
	c.Version, c.MachinegunMiss = SequenceHitscanRewardVersion, -.02
	if c.HasProjectileMissCost() || !c.HasHitscanMissCost() || c.Evaluate(&s, o).Reason != "sequence_context_required" {
		t.Fatal("incorrect version capabilities")
	}
	for _, mode := range []string{"missing", "duplicate", "wrong_window", "recovery"} {
		t.Run(mode, func(t *testing.T) {
			_, step, outcome := commitmentFixture(0, 7, 0)
			miss := HitscanMiss{Shot: 1, Outcome: "geometry", Window: *step.Native}
			outcome.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1", Misses: []HitscanMiss{miss}}
			switch mode {
			case "missing":
				outcome.HitscanMisses = nil
			case "duplicate":
				outcome.HitscanMisses.Misses = append(outcome.HitscanMisses.Misses, miss)
			case "wrong_window":
				outcome.HitscanMisses.Misses[0].Window.Sequence++
			case "recovery":
				step.Execution.RecoveryCommands = 1
			}
			q := SequenceReward{Config: c}
			if q.Evaluate(&step, outcome).Available {
				t.Fatal("unproven miss accepted")
			}
		})
	}
	for _, value := range []float64{.01, -.1001} {
		bad := c
		bad.MachinegunMiss = value
		if bad.Validate() == nil {
			t.Fatal("unbounded coefficient accepted")
		}
	}
	bad := c
	bad.BlasterMiss = -.02
	if bad.Validate() == nil {
		t.Fatal("projectile cost leaked into v13")
	}
	bad = c
	bad.Version = TargetCommitmentRewardVersion
	if bad.Validate() == nil {
		t.Fatal("new coefficient leaked into legacy v12")
	}
}

func TestSequenceHitscanV13RetainsGoalAndDeath(t *testing.T) {
	for _, death := range []bool{false, true} {
		c, s, o := commitmentFixture(0, 7, 0)
		s.Terminal = true
		if death {
			s.Next.Health = 0
			o.Deaths = 1
		} else {
			s.Reason = "combat_goal_complete"
		}
		q := SequenceReward{Config: c}
		before := q.Evaluate(&s, o)
		c.Version, c.MachinegunMiss = SequenceHitscanRewardVersion, -.02
		o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1", Misses: []HitscanMiss{{Shot: 1, Outcome: "geometry", Window: *s.Native}}}
		q = SequenceReward{Config: c}
		after := q.Evaluate(&s, o)
		if !before.Available || !after.Available || math.Abs(*after.Score-*before.Score+.02) > 1e-12 || before.Components["death"] != after.Components["death"] {
			t.Fatal("terminal semantics changed", death, before, after)
		}
	}
}
