package learningenv

import (
	"math"
	"reflect"
	"testing"

	"q2coopbot/internal/policy"
)

func TestBlockedGroundRequestV14ZeroControlAndScope(t *testing.T) {
	for _, reason := range []string{"static_hull_blocked", "static_hull_component_clipped", "no_ground_support", "unsupported_motion_guard", "dynamic_door_blocked", ""} {
		for _, ground := range []bool{true, false} {
			for _, vertical := range []string{"release", "jump", "crouch"} {
				c, s, o := commitmentFixture(0, 7, 0)
				c.Version = SequenceHitscanRewardVersion
				o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1"}
				s.Observation.OnGround = ground
				s.Action.Forward, s.Action.Vertical = .8, vertical
				s.AppliedAction.Forward, s.AppliedAction.Side = 0, 0
				s.Interventions = []policy.Intervention{{Component: "movement", Reason: reason}}
				legacy := (&SequenceReward{Config: c}).Evaluate(&s, o)
				c.Version = BlockedGroundRequestRewardVersion
				zero := (&SequenceReward{Config: c}).Evaluate(&s, o)
				if !legacy.Available || !zero.Available || *legacy.Score != *zero.Score {
					t.Fatal("zero control changed score", reason, ground, vertical, legacy, zero)
				}
				delete(zero.Components, "blocked_ground_request")
				if !reflect.DeepEqual(legacy.Components, zero.Components) || !reflect.DeepEqual(legacy.TargetCommitment, zero.TargetCommitment) {
					t.Fatal("legacy components changed")
				}
				c.BlockedGroundRequest = -.01
				candidate := (&SequenceReward{Config: c}).Evaluate(&s, o)
				want := 0.0
				if ground && vertical == "release" && (reason == "static_hull_blocked" || reason == "static_hull_component_clipped") {
					want = -.01
				}
				if !candidate.Available || candidate.Components["blocked_ground_request"] != want || math.Abs(*candidate.Score-*legacy.Score-want) > 1e-12 {
					t.Fatal("wrong proposal cost", reason, ground, vertical, candidate)
				}
			}
		}
	}
}

func TestBlockedGroundRequestV14PreservesUsefulMovement(t *testing.T) {
	c := RewardConfig{BlockedGroundRequest: -.01}
	s := Step{Observation: policy.Observation{OnGround: true}, Action: policy.Action{Forward: .8, Vertical: "release"},
		Interventions: []policy.Intervention{{Component: "movement", Reason: "static_hull_component_clipped"}}}
	for _, applied := range []float64{.1, .3, .8} {
		s.AppliedAction.Side = applied
		if c.blockedGroundRequestCost(&s) != 0 {
			t.Fatal("penalized remaining movement", applied)
		}
	}
	s.Action.Forward, s.AppliedAction.Side = .2, 0
	if c.blockedGroundRequestCost(&s) != 0 {
		t.Fatal("penalized small request")
	}
}

func TestBlockedGroundRequestV14Validation(t *testing.T) {
	c, _, _ := commitmentFixture(0, 7, 0)
	c.Version = BlockedGroundRequestRewardVersion
	for _, cost := range []float64{0, -.01, -.05} {
		c.BlockedGroundRequest = cost
		if err := c.Validate(); err != nil {
			t.Fatal(cost, err)
		}
	}
	for _, cost := range []float64{.01, -.051, math.NaN(), math.Inf(1)} {
		c.BlockedGroundRequest = cost
		if c.Validate() == nil {
			t.Fatal("accepted invalid cost", cost)
		}
	}
	c.Version, c.BlockedGroundRequest = SequenceHitscanRewardVersion, -.01
	if c.Validate() == nil {
		t.Fatal("legacy objective accepted new cost")
	}
}

func TestBlockedGroundRequestV14ZeroControlRetainsSequenceHistory(t *testing.T) {
	var legacy, control SequenceReward
	for i := 0; i < 12; i++ {
		c, s, o := commitmentFixture(i, 7, 10)
		c.Version = SequenceHitscanRewardVersion
		legacy.Config = c
		o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1"}
		before := legacy.Evaluate(&s, o)
		c.Version = BlockedGroundRequestRewardVersion
		control.Config = c
		after := control.Evaluate(&s, o)
		if !before.Available || !after.Available || *before.Score != *after.Score {
			t.Fatal("sequence zero control changed score", i)
		}
		after.Version = before.Version
		delete(after.Components, "blocked_ground_request")
		if !reflect.DeepEqual(before, after) {
			t.Fatal("sequence zero control changed evidence", i)
		}
	}
}
