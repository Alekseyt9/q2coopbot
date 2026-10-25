package learningenv

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestNativeCampaignObjectivePreservesUnfinishedHandoff(t *testing.T) {
	b, err := os.ReadFile("../../scripts/scenarios/combat-reward-native-goal-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var c RewardConfig
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	_, s, _ := aimFixture()
	s.Truncated, s.Reason = true, "control_handoff"
	if c.HandoffBootstrap != "nonterminal_v1" || c.ZeroBootstrap(&s) {
		t.Fatal("Unfinished native campaign fight must retain continuation value")
	}
	if err := MarkGoalHandoff(&s, s.Next.Identity.Frame-1, s.Next.Identity.Frame); err != nil {
		t.Fatal(err)
	}
	if !c.ZeroBootstrap(&s) {
		t.Fatal("Verified goal must close continuation")
	}
}

// Scalar reward/boundary arithmetic only: no neural network is instantiated.
func TestNonterminalHandoffBootstrap(t *testing.T) {
	c, s, outcome := aimFixture()
	s.Observation.ViewAngles[1] = 16384
	s.Next.ViewAngles = s.Observation.ViewAngles
	s.Truncated, s.Reason = true, "control_handoff"
	legacy := c.Evaluate(&s, outcome)
	if !legacy.Available || !c.ZeroBootstrap(&s) || math.Abs(legacy.Components["aim_potential"]-.25) > 1e-9 {
		t.Fatal("legacy behavior changed", legacy)
	}
	c.HandoffBootstrap = "nonterminal_v1"
	r := c.Evaluate(&s, outcome)
	if !r.Available || c.ZeroBootstrap(&s) || math.Abs(r.Components["aim_potential"]-.0025) > 1e-9 {
		t.Fatal("visibility loss must retain next-state potential and value", r)
	}
	if err := MarkGoalHandoff(&s, s.Next.Identity.Frame-1, s.Next.Identity.Frame); err != nil {
		t.Fatal(err)
	}
	r = c.Evaluate(&s, outcome)
	if !r.Available || !s.GoalHandoff || !c.ZeroBootstrap(&s) || math.Abs(r.Components["aim_potential"]-.25) > 1e-9 {
		t.Fatal("verified goal must close bootstrap", r)
	}
}

func TestNonterminalHandoffConfigAndGoalEvidence(t *testing.T) {
	c, s, _ := aimFixture()
	c.HandoffBootstrap = "unknown"
	if c.Validate() == nil {
		t.Fatal("unknown objective accepted")
	}
	c.HandoffBootstrap = "nonterminal_v1"
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	s.Truncated, s.Reason = true, "control_handoff"
	if err := MarkGoalHandoff(&s, s.Next.Identity.Frame+1, s.Next.Identity.Frame+2); err == nil || s.GoalHandoff {
		t.Fatal("future kill cannot close this handoff")
	}
	if err := MarkGoalHandoff(&s, 0, s.Next.Identity.Frame); err == nil {
		t.Fatal("unverified kill frame accepted")
	}
	s.Terminal, s.Truncated, s.Reason = true, false, "observed_death"
	if !c.ZeroBootstrap(&s) {
		t.Fatal("death must have zero bootstrap")
	}
}
