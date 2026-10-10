package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func selectedAimFixture() (RewardConfig, Step, ServerOutcome) {
	c, s, o := qualityFixture()
	c.Version = SelectedAimRewardVersion
	s.Owner = "provider"
	s.Observation.Enemies[0].Relative, s.Observation.Enemies[0].Distance = quake.Vec3{100, 0, 0}, 100
	e := s.Observation.Enemies[0]
	track := 2
	e.ID, e.Track, e.Relative, e.Distance = 8, &track, quake.Vec3{0, 300, 0}, 300
	s.Observation.Enemies = append(s.Observation.Enemies, e)
	s.Next.Enemies = append([]policy.Enemy(nil), s.Observation.Enemies...)
	s.Action.TargetEntity, s.Action.TargetTrack = 8, 2
	previousID := s.Observation.Identity
	previousID.Frame--
	s.Observation.PreviousTarget = &policy.TargetIntent{Identity: previousID, Entity: 8, Track: 2}
	s.Next.PreviousTarget = &policy.TargetIntent{Identity: s.Observation.Identity, Entity: 8, Track: 2}
	s.Next.ViewAngles[1] = 16384
	s.AppliedAction.YawDelta = 90
	return c, s, o
}

func TestSelectedAimRewardFollowsChosenTarget(t *testing.T) {
	c, s, o := selectedAimFixture()
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["aim_potential"] <= .2 || r.AimReference == nil || !r.AimReference.Applied || r.AimReference.Entity != 8 {
		t.Fatal(r)
	}
	c.Version = ActionQualityRewardVersion
	legacy := c.Evaluate(&s, o)
	if !legacy.Available || legacy.Components["aim_potential"] >= -.2 || legacy.AimReference != nil {
		t.Fatal("nearest-target baseline changed", legacy)
	}
}

func TestSelectedAimRewardNoStationaryBonusAndWrongWayCosts(t *testing.T) {
	c, s, o := selectedAimFixture()
	s.Next.ViewAngles = s.Observation.ViewAngles
	s.AppliedAction.YawDelta = 0
	r := c.Evaluate(&s, o)
	if !r.Available || !r.AimReference.Applied || r.Components["aim_potential"] != 0 {
		t.Fatal("stationary aim received shaping", r)
	}
	s.Observation.ViewAngles[1], s.Next.ViewAngles[1] = 16384, 0
	s.AppliedAction.YawDelta = -90
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["aim_potential"] >= -.2 {
		t.Fatal("turn away from chosen target not charged", r)
	}
}

func TestSelectedAimRewardMasksSwitchVisibilityAndIdentity(t *testing.T) {
	for _, reason := range []string{"first_choice", "switch", "track", "stale_intent", "new_world", "occluded_current", "occluded_next", "missing_next", "next_intent", "no_target", "nonprovider"} {
		t.Run(reason, func(t *testing.T) {
			c, s, o := selectedAimFixture()
			switch reason {
			case "first_choice":
				s.Observation.PreviousTarget = nil
			case "switch":
				s.Observation.PreviousTarget.Entity = 7
			case "track":
				s.Observation.PreviousTarget.Track = 3
			case "stale_intent":
				s.Observation.PreviousTarget.Identity.Frame--
			case "new_world":
				s.Observation.PreviousTarget.Identity.Connection++
			case "occluded_current":
				f := false
				s.Observation.Enemies[1].ClearShot = &f
			case "occluded_next":
				f := false
				s.Next.Enemies[1].ClearShot = &f
			case "missing_next":
				s.Next.Enemies = s.Next.Enemies[:1]
			case "next_intent":
				s.Next.PreviousTarget.Entity = 7
			case "no_target":
				s.Action.TargetEntity = 0
			case "nonprovider":
				s.Owner = "rules"
			}
			r := c.Evaluate(&s, o)
			if !r.Available || r.Components["aim_potential"] != 0 || r.AimReference == nil || r.AimReference.Applied || r.Components["monster_damage"] != .1 {
				t.Fatal("masked aim lost native reward or gained switch bonus", r)
			}
		})
	}
}

func TestSelectedAimRewardTerminalAndHandoffKeepNativeCredit(t *testing.T) {
	for _, reason := range []string{"combat_goal_complete", "death", "control_handoff"} {
		t.Run(reason, func(t *testing.T) {
			c, s, o := selectedAimFixture()
			s.Next.Enemies = nil
			s.Reason = reason
			s.Terminal = reason != "control_handoff"
			s.Truncated = reason == "control_handoff"
			if reason == "death" {
				s.Next.Health = 0
				o.Deaths = 1
			}
			o.MonsterKills = 1
			o.Events = []DamageEvent{{Map: "base1", Spawncount: 42, Attacker: 1, Target: 8, TargetClass: "monster_soldier", HealthBefore: 10, HealthAfter: 0, Take: 10}}
			r := c.Evaluate(&s, o)
			if !r.Available || r.Components["monster_kill"] != 5 || r.Components["aim_potential"] != 0 || r.AimReference.Applied {
				t.Fatal(r)
			}
			if reason == "death" && r.Components["death"] != -5 {
				t.Fatal("death cost lost", r)
			}
		})
	}
}

func TestSelectedAimRewardRetainsKickAndConfigChecks(t *testing.T) {
	c, s, o := selectedAimFixture()
	c.AimKickAngles = true
	if c.Evaluate(&s, o).Available {
		t.Fatal("unknown kick accepted")
	}
	kick := quake.Vec3{0, 15, 0}
	s.Observation.KickAngles = &kick
	s.Next.KickAngles = &kick
	if !c.Evaluate(&s, o).Available {
		t.Fatal("known kick rejected")
	}
	c.TargetChurn = -.02
	if c.Validate() == nil {
		t.Fatal("v7 churn leaked into v8")
	}
	c.TargetChurn = 0
	c.AimPotential = math.Inf(1)
	if c.Validate() == nil {
		t.Fatal("nonfinite scale accepted")
	}
}
