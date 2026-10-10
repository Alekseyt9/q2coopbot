package learningenv

import (
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func qualityFixture() (RewardConfig, Step, ServerOutcome) {
	c, s, o := rewardFixture()
	c.Version, c.MonsterKill, c.AimPotential, c.AimGamma = ActionQualityRewardVersion, 5, .5, .99
	c.SpacingPotential, c.ParasiteRange = 2, 288
	c.OffTargetAttack, c.TurnAway, c.StalledMovement = -.01, -.005, -.002
	clear, solid, track := true, uint16(8290), 1
	s.Observation.Weapon = "Machinegun"
	s.Observation.Enemies = []policy.Enemy{{ID: 7, Relative: quake.Vec3{200, 0, 0}, Distance: 200, ClearShot: &clear, Solid: &solid, Track: &track}}
	s.Next.Enemies = s.Observation.Enemies
	s.Action.TargetEntity, s.Action.TargetTrack = 7, 1
	s.Observation.PreviousTarget = &policy.TargetIntent{Entity: 7, Track: 1}
	return c, s, o
}

func TestActionQualityRewardBadAimAndTurn(t *testing.T) {
	c, s, o := qualityFixture()
	s.AppliedAction.Attack, s.AppliedAction.YawDelta = true, 90
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["off_target_attack"] != -.01 || r.Components["turn_away"] != -.005 {
		t.Fatal(r)
	}
	s.AppliedAction.YawDelta = 0
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["off_target_attack"] != 0 || r.Components["turn_away"] != 0 {
		t.Fatal("aligned shot penalized", r)
	}
}

func TestActionQualityRewardUnknownTargetAndLead(t *testing.T) {
	c, s, o := qualityFixture()
	s.AppliedAction.Attack, s.AppliedAction.YawDelta = true, 90
	s.Observation.Enemies[0].Solid = nil
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["off_target_attack"] != 0 {
		t.Fatal("unknown labelled miss", r)
	}
	c, s, o = qualityFixture()
	s.Observation.Weapon = "Blaster"
	v := quake.Vec3{0, 200, 0}
	s.Observation.Enemies[0].Velocity = &v
	s.AppliedAction.Attack, s.AppliedAction.YawDelta = true, 11
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["off_target_attack"] != 0 {
		t.Fatal("projectile lead penalized", r)
	}
}

func TestActionQualityRewardStallAndUsefulMovement(t *testing.T) {
	c, s, o := qualityFixture()
	s.Observation.OnGround, s.Next.OnGround = true, true
	s.Observation.PreviousCommand.Forward = 200
	s.AppliedAction.Forward, s.AppliedAction.Vertical = .5, "release"
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["stalled_movement"] != -.002 {
		t.Fatal(r)
	}
	s.Next.Position[0] = 12
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["stalled_movement"] != 0 {
		t.Fatal("real movement penalized", r)
	}
	s.Next.Position[0], s.AppliedAction.Vertical = 0, "jump"
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["stalled_movement"] != 0 {
		t.Fatal("jump penalized", r)
	}
}

func TestActionQualityRewardLegacyAndConfig(t *testing.T) {
	c, s, o := qualityFixture()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	c.OffTargetAttack = -.2
	if c.Validate() == nil {
		t.Fatal("unbounded cost accepted")
	}
	c, s, o = rewardFixture()
	r := c.Evaluate(&s, o)
	if !r.Available {
		t.Fatal(r)
	}
	if _, exists := r.Components["off_target_attack"]; exists {
		t.Fatal("legacy reward changed")
	}
}

func TestActionQualityRewardTargetChangesAndOtherWeapons(t *testing.T) {
	c, s, o := qualityFixture()
	s.AppliedAction.Attack, s.AppliedAction.YawDelta = true, 90
	s.Observation.PreviousTarget = nil
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["turn_away"] != 0 {
		t.Fatal("new target penalized", r)
	}
	s.Observation.Weapon = "Grenade Launcher"
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["off_target_attack"] != 0 {
		t.Fatal("unsupported trajectory penalized", r)
	}
	c, s, o = qualityFixture()
	s.AppliedAction.YawDelta = 90
	s.Next.Enemies = nil
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["turn_away"] != 0 {
		t.Fatal("lost target penalized", r)
	}
}
