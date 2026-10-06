package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestRecoilAimPotentialUsesObservedPunchAndPreservesLegacy(t *testing.T) {
	c, s, _ := aimFixture()
	o := s.Observation
	direction, known, e := policy.ObservedAimDirection(o, o.Enemies[0])
	if e != nil || !known {
		t.Fatal(e)
	}
	desired := -math.Atan2(direction[2], math.Hypot(direction[0], direction[1])) * 180 / math.Pi
	o.ViewAngles[0] = int16(desired * 65536 / 360)
	legacy, e := aimPotential(o, c.AimPotential)
	if e != nil {
		t.Fatal(e)
	}
	kick := quake.Vec3{-13.5, 0, 0}
	o.KickAngles = &kick
	shifted, e := aimPotentialWithKick(o, c.AimPotential, true)
	if e != nil || shifted >= legacy-.001 {
		t.Fatalf("punch not reflected %v %v", shifted, e)
	}
	o.ViewAngles[0] = int16((desired + 13.5) * 65536 / 360)
	compensated, e := aimPotentialWithKick(o, c.AimPotential, true)
	if e != nil || math.Abs(compensated-legacy) > 1e-6 {
		t.Fatalf("compensation not rewarded %v %v", compensated, e)
	}
	unchanged, _ := aimPotential(s.Observation, c.AimPotential)
	withField := s.Observation
	withField.KickAngles = &kick
	again, _ := aimPotential(withField, c.AimPotential)
	if unchanged != again {
		t.Fatal("legacy reward changed")
	}
	o.KickAngles = nil
	if _, e = aimPotentialWithKick(o, c.AimPotential, true); e == nil {
		t.Fatal("missing punch accepted")
	}
}
