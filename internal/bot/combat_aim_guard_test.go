package bot

import (
	"math"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func TestLearnedCombatAimGuardLimitsAndWorldMovement(t *testing.T) {
	p := &Planner{}
	s := quake.Snapshot{}
	o := policy.Observation{}
	cmd := quake.UserCmd{Yaw: 16384, Pitch: -16000, Forward: 100}
	got, changes := p.guardLearnedCombatAim(o, s, policy.Action{}, cmd)
	if math.Abs(float64(got.Yaw)*360/65536-36) > .01 || math.Abs(float64(got.Pitch)*360/65536+18) > .01 || len(changes) < 3 {
		t.Fatal(got, changes)
	}
	yaw := float64(got.Yaw) * 2 * math.Pi / 65536
	x := float64(got.Forward)*math.Cos(yaw) + float64(got.Side)*math.Sin(yaw)
	y := float64(got.Forward)*math.Sin(yaw) - float64(got.Side)*math.Cos(yaw)
	if math.Abs(x) > 1 || math.Abs(y-100) > 1 {
		t.Fatal("requested world movement changed", x, y)
	}
}

func TestLearnedCombatAimGuardRejectsTurnAwayWithoutAutoAim(t *testing.T) {
	p := &Planner{}
	track := 1
	s := quake.Snapshot{Enemies: []quake.Object{{ID: 7, Origin: quake.Vec3{200, 0, 0}}}}
	o := policy.Observation{Enemies: []policy.Enemy{{ID: 7, Track: &track}}}
	got, changes := p.guardLearnedCombatAim(o, s, policy.Action{TargetEntity: 7, TargetTrack: 1}, quake.UserCmd{Yaw: 6000})
	if got.Yaw != 0 || len(changes) == 0 {
		t.Fatal("turn away retained", got, changes)
	}
	// A deliberately off-axis view is held, never replaced by perfect aiming.
	o.ViewAngles[1] = 2000
	got, _ = p.guardLearnedCombatAim(o, s, policy.Action{TargetEntity: 7, TargetTrack: 1}, quake.UserCmd{Yaw: 8000})
	if got.Yaw != 2000 {
		t.Fatal("guard invented aim", got)
	}
}

func TestLearnedCombatAimGuardRayBounds(t *testing.T) {
	min, max := quake.Vec3{100, -16, -24}, quake.Vec3{132, 16, 32}
	if f, ok := learnedRayBox(quake.Vec3{}, quake.Vec3{1000, 0, 0}, min, max); !ok || math.Abs(f-.1) > 1e-8 {
		t.Fatal(f, ok)
	}
	if _, ok := learnedRayBox(quake.Vec3{}, quake.Vec3{0, 1000, 0}, min, max); ok {
		t.Fatal("wall-directed ray accepted")
	}
}
