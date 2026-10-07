package aimquery

import (
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func observation() policy.Observation {
	clear := true
	solid := uint16(2 | (3 << 5) | (8 << 10))
	return policy.Observation{Version: policy.ObservationVersion, Identity: policy.Identity{Frame: 101, Life: 1, Map: "base1", Actor: 1, Connection: 1, Spawncount: 1}, Health: 100, Weapon: "Blaster", Enemies: []policy.Enemy{{ID: 2, Class: "monster_parasite", Relative: quake.Vec3{100, 0, 0}, ClearShot: &clear, Solid: &solid}}}
}
func TestQueryRecoversLargeYawWithoutInventingAppliedEffects(t *testing.T) {
	o := observation()
	o.ViewAngles[1] = 16384
	q, e := Query(o)
	if e != nil || q == nil || math.Abs(q.Action.YawDelta+90) > .01 {
		t.Fatal(q, e)
	}
	if q.Action.Attack || q.Action.Forward != 0 || q.Action.Side != 0 || q.Action.Vertical != "release" || q.Action.Weapon != "" {
		t.Fatal("query invented another head", q)
	}
}
func TestQueryWrapAndCrouchedEye(t *testing.T) {
	o := observation()
	o.ViewAngles[1] = int16(math.Round(179 * 65536 / 360))
	yaw := -179 * math.Pi / 180
	o.Enemies[0].Relative = quake.Vec3{100 * math.Cos(yaw), 100 * math.Sin(yaw), 0}
	q, e := Query(o)
	if e != nil || q == nil || math.Abs(q.Action.YawDelta-2) > .02 {
		t.Fatal(q, e)
	}
	o = observation()
	o.Ducked = true
	q, e = Query(o)
	if e != nil || q == nil || q.Action.PitchDelta >= -10 {
		t.Fatal("ignored current eye/bbox", q, e)
	}
}
func TestQueryRejectsUnknownOrOccludedTargets(t *testing.T) {
	o := observation()
	o.Enemies[0].Solid = nil
	q, e := Query(o)
	if e != nil || q != nil {
		t.Fatal(q, e)
	}
	o = observation()
	clear := false
	o.Enemies[0].ClearShot = &clear
	other := observation().Enemies[0]
	other.ID = 3
	other.Relative[0] = 150
	o.Enemies = append(o.Enemies, other)
	q, e = Query(o)
	if e != nil || q == nil || q.Target != 3 {
		t.Fatal("occluded nearer target blocked visible target", q, e)
	}
	o.AgeMS = 301
	if _, e = Query(o); e == nil {
		t.Fatal("stale observation accepted")
	}
}
