package quake

import "testing"

func TestLaserEntityAndPlayerHullHazard(t *testing.T) {
	entities := parseMapEntities(`{"classname" "target_laser" "origin" "-266 -400 -280" "angle" "180" "dmg" "100" "spawnflags" "3" "targetname" "t145"}`)
	if len(entities) != 1 || entities[0].Angles[1] != 180 || entities[0].Damage != 100 || entities[0].TargetName != "t145" {
		t.Fatalf("laser metadata missing: %+v", entities)
	}
	m := MapInfo{laserBeams: []laserBeam{{start: Vec3{-266, -400, -280}, end: Vec3{-840, -400, -280}, damage: 100}}}
	if !m.LaserMoveHazard(Vec3{-730, -455, -279.875}, Vec3{-730, -410, -279.875}) {
		t.Fatal("player hull crossed active beam without hazard")
	}
	if m.LaserMoveHazard(Vec3{-730, -455, -279.875}, Vec3{-730, -440, -279.875}) {
		t.Fatal("safe approach blocked too early")
	}
	if m.LaserMoveHazard(Vec3{-730, -455, -180}, Vec3{-730, -410, -180}) {
		t.Fatal("beam at another height blocked movement")
	}
}
