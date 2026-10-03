package quake

import "testing"

func TestTranslatedTouchExitBounds(t *testing.T) {
	m := &MapInfo{Models: []BSPModel{{}, {Min: Vec3{0, 0, 0}, Max: Vec3{8, 32, 64}}}, Entities: []MapEntity{{Class: "trigger_changelevel", Model: 1, Origin: Vec3{100, -200, 16}, Map: "next"}}}
	x := m.Exits()
	if len(x) != 1 || x[0].Min != (Vec3{100, -200, 16}) || x[0].Center != (Vec3{104, -184, 48}) {
		t.Fatal("entity origin ignored", x)
	}
}

func TestUnobservedDoorUsesEntityTranslation(t *testing.T) {
	m := &MapInfo{Models: []BSPModel{{}, {Min: Vec3{0, 0, 0}, Max: Vec3{8, 32, 64}}}, Entities: []MapEntity{{Class: "func_door", Model: 1, Origin: Vec3{100, -200, 0}}}}
	if id, reason := m.DoorMoveBlockStep(nil, Vec3{70, -184, 24}, 1, 0, 40); id != 1 || reason != "dynamic_door_unobserved" {
		t.Fatal("translated missing door ignored", id, reason)
	}
	if id, _ := m.DoorMoveBlockStep(nil, Vec3{-30, 16, 24}, 1, 0, 40); id != 0 {
		t.Fatal("untranslated ghost door", id)
	}
}

func TestMapExitsResolveTouchTriggerInsteadOfTargetOrigin(t *testing.T) {
	m := &MapInfo{Models: []BSPModel{{}, {Min: Vec3{-1840, 1448, -24}, Max: Vec3{-1712, 1640, 32}}}}
	m.Entities = parseMapEntities(`{"classname" "trigger_multiple" "model" "*1" "target" "exit"} {"classname" "target_changelevel" "targetname" "exit" "map" "base2$base1" "origin" "999 999 999"} {"classname" "func_door" "model" "*1" "target" "exit"} {"classname" "trigger_multiple" "model" "*9" "target" "exit"}`)
	exits := m.Exits()
	if len(exits) != 1 || exits[0].Center != (Vec3{-1776, 1544, 4}) || exits[0].Destination != "base2$base1" {
		t.Fatal(exits)
	}
	if (*MapInfo)(nil).Exits() != nil {
		t.Fatal("nil geometry")
	}
}
