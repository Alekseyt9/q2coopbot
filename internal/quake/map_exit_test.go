package quake

import "testing"

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
