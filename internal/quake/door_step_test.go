package quake

import "testing"

func TestDoorStepRequiresObservedLowLipAndClearSupportedHull(t *testing.T) {
	for _, tc := range []struct {
		name                                  string
		z                                     float64
		visible, floor, ceiling, overheadDoor bool
		want                                  string
	}{
		{"lowered", -122, true, true, false, false, ""},
		{"closed", 0, true, true, false, false, "dynamic_door_blocked"},
		{"too_high", -104, true, true, false, false, "dynamic_door_blocked"},
		{"unknown", -122, false, true, false, false, "dynamic_door_unobserved"},
		{"no_floor", -122, true, false, false, false, "dynamic_door_blocked"},
		{"ceiling", -122, true, true, true, false, "dynamic_door_blocked"},
		{"overhead_door", -122, true, true, false, true, "dynamic_door_blocked"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &CollisionMap{}
			add := func(lo, hi Vec3) {
				planes, brush := testBoxBrush(lo, hi, len(c.sides))
				for range planes {
					c.sides = append(c.sides, uint16(len(c.sides)))
				}
				c.planes = append(c.planes, planes...)
				c.worldBrushes = append(c.worldBrushes, len(c.brushes))
				c.brushes = append(c.brushes, brush)
			}
			if tc.floor {
				add(Vec3{-100, -100, -20}, Vec3{100, 100, 0})
			} else {
				add(Vec3{-100, -100, -200}, Vec3{100, 100, -180})
			}
			if tc.ceiling {
				add(Vec3{-100, -100, 60}, Vec3{100, 100, 80})
			}
			m := &MapInfo{collision: c, Entities: []MapEntity{{Class: "func_door", Model: 1}}, Models: []BSPModel{{}, {Min: Vec3{20, -50, 0}, Max: Vec3{30, 50, 128}}}}
			var movers []Mover
			if tc.visible {
				movers = append(movers, Mover{Model: 1, Origin: Vec3{0, 0, tc.z}})
			}
			if tc.overheadDoor {
				m.Entities = append(m.Entities, MapEntity{Class: "func_door", Model: 2})
				m.Models = append(m.Models, BSPModel{Min: Vec3{20, -50, 60}, Max: Vec3{30, 50, 80}})
				movers = append(movers, Mover{Model: 2})
			}
			if got := m.DoorMoveHazard(movers, Vec3{0, 0, 24.125}, 1, 0); got != tc.want {
				t.Fatalf("got %s want %s", got, tc.want)
			}
		})
	}
}
