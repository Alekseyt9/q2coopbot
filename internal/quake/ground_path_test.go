package quake

import (
	"math"
	"testing"
)

func TestGroundPathStatus(t *testing.T) {
	for _, tc := range []struct{ name, want string }{
		{"clear", "static_sampled_clear"}, {"wall", "static_hull_blocked"},
		{"edge", "uneven_or_missing_support"}, {"gap", "uneven_or_missing_support"},
		{"slick_middle", "slick"}, {"seam", "static_sampled_clear"}, {"low_ceiling", "static_hull_blocked"}, {"duck", "static_sampled_clear"},
		{"long", "unsupported_path"}, {"vertical", "unsupported_path"}, {"nan", "invalid_path"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &CollisionMap{}
			add := func(lo, hi Vec3, flags uint32) {
				planes, b := testBoxBrush(lo, hi, len(c.sides))
				for range planes {
					c.sides = append(c.sides, uint16(len(c.sides)))
					c.sideSurfaces = append(c.sideSurfaces, groundSurface{known: true, flags: flags})
				}
				c.planes = append(c.planes, planes...)
				c.worldBrushes = append(c.worldBrushes, len(c.brushes))
				c.brushes = append(c.brushes, b)
			}
			from, to := Vec3{0, 0, 24.125}, Vec3{64, 0, 24.125}
			switch tc.name {
			case "gap":
				add(Vec3{-100, -100, -20}, Vec3{24, 100, 0}, 0)
				add(Vec3{40, -100, -20}, Vec3{150, 100, 0}, 0)
			case "slick_middle":
				add(Vec3{-100, -100, -20}, Vec3{24.5, 100, 0}, 0)
				add(Vec3{24.5, -100, -20}, Vec3{40.5, 100, 0}, 2)
				add(Vec3{40.5, -100, -20}, Vec3{150, 100, 0}, 0)
			case "seam":
				add(Vec3{-100, -100, -20}, Vec3{24, 100, 0}, 0)
				add(Vec3{24, -100, -20}, Vec3{150, 100, 0}, 0)
			case "edge":
				add(Vec3{-100, -100, -20}, Vec3{70, 100, 0}, 0)
			default:
				add(Vec3{-100, -100, -20}, Vec3{150, 100, 0}, 0)
			}
			switch tc.name {
			case "wall":
				add(Vec3{32, -100, 0}, Vec3{34, 100, 100}, 0)
			case "duck", "low_ceiling":
				add(Vec3{32, -100, 36}, Vec3{34, 100, 100}, 0)
			case "long":
				to[0] = 65
			case "vertical":
				to[2]++
			case "nan":
				to[0] = math.NaN()
			}
			m := &MapInfo{collision: c, Models: []BSPModel{{}}}
			if got := m.GroundPathStatus(from, to, tc.name == "duck"); got != tc.want {
				t.Fatalf("got %s want %s", got, tc.want)
			}
		})
	}
	if got := (*MapInfo)(nil).GroundPathStatus(Vec3{}, Vec3{}, false); got != "unknown_geometry" {
		t.Fatal(got)
	}
}
