package quake

import (
	"math"
	"testing"
)

func TestGroundFrictionStatus(t *testing.T) {
	for _, tc := range []struct{ name, want string }{
		{"flat", "dry_flat"}, {"slick", "slick"}, {"unknown", "unknown_surface"},
		{"edge", "uneven_or_missing_support"}, {"water", "liquid_or_current"},
		{"current", "liquid_or_current"}, {"remote_water", "dry_flat"},
		{"slope", "slope"}, {"invalid", "invalid_origin"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &CollisionMap{}
			add := func(lo, hi Vec3, contents int) {
				planes, b := testBoxBrush(lo, hi, len(c.sides))
				b.contents = contents
				for range planes {
					c.sides = append(c.sides, uint16(len(c.sides)))
					c.sideSurfaces = append(c.sideSurfaces, groundSurface{known: true})
				}
				c.planes = append(c.planes, planes...)
				c.worldBrushes = append(c.worldBrushes, len(c.brushes))
				c.brushes = append(c.brushes, b)
			}
			add(Vec3{-100, -100, -20}, Vec3{100, 100, 0}, 1)
			origin := Vec3{0, 0, 24}
			switch tc.name {
			case "slick":
				c.sideSurfaces[4].flags = 2
			case "unknown":
				c.sideSurfaces = nil
			case "edge":
				origin[0] = 90
			case "water":
				add(Vec3{-30, -30, 0}, Vec3{30, 30, 8}, 32)
			case "current":
				add(Vec3{-30, -30, 0}, Vec3{30, 30, 8}, 0x40000)
			case "remote_water":
				add(Vec3{200, 200, 0}, Vec3{230, 230, 8}, 32)
			case "slope":
				c.planes[4].normal = Vec3{0.1, 0, math.Sqrt(0.99)}
			case "invalid":
				origin[0] = math.NaN()
			}
			m := &MapInfo{collision: c}
			if got := m.GroundFrictionStatus(origin); got != tc.want {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
	if got := (*MapInfo)(nil).GroundFrictionStatus(Vec3{}); got != "unknown_geometry" {
		t.Fatal(got)
	}
}
