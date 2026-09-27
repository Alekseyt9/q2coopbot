package quake

import "math"

type groundSurface struct {
	known bool
	flags uint32
}

// GroundFrictionStatus checks static support under the player footprint.
// "dry_flat" describes only the surface, not path clearance, dynamic support,
// server physics settings or applicability of a complete movement prediction.
func (m *MapInfo) GroundFrictionStatus(origin Vec3) string {
	if m == nil || !m.HasCollision() {
		return "unknown_geometry"
	}
	for _, v := range origin {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return "invalid_origin"
		}
	}
	c := m.collision
	// Conservative brush/standing-hull overlap. Extra rejection is preferable
	// to assuming ordinary friction in a liquid or conveyor current volume.
	for _, index := range c.worldBrushes {
		b := c.brushes[index]
		if b.contents&(8|16|32|0xFC0000) == 0 {
			continue
		}
		outside := false
		for side := b.first; side < b.first+b.count; side++ {
			p := c.planes[c.sides[side]]
			d := -p.dist
			for axis, n := range p.normal {
				lo, hi := -16.0, 16.0
				if axis == 2 {
					lo, hi = -24, 32
				}
				offset := lo
				if n < 0 {
					offset = hi
				}
				d += (origin[axis] + offset) * n
			}
			if d > 0 {
				outside = true
				break
			}
		}
		if !outside {
			return "liquid_or_current"
		}
	}
	for _, offset := range []Vec3{{}, {-16, -16, 0}, {-16, 16, 0}, {16, -16, 0}, {16, 16, 0}} {
		p := Vec3{origin[0] + offset[0], origin[1] + offset[1], origin[2]}
		// Surface classification uses the actual brush boundary. Shrinking all
		// faces by the movement epsilon creates artificial gaps at floor seams.
		drop, normal, surface, ok := c.groundContactInset(p, 1, 0)
		if !ok || drop > 0.25 {
			return "uneven_or_missing_support"
		}
		if normal[2] < 0.9999 {
			return "slope"
		}
		if !surface.known {
			return "unknown_surface"
		}
		if surface.flags&2 != 0 {
			return "slick"
		}
	}
	return "dry_flat"
}
