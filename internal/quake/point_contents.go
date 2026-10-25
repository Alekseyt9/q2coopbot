package quake

import "math"

// StaticPointContents reads original world brushes, excluding dynamic entities.
func (m *MapInfo) StaticPointContents(point Vec3) (int, bool) {
	if m == nil || !m.HasCollision() {
		return 0, false
	}
	for _, v := range point {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, false
		}
	}
	contents := 0
	c := m.collision
	for _, index := range c.worldBrushes {
		b := c.brushes[index]
		inside := true
		for side := b.first; side < b.first+b.count; side++ {
			p := c.planes[c.sides[side]]
			d := -p.dist
			for axis, n := range p.normal {
				d += point[axis] * n
			}
			if d > 0 {
				inside = false
				break
			}
		}
		if inside && b.count > 0 {
			contents |= b.contents
		}
	}
	return contents, true
}
