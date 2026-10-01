package quake

import "math"

// ProjectileBoxClear certifies separation from static MASK_SHOT brushes.
// Failure to find a separating brush plane means possible collision, not an
// exact hit/normal. Checking only corners would miss an obstacle inside a box.
func (m *MapInfo) ProjectileBoxClear(min, max Vec3) (clear, valid bool) {
	if !m.HasCollision() {
		return false, false
	}
	return m.collision.projectileBoxClear(min, max)
}

func (c *CollisionMap) projectileBoxClear(min, max Vec3) (clear, valid bool) {
	for i := range min {
		if math.IsNaN(min[i]) || math.IsNaN(max[i]) || math.IsInf(min[i], 0) || math.IsInf(max[i], 0) || min[i] > max[i] {
			return false, false
		}
	}
	for _, index := range c.worldBrushes {
		b := c.brushes[index]
		if b.contents&(1|2|0x2000000|0x4000000) == 0 {
			continue
		}
		separated := false
		for side := b.first; side < b.first+b.count; side++ {
			p := c.planes[c.sides[side]]
			distance := -p.dist
			for i := range min {
				if p.normal[i] >= 0 {
					distance += min[i] * p.normal[i]
				} else {
					distance += max[i] * p.normal[i]
				}
			}
			if distance > .03125 {
				separated = true
				break
			}
		}
		if !separated {
			return false, true
		}
	}
	return true, true
}

// PointTrace describes a zero-sized projectile against static MASK_SHOT brushes.
// Dynamic entities must be handled separately by the caller.
type PointTrace struct {
	Fraction               float64
	End, Normal            Vec3
	StartSolid, Sky, Valid bool
}

func (m *MapInfo) TraceProjectile(from, to Vec3) PointTrace {
	if !m.HasCollision() {
		return PointTrace{}
	}
	return m.collision.traceProjectile(from, to)
}

func (c *CollisionMap) traceProjectile(from, to Vec3) PointTrace {
	tr := PointTrace{Fraction: 1, End: to, Valid: true}
	for _, p := range []Vec3{from, to} {
		for _, v := range p {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return PointTrace{}
			}
		}
	}
	for _, index := range c.worldBrushes {
		b := c.brushes[index]
		if b.contents&(1|2|0x2000000|0x4000000) == 0 {
			continue
		}
		enter, leave, outside, rejected := -1.0, 1.0, false, false
		normal, sky := Vec3{}, false
		for side := b.first; side < b.first+b.count; side++ {
			p := c.planes[c.sides[side]]
			d1, d2 := -p.dist, -p.dist
			for i := range from {
				d1 += from[i] * p.normal[i]
				d2 += to[i] * p.normal[i]
			}
			if d1 > 0 {
				outside = true
			}
			if d1 > 0 && d2 >= d1 {
				rejected = true
				break
			}
			if d1 <= 0 && d2 <= 0 {
				continue
			}
			if d1 > d2 {
				f := (d1 - 0.03125) / (d1 - d2)
				if f > enter {
					enter = f
					normal = p.normal
					sky = side < len(c.sideSurfaces) && c.sideSurfaces[side].known && c.sideSurfaces[side].flags&4 != 0
				}
			} else {
				leave = math.Min(leave, (d1+0.03125)/(d1-d2))
			}
		}
		if rejected {
			continue
		}
		if !outside {
			tr.StartSolid = true
			tr.Fraction = 0
			tr.End = from
			return tr
		}
		if enter < leave && enter > -1 && enter < tr.Fraction {
			tr.Fraction = math.Max(0, enter)
			tr.Normal = normal
			tr.Sky = sky
		}
	}
	for i := range from {
		tr.End[i] = from[i] + (to[i]-from[i])*tr.Fraction
	}
	return tr
}
