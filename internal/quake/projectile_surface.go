package quake

import "math"

// ProjectileSurfaceContact bounds the first static contact of EVERY segment
// connecting the two endpoint boxes. It deliberately accepts only a single
// axis-aligned brush face, with all other faces strictly enclosing the sweep.
// Bodies, movers and subsequent reflections are outside this certificate.
type ProjectileSurfaceContact struct {
	Normal   Vec3    `json:"normal"`
	Distance float64 `json:"distance"`
	Min      Vec3    `json:"min"`
	Max      Vec3    `json:"max"`
}

func (m *MapInfo) ProjectileCommonSurface(fromMin, fromMax, toMin, toMax Vec3) *ProjectileSurfaceContact {
	if !m.HasCollision() {
		return nil
	}
	return m.collision.projectileCommonSurface(fromMin, fromMax, toMin, toMax)
}

func planeBoxDistances(p bspPlane, min, max Vec3) (float64, float64) {
	lo, hi := -p.dist, -p.dist
	for i, n := range p.normal {
		if n >= 0 {
			lo += n * min[i]
			hi += n * max[i]
		} else {
			lo += n * max[i]
			hi += n * min[i]
		}
	}
	return lo, hi
}

func (c *CollisionMap) projectileCommonSurface(a, b, d, e Vec3) *ProjectileSurfaceContact {
	min, max := Vec3{}, Vec3{}
	for i := range a {
		for _, v := range []float64{a[i], b[i], d[i], e[i]} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil
			}
		}
		if a[i] > b[i] || d[i] > e[i] {
			return nil
		}
		min[i], max[i] = math.Min(a[i], d[i]), math.Max(b[i], e[i])
	}
	var result *ProjectileSurfaceContact
	for _, index := range c.worldBrushes {
		brush := c.brushes[index]
		if brush.contents&(1|2|0x2000000|0x4000000) == 0 {
			continue
		}
		separated := false
		for side := brush.first; side < brush.first+brush.count; side++ {
			lo, _ := planeBoxDistances(c.planes[c.sides[side]], min, max)
			if lo > .03125 {
				separated = true
				break
			}
		}
		if separated {
			continue
		}
		// More than one possible brush includes corners and thin obstacles.
		if result != nil {
			return nil
		}
		candidate := -1
		for side := brush.first; side < brush.first+brush.count; side++ {
			p := c.planes[c.sides[side]]
			lo, _ := planeBoxDistances(p, a, b)
			_, hi := planeBoxDistances(p, d, e)
			if lo > .03125 && hi < -.03125 {
				if candidate != -1 {
					return nil
				}
				candidate = side
			}
		}
		if candidate < 0 {
			return nil
		}
		for side := brush.first; side < brush.first+brush.count; side++ {
			if side == candidate {
				continue
			}
			_, hi := planeBoxDistances(c.planes[c.sides[side]], min, max)
			if hi >= -.03125 {
				return nil
			}
		}
		if candidate >= len(c.sideSurfaces) || !c.sideSurfaces[candidate].known || c.sideSurfaces[candidate].flags&4 != 0 {
			return nil
		}
		p := c.planes[c.sides[candidate]]
		axis := -1
		for i, n := range p.normal {
			if n == 0 {
				continue
			}
			if axis != -1 || math.Abs(n) != 1 {
				return nil
			}
			axis = i
		}
		if axis < 0 {
			return nil
		}
		result = &ProjectileSurfaceContact{Normal: p.normal, Distance: p.dist, Min: min, Max: max}
		// Trace epsilon stops outside the face. Pad floating point rounding.
		position := p.normal[axis] * (p.dist + .03125)
		result.Min[axis], result.Max[axis] = position-1e-6, position+1e-6
	}
	return result
}
