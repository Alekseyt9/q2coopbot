package quake

import "math"

// MoverFooting checks an observed brush model rather than its bounding box.
// The caller must establish that the mover remains stationary during motion.
func (m *MapInfo) MoverFooting(live Mover, point Vec3, maxDrop float64) (float64, bool) {
	c, ok := m.moverCollision(live)
	if !ok {
		return 0, false
	}
	for axis := range point {
		point[axis] -= live.Origin[axis]
	}
	return c.groundDrop(point, maxDrop)
}
func (m *MapInfo) MoverHullClear(live Mover, from, to Vec3) bool {
	c, ok := m.moverCollision(live)
	if !ok {
		return false
	}
	for axis := range from {
		from[axis] -= live.Origin[axis]
		to[axis] -= live.Origin[axis]
	}
	if !c.boxClear(from, to, Vec3{-16, -16, -24}, Vec3{16, 16, 32}) {
		return false
	}
	if live.Angles != (Vec3{}) {
		// Yamagi CM_TransformedBoxTrace rotates the center segment into
		// model space but keeps mins/maxs unchanged. Require its sweep too:
		// a geometrically clear hull must not command a native collision.
		local := *m.collision
		local.worldBrushes = local.modelBrushes[live.Model]
		from = inverseBrushVector(from, live.Angles)
		to = inverseBrushVector(to, live.Angles)
		return local.boxClear(from, to, Vec3{-16, -16, -24}, Vec3{16, 16, 32})
	}
	return true
}
func (m *MapInfo) moverCollision(live Mover) (CollisionMap, bool) {
	if m == nil || !m.HasCollision() || live.Model <= 0 || live.Model >= len(m.collision.modelBrushes) || len(m.collision.modelBrushes[live.Model]) == 0 {
		return CollisionMap{}, false
	}
	c := *m.collision
	c.worldBrushes = c.modelBrushes[live.Model]
	for _, angle := range live.Angles {
		if math.IsNaN(angle) || math.IsInf(angle, 0) {
			return CollisionMap{}, false
		}
	}
	if live.Angles != (Vec3{}) {
		// Rotate brush planes, leaving the player's hull axis aligned in
		// world space. Rotating only the segment would rotate the hull too.
		// Coordinates remain relative to live.Origin, as in the fast path.
		c.planes, c.sides, c.brushes, c.worldBrushes = nil, nil, nil, nil
		c.sideSurfaces = nil
		for _, index := range m.collision.modelBrushes[live.Model] {
			brush := m.collision.brushes[index]
			copyBrush := brush
			copyBrush.first = len(c.sides)
			for side := brush.first; side < brush.first+brush.count; side++ {
				plane := m.collision.planes[m.collision.sides[side]]
				plane.normal = rotateBrushVector(plane.normal, live.Angles)
				// CollisionMap's plane references are uint16.
				if len(c.planes) >= 1<<16 {
					return CollisionMap{}, false
				}
				c.sides = append(c.sides, uint16(len(c.planes)))
				c.planes = append(c.planes, plane)
				if side < len(m.collision.sideSurfaces) {
					c.sideSurfaces = append(c.sideSurfaces, m.collision.sideSurfaces[side])
				} else {
					c.sideSurfaces = append(c.sideSurfaces, groundSurface{})
				}
			}
			c.worldBrushes = append(c.worldBrushes, len(c.brushes))
			c.brushes = append(c.brushes, copyBrush)
		}
	}
	return c, true
}

// Quake II AngleVectors: local axes are forward, -right, up.
func rotateBrushVector(v, angles Vec3) Vec3 {
	sp, cp := math.Sincos(angles[0] * math.Pi / 180)
	sy, cy := math.Sincos(angles[1] * math.Pi / 180)
	sr, cr := math.Sincos(angles[2] * math.Pi / 180)
	return Vec3{
		v[0]*cp*cy + v[1]*(sr*sp*cy-cr*sy) + v[2]*(cr*sp*cy+sr*sy),
		v[0]*cp*sy + v[1]*(sr*sp*sy+cr*cy) + v[2]*(cr*sp*sy-sr*cy),
		-v[0]*sp + v[1]*sr*cp + v[2]*cr*cp,
	}
}

func inverseBrushVector(v, angles Vec3) Vec3 {
	var out Vec3
	for axis := range out {
		basis := Vec3{}
		basis[axis] = 1
		rotated := rotateBrushVector(basis, angles)
		for i := range v {
			out[axis] += v[i] * rotated[i]
		}
	}
	return out
}
