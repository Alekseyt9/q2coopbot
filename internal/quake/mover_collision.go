package quake

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
	return c.boxClear(from, to, Vec3{-16, -16, -24}, Vec3{16, 16, 32})
}
func (m *MapInfo) moverCollision(live Mover) (CollisionMap, bool) {
	if m == nil || !m.HasCollision() || live.Model <= 0 || live.Model >= len(m.collision.modelBrushes) || len(m.collision.modelBrushes[live.Model]) == 0 {
		return CollisionMap{}, false
	}
	c := *m.collision
	c.worldBrushes = c.modelBrushes[live.Model]
	return c, true
}
