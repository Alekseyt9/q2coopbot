package quake

// A lowered door may leave a small lip that PM_StepSlideMove can climb.
// Require observed bounds, static support and clearance for the whole hull;
// this does not authorize touching a closed or unobserved full-height door.
func (m *MapInfo) doorStepClear(movers []Mover, from, to Vec3, top float64) bool {
	rise := top - (from[2] - 24)
	if !m.HasCollision() || rise < 0 || rise > 18 {
		return false
	}
	raised, end := from, to
	raised[2], end[2] = top+24+0.125, top+24+0.125
	if !m.PlayerMoveClear(from, raised) || !m.PlayerMoveClear(raised, end) {
		return false
	}
	if _, ok := m.GroundDrop(end, 18.125); !ok {
		return false
	}
	for _, entity := range m.Entities {
		if entity.Class != "func_door" {
			continue
		}
		model, ok := m.Model(entity.Model)
		if !ok {
			continue
		}
		lo, hi := model.Min, model.Max
		for _, live := range movers {
			if live.Model == entity.Model {
				for axis := range lo {
					lo[axis] += live.Origin[axis]
					hi[axis] += live.Origin[axis]
				}
				break
			}
		}
		mins, maxs := Vec3{-16, -16, -24}, Vec3{16, 16, 32}
		if sweptBoxAABB(from, raised, lo, hi, mins, maxs) || sweptBoxAABB(raised, end, lo, hi, mins, maxs) {
			return false
		}
	}
	return true
}
