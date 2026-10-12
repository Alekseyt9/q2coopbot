package quake

import "math"

// MoverNearMotion checks the entire stock translation travel, not just the
// current door pose. Unknown mover behavior remains unsupported.
func (m *MapInfo) MoverNearMotion(live Mover, from, to Vec3) bool {
	if live.SolidKnown && live.Solid == 0 {
		return false
	}
	model, ok := m.Model(live.Model)
	if !ok {
		return true
	}
	var entity *MapEntity
	for i := range m.Entities {
		if m.Entities[i].Model == live.Model {
			entity = &m.Entities[i]
			break
		}
	}
	if entity == nil {
		return true
	}
	start, end := entity.Origin, entity.Origin
	size := Vec3{}
	for axis := range size {
		size[axis] = model.Max[axis] - model.Min[axis]
	}
	switch entity.Class {
	case "func_wall", "func_explosive":
	case "func_door", "func_button":
		lip := entity.Lip
		if lip == 0 {
			lip = 8
			if entity.Class == "func_button" {
				lip = 4
			}
		}
		direction := Vec3{}
		if entity.Angles == (Vec3{0, -1, 0}) {
			direction[2] = 1
		} else if entity.Angles == (Vec3{0, -2, 0}) {
			direction[2] = -1
		} else {
			pitch, yaw := entity.Angles[0]*math.Pi/180, entity.Angles[1]*math.Pi/180
			direction = Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
		}
		distance := -lip
		for axis := range size {
			distance += math.Abs(direction[axis]) * size[axis]
		}
		for axis := range end {
			end[axis] += direction[axis] * distance
		}
	case "func_plat":
		height := entity.Height
		if height == 0 {
			lip := entity.Lip
			if lip == 0 {
				lip = 8
			}
			height = size[2] - lip
		}
		end[2] -= height
	case "func_rotating", "func_door_rotating", "turret_base", "turret_breach":
		radius := 0.0
		for bits := 0; bits < 8; bits++ {
			corner := model.Min
			for axis := range corner {
				if bits&(1<<axis) != 0 {
					corner[axis] = model.Max[axis]
				}
			}
			radius = math.Max(radius, Distance(Vec3{}, corner))
		}
		model.Min = Vec3{-radius, -radius, -radius}
		model.Max = Vec3{radius, radius, radius}
	case "func_train":
		lo, hi := model.Min, model.Max
		for axis := range lo {
			lo[axis] += live.Origin[axis]
			hi[axis] += live.Origin[axis]
		}
		queue := []string{entity.Target}
		seen := map[string]bool{}
		found := false
		for len(queue) > 0 {
			name := queue[0]
			queue = queue[1:]
			if name == "" || seen[name] {
				continue
			}
			seen[name] = true
			matched := false
			for _, corner := range m.Entities {
				if corner.Class != "path_corner" || corner.TargetName != name {
					continue
				}
				matched = true
				found = true
				for axis := range lo {
					lo[axis] = math.Min(lo[axis], corner.Origin[axis])
					hi[axis] = math.Max(hi[axis], corner.Origin[axis]+size[axis])
				}
				queue = append(queue, corner.Target)
			}
			if !matched {
				return true
			}
		}
		if !found {
			return true
		}
		for axis := range lo {
			lo[axis] -= .125
			hi[axis] += .125
		}
		return sweptBoxAABB(from, to, lo, hi, Vec3{-16, -16, -24}, Vec3{16, 16, 32})
	default:
		return true
	}
	lo, hi := model.Min, model.Max
	for axis := range lo {
		if math.IsNaN(start[axis]) || math.IsInf(start[axis], 0) || math.IsNaN(end[axis]) || math.IsInf(end[axis], 0) || math.IsNaN(live.Origin[axis]) || math.IsInf(live.Origin[axis], 0) {
			return true
		}
		lo[axis] += math.Min(start[axis], math.Min(end[axis], live.Origin[axis])) - .125
		hi[axis] += math.Max(start[axis], math.Max(end[axis], live.Origin[axis])) + .125
	}
	return sweptBoxAABB(from, to, lo, hi, Vec3{-16, -16, -24}, Vec3{16, 16, 32})
}
