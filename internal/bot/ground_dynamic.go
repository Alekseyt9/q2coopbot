package bot

import "q2coopbot/internal/quake"

// nearbyGroundBrushModel identifies a reason to exclude static braking.
// Both spawn and observed bounds are considered; zero means none found.
func nearbyGroundBrushModel(s quake.Snapshot, g *quake.MapInfo) int {
	return nearbyGroundBrushModelExcept(s, g, 0)
}

func nearbyGroundBrushModelExcept(s quake.Snapshot, g *quake.MapInfo, exclude int) int {
	if g == nil {
		return 0
	}
	// Reject nearby brush models at both spawn and observed positions. Their
	// motion/rotation and support are outside the static friction model.
	for index, model := range g.Models {
		if index == 0 || index == exclude {
			continue
		}
		origins := []quake.Vec3{{}}
		for _, mover := range s.Movers {
			if mover.Model == index {
				origins = append(origins, mover.Origin)
			}
		}
		for _, origin := range origins {
			near := true
			for axis := range s.Self {
				if s.Self[axis] < model.Min[axis]+origin[axis]-32 || s.Self[axis] > model.Max[axis]+origin[axis]+32 {
					near = false
				}
			}
			if near {
				return index
			}
		}
	}
	return 0
}
