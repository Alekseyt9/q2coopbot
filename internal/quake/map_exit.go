package quake

// MapExit is a BSP touch volume, not an AAS waypoint.
type MapExit struct {
	Destination string `json:"destination"`
	Model       int    `json:"model"`
	Min         Vec3   `json:"min"`
	Max         Vec3   `json:"max"`
	Center      Vec3   `json:"center"`
}

func (m *MapInfo) Exits() []MapExit {
	if m == nil {
		return nil
	}
	var out []MapExit
	for _, e := range m.Entities {
		if e.Class != "trigger_changelevel" && e.Class != "trigger_multiple" && e.Class != "trigger_once" {
			continue
		}
		destination := e.Map
		if e.Class != "trigger_changelevel" {
			destination = ""
			for _, target := range m.Entities {
				if e.Target != "" && target.TargetName == e.Target && target.Class == "target_changelevel" {
					destination = target.Map
					break
				}
			}
		}
		if destination == "" {
			continue
		}
		bounds, ok := m.TouchBounds(e)
		if !ok {
			continue
		}
		x := MapExit{Destination: destination, Model: e.Model, Min: bounds.Min, Max: bounds.Max}
		valid := true
		for i := 0; i < 3; i++ {
			if x.Max[i] <= x.Min[i] {
				valid = false
			}
			x.Center[i] = (x.Min[i] + x.Max[i]) / 2
		}
		if valid {
			out = append(out, x)
		}
	}
	return out
}
