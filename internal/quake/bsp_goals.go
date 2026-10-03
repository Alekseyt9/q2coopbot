package quake

// Activation is a physical touch volume followed by a bounded BSP target chain.
// It describes a possible activation, not its live availability or reachability.
type Activation struct {
	Trigger MapEntity   `json:"trigger"`
	Chain   []MapEntity `json:"chain"`
}

// targetPath follows only BSP classes whose ordinary use forwards Target.
// Entity names are map-local; they must never be matched across different maps.
func (m *MapInfo) targetPath(name, target string, chain []MapEntity, seen map[string]bool) ([]MapEntity, bool) {
	if name == target && target != "" {
		return chain, true
	}
	if name == "" || seen[name] || len(chain) >= 8 {
		return nil, false
	}
	seen[name] = true
	defer delete(seen, name)
	for _, e := range m.Entities {
		if e.TargetName != name || (e.Class != "trigger_relay" && e.Class != "func_door" && e.Class != "func_door_rotating" && e.Class != "func_button") {
			continue
		}
		if path, ok := m.targetPath(e.Target, target, append(append([]MapEntity(nil), chain...), e), seen); ok {
			return path, true
		}
	}
	return nil, false
}

func (m *MapInfo) TouchActivationsForDoor(model int) []Activation {
	if m == nil {
		return nil
	}
	var target string
	for _, e := range m.Entities {
		if e.Model == model && (e.Class == "func_door" || e.Class == "func_door_rotating") {
			target = e.TargetName
			break
		}
	}
	if target == "" {
		return nil
	}
	var result []Activation
	for _, trigger := range m.Entities {
		if (trigger.Class != "trigger_once" && trigger.Class != "trigger_multiple") || trigger.Target == "" || trigger.TargetName != "" || trigger.SpawnFlags&6 != 0 {
			continue
		}
		if _, ok := m.Model(trigger.Model); !ok {
			continue
		}
		if chain, ok := m.targetPath(trigger.Target, target, []MapEntity{trigger}, map[string]bool{}); ok {
			result = append(result, Activation{Trigger: trigger, Chain: chain})
		}
	}
	return result
}
