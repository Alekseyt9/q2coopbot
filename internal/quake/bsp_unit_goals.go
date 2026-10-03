package quake

// UnitCondition describes a native cross-level flag requirement. All bits in
// RequiredFlags must be set; UDP does not expose the actual game.serverflags.
type UnitCondition struct {
	Map           string        `json:"map"`
	RequiredFlags int           `json:"required_flags"`
	FlagState     string        `json:"flag_state"`
	Target        MapEntity     `json:"target"`
	Chain         []MapEntity   `json:"chain"`
	Activations   []UnitAction  `json:"activations,omitempty"`
}

type UnitAction struct {
	Map        string     `json:"map"`
	SetsFlags  int        `json:"sets_flags"`
	Action     string     `json:"action"`
	Activation Activation `json:"activation"`
}

// UnitDoorConditions does not interpret a same-named target on another map as
// a link. Only native cross-level bits can connect two maps within a unit.
func (m *MapInfo) UnitDoorConditions(model int) []UnitCondition {
	if m == nil {
		return nil
	}
	var name string
	for _, e := range m.Entities {
		if e.Model == model && (e.Class == "func_door" || e.Class == "func_door_rotating") {
			name = e.TargetName
		}
	}
	if name == "" {
		return nil
	}
	var conditions []UnitCondition
	for _, e := range m.Entities {
		if e.Class != "target_crosslevel_target" || e.SpawnFlags <= 0 || e.SpawnFlags&^255 != 0 {
			continue
		}
		if chain, ok := m.targetPath(e.Target, name, []MapEntity{e}, map[string]bool{}); ok {
			conditions = append(conditions, UnitCondition{Map: m.Name, RequiredFlags: e.SpawnFlags, FlagState: "unknown", Target: e, Chain: chain})
		}
	}
	return conditions
}

// UnitActions lists physical actions that set at least one required bit.
// Several actions can be needed; their presence never proves flags are set.
func (m *MapInfo) UnitActions(required int) []UnitAction {
	if m == nil || required <= 0 || required&^255 != 0 {
		return nil
	}
	var actions []UnitAction
	for _, setter := range m.Entities {
		if setter.Class != "target_crosslevel_trigger" || setter.TargetName == "" || setter.SpawnFlags&required == 0 {
			continue
		}
		for _, trigger := range m.Entities {
			if trigger.TargetName != "" || trigger.Target == "" {
				continue
			}
			action := "touch"
			switch trigger.Class {
			case "trigger_once", "trigger_multiple":
				if trigger.SpawnFlags&6 != 0 {
					continue
				}
			case "func_button":
				if trigger.Health > 0 {
					action = "shoot"
				}
			default:
				continue
			}
			if _, ok := m.Model(trigger.Model); !ok {
				continue
			}
			chain, ok := m.targetPath(trigger.Target, setter.TargetName, []MapEntity{trigger}, map[string]bool{})
			if !ok {
				continue
			}
			chain = append(chain, setter)
			actions = append(actions, UnitAction{Map: m.Name, SetsFlags: setter.SpawnFlags & 255, Action: action, Activation: Activation{Trigger: trigger, Chain: chain}})
		}
	}
	return actions
}
