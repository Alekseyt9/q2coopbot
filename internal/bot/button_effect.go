package bot

import (
	"q2coopbot/internal/quake"
	"sort"
)

type buttonEffectKey struct{ button, door int }

// Permanent buttons are consumed by an observed movement, not by attempted
// contact or a sent shot. Retain this independently of the short interaction.
type ButtonEffect struct {
	ButtonModel   int    `json:"button_model"`
	DoorModel     int    `json:"door_model"`
	Action        string `json:"action"`
	State         string `json:"state"`
	PressedFrame  int    `json:"pressed_frame"`
	CheckedFrame  int    `json:"checked_frame"`
	buttonInitial quake.Vec3
	doorInitial   quake.Vec3
}

func (p *Planner) rememberButtonEffect(task *buttonTask, frame int) {
	if p.World.Geometry == nil {
		return
	}
	permanent := false
	for _, e := range p.World.Geometry.Entities {
		if e.Class == "func_button" && e.Model == task.buttonModel && e.Wait < 0 {
			permanent = true
			break
		}
	}
	if !permanent {
		return
	}
	key := buttonEffectKey{task.buttonModel, task.doorModel}
	if p.buttonEffects[key] != nil {
		return
	}
	if p.buttonEffects == nil {
		p.buttonEffects = map[buttonEffectKey]*ButtonEffect{}
	}
	p.buttonEffects[key] = &ButtonEffect{ButtonModel: task.buttonModel, DoorModel: task.doorModel, Action: task.action, State: "awaiting_door_effect", PressedFrame: frame, CheckedFrame: frame, buttonInitial: task.buttonInitial, doorInitial: task.initial}
}

func (p *Planner) updateButtonEffects(s quake.Snapshot) {
	for key, effect := range p.buttonEffects {
		for _, mover := range s.Movers {
			if mover.Model == effect.ButtonModel && quake.Distance(mover.Origin, effect.buttonInitial) <= 1 {
				// A real reset contradicts the remembered consumption. PVS loss
				// alone cannot make a permanent button available again.
				delete(p.buttonEffects, key)
				break
			}
			if mover.Model == effect.DoorModel {
				effect.CheckedFrame = s.Frame
				if quake.Distance(mover.Origin, effect.doorInitial) > 60 {
					if effect.State != "door_effect_observed" {
						p.routeKnown = false
					}
					effect.State = "door_effect_observed"
				} else if effect.State != "door_effect_observed" && s.Frame-effect.PressedFrame >= 80 {
					effect.State = "door_effect_not_observed"
				}
			}
		}
	}
}

func (p *Planner) buttonEffectDecisions() []ButtonEffect {
	var results []ButtonEffect
	for _, effect := range p.buttonEffects {
		results = append(results, *effect)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].ButtonModel == results[j].ButtonModel {
			return results[i].DoorModel < results[j].DoorModel
		}
		return results[i].ButtonModel < results[j].ButtonModel
	})
	return results
}
