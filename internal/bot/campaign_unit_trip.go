package bot

import "q2coopbot/internal/quake"

// Stack order is verify, return, activate, outbound; the final entry owns the
// current goal. Campaign progress is suspended until the original door clears.
type CampaignUnitGoal struct {
	Kind          string            `json:"kind"`
	Map           string            `json:"map"`
	Path          []string          `json:"path,omitempty"`
	Leg           int               `json:"leg,omitempty"`
	Action        *quake.UnitAction `json:"action,omitempty"`
	contactFrames int
}

type CampaignUnitTrip struct {
	State            string              `json:"state"`
	OriginMap        string              `json:"origin_map"`
	DoorModel        int                 `json:"door_model"`
	Stack            []CampaignUnitGoal  `json:"stack"`
	Attempted        bool                `json:"action_attempted"`
	ElapsedFrames    int                 `json:"elapsed_frames"`
	Dependency       *CampaignDependency `json:"dependency"`
	EffectEvidence   string              `json:"effect_evidence,omitempty"`
	ProbeFrames      int                 `json:"probe_frames,omitempty"`
	probing          bool
	lastMap          string
	lastFrame        int
	verifyStarted    int
	verifyStartedSet bool
}

func (p *Planner) startCampaignUnitTrip(s quake.Snapshot) bool {
	d := p.campaignDependency
	if d == nil || len(p.CampaignRoute) == 0 {
		return false
	}
	var selected *quake.UnitAction
	for _, condition := range d.UnitConditions {
		for _, action := range condition.Activations {
			// Partial masks and shoot actions need their own execution plan.
			// Never silently treat a single partial setter as the full condition.
			if action.Action != "touch" || action.Map == s.Map || action.SetsFlags&condition.RequiredFlags != condition.RequiredFlags || !validUnitRoundTrip(s.Map, action) {
				continue
			}
			if selected == nil || len(action.TravelMaps)+len(action.ReturnMaps) < len(selected.TravelMaps)+len(selected.ReturnMaps) {
				a := action
				selected = &a
			}
		}
	}
	if selected == nil {
		return false
	}
	p.campaignUnitTrip = &CampaignUnitTrip{State: "travel_outbound", OriginMap: s.Map, DoorModel: d.DoorModel, Dependency: d, lastMap: s.Map, lastFrame: s.Frame,
		Stack: []CampaignUnitGoal{{Kind: "verify_effect", Map: s.Map}, {Kind: "travel_return", Map: s.Map, Path: append([]string(nil), selected.ReturnMaps...)}, {Kind: "activate", Map: selected.Map, Action: selected}, {Kind: "travel_outbound", Map: selected.Map, Path: append([]string(nil), selected.TravelMaps...)}}}
	p.button = nil
	p.routeKnown = false
	return true
}

func validUnitRoundTrip(origin string, a quake.UnitAction) bool {
	for _, path := range [][]string{a.TravelMaps, a.ReturnMaps} {
		if len(path) < 2 || len(path) > 16 {
			return false
		}
		for i, name := range path {
			if name == "" || i > 0 && name == path[i-1] {
				return false
			}
		}
	}
	return a.TravelMaps[0] == origin && a.TravelMaps[len(a.TravelMaps)-1] == a.Map && a.ReturnMaps[0] == a.Map && a.ReturnMaps[len(a.ReturnMaps)-1] == origin
}

// Returns a local action goal or the next map in a travel leg. Coordinates of
// another map are never returned. Contact records an attempt, not serverflags.
func (p *Planner) campaignUnitGoal(s quake.Snapshot, d *CampaignDecision) (quake.Vec3, bool, bool, string) {
	t := p.campaignUnitTrip
	d.UnitTrip = t
	d.RouteIndex, d.CompletedLevels = p.campaignRouteIndex, p.campaignRouteIndex
	advanced := s.Map != t.lastMap || s.Frame > t.lastFrame
	if advanced && t.probing {
		if s.Map == t.lastMap && s.Frame > t.lastFrame {
			t.ProbeFrames += s.Frame - t.lastFrame
		} else {
			t.ProbeFrames++
		}
	}
	if s.Map == t.lastMap && s.Frame > t.lastFrame {
		t.ElapsedFrames += s.Frame - t.lastFrame
	}
	t.lastMap, t.lastFrame = s.Map, s.Frame
	if t.ElapsedFrames > 1800 {
		t.State = "unit_goal_timeout"
	}
	if t.State == "unit_goal_timeout" || t.State == "unexpected_unit_map" || t.State == "effect_unconfirmed" {
		d.State = t.State
		return quake.Vec3{}, false, true, ""
	}
	for len(t.Stack) > 0 {
		goal := &t.Stack[len(t.Stack)-1]
		t.State, d.State = goal.Kind, goal.Kind
		switch goal.Kind {
		case "travel_outbound", "travel_return":
			if s.Map != goal.Path[goal.Leg] {
				if goal.Leg+1 >= len(goal.Path) || s.Map != goal.Path[goal.Leg+1] {
					t.State = "unexpected_unit_map"
					d.State = t.State
					return quake.Vec3{}, false, true, ""
				}
				goal.Leg++
				p.routeKnown = false
			}
			if goal.Leg+1 < len(goal.Path) {
				return quake.Vec3{}, false, false, goal.Path[goal.Leg+1]
			}
			t.Stack = t.Stack[:len(t.Stack)-1]
			continue
		case "activate":
			if s.Map != goal.Map {
				t.State = "unexpected_unit_map"
				d.State = t.State
				return quake.Vec3{}, false, true, ""
			}
			bounds, ok := p.World.Geometry.TouchBounds(goal.Action.Activation.Trigger)
			if !ok {
				d.State = "unit_activation_geometry_missing"
				return quake.Vec3{}, false, true, ""
			}
			exit := quake.MapExit{Min: bounds.Min, Max: bounds.Max, Center: quake.Vec3{(bounds.Min[0] + bounds.Max[0]) / 2, (bounds.Min[1] + bounds.Max[1]) / 2, (bounds.Min[2] + bounds.Max[2]) / 2}}
			at, ok := p.campaignExitContact(exit, s.Self)
			if !ok {
				d.State = "unit_activation_contact_unavailable"
				return quake.Vec3{}, false, true, ""
			}
			inside := true
			for axis := 0; axis < 3; axis++ {
				lo, hi := -16.0, 16.0
				if axis == 2 {
					lo, hi = -24, 32
				}
				inside = inside && s.Self[axis]+hi > bounds.Min[axis] && s.Self[axis]+lo < bounds.Max[axis]
			}
			if inside && s.Health > 0 {
				if advanced {
					goal.contactFrames++
				}
			} else {
				goal.contactFrames = 0
			}
			if goal.contactFrames >= 3 {
				t.Attempted = true
				t.Stack = t.Stack[:len(t.Stack)-1]
				p.routeKnown = false
				continue
			}
			return at, true, true, ""
		case "verify_effect":
			if s.Map != t.OriginMap {
				t.State = "unexpected_unit_map"
				d.State = t.State
				return quake.Vec3{}, false, true, ""
			}
			for _, mover := range s.Movers {
				if mover.Model == t.DoorModel && quake.Distance(mover.Origin, t.Dependency.initial) > 60 {
					t.EffectEvidence = "observed_mover_open"
					t.State = "effect_confirmed"
					t.Stack = nil
					p.campaignUnitTrip = nil
					p.campaignDependency = nil
					p.routeKnown = false
					return quake.Vec3{}, false, false, ""
				}
			}
			if p.campaignUnitProbePassed(s) {
				t.State = "effect_confirmed"
				t.EffectEvidence = "native_probe_passed"
				t.Stack = nil
				p.campaignUnitTrip = nil
				p.campaignDependency = nil
				p.routeKnown = false
				return quake.Vec3{}, false, false, ""
			}
			if t.verifyStarted == 0 && !t.verifyStartedSet {
				t.verifyStarted = s.Frame
				t.verifyStartedSet = true
			}
			if s.Frame-t.verifyStarted > 100 {
				t.State = "effect_unconfirmed"
				d.State = t.State
			}
			if t.ProbeFrames > 12 {
				t.State = "effect_unconfirmed"
				d.State = t.State
				return quake.Vec3{}, false, true, ""
			}
			if t.Dependency.ProbeFrom != t.Dependency.ProbeTo && quake.Horizontal(s.Self, t.Dependency.ProbeFrom) > 4 && !t.probing {
				return t.Dependency.ProbeFrom, true, true, ""
			}
			return quake.Vec3{}, false, true, ""
		}
	}
	return quake.Vec3{}, false, true, ""
}
