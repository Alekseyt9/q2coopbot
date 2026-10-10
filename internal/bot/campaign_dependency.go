package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

type CampaignDependency struct {
	Parent         *CampaignDependency   `json:"parent,omitempty"`
	State          string                `json:"state"`
	DoorModel      int                   `json:"door_model"`
	Activation     quake.Activation      `json:"activation"`
	Goal           quake.Vec3            `json:"goal"`
	UnitConditions []quake.UnitCondition `json:"unit_conditions,omitempty"`
	ProbeFrom      quake.Vec3            `json:"probe_from"`
	ProbeTo        quake.Vec3            `json:"probe_to"`
	initial        quake.Vec3
	started        int
}

// Discover goals only after an observed named door actually blocks movement.
func (p *Planner) discoverCampaignDependency(s quake.Snapshot, dx, dy float64) {
	if !p.campaignActive(s) || p.World.Goal != "reach_level_exit" || p.campaignUnitTrip != nil || s.Frame < p.campaignDependencyRetry {
		return
	}
	g := p.World.Geometry
	model, reason := g.DoorMoveBlock(s.Movers, s.Self, dx, dy)
	if reason != "dynamic_door_blocked" {
		return
	}
	depth := 0
	for d := p.campaignDependency; d != nil; d = d.Parent {
		depth++
		if d.DoorModel == model {
			return
		}
	}
	if depth >= 4 {
		return
	}
	parent := p.campaignDependency
	if parent != nil && parent.State != "approach_activation" {
		return
	}
	var live *quake.Mover
	for _, mover := range s.Movers {
		if mover.Model == model {
			value := mover
			live = &value
			break
		}
	}
	if live == nil {
		return
	}
	conditions := p.campaignUnitConditions(model)
	best := math.Inf(1)
	var selected *CampaignDependency
	for _, activation := range g.TouchActivationsForDoor(model) {
		bounds, _ := g.TouchBounds(activation.Trigger)
		exit := quake.MapExit{Min: bounds.Min, Max: bounds.Max, Center: quake.Vec3{(bounds.Min[0] + bounds.Max[0]) / 2, (bounds.Min[1] + bounds.Max[1]) / 2, (bounds.Min[2] + bounds.Max[2]) / 2}}
		at, ok := p.campaignExitContact(exit, s.Self)
		if !ok {
			continue
		}
		if cost := quake.Distance(s.Self, at); cost < best {
			best = cost
			selected = &CampaignDependency{Parent: parent, State: "approach_activation", DoorModel: model, Activation: activation, Goal: at, UnitConditions: conditions, initial: live.Origin, started: s.Frame}
		}
	}
	if selected == nil && len(conditions) > 0 && parent == nil {
		selected = &CampaignDependency{State: "unit_activation_required", DoorModel: model, UnitConditions: conditions, initial: live.Origin, started: s.Frame}
	}
	if selected != nil {
		p.campaignDependency = selected
		p.campaignDependency.ProbeFrom = s.Self
		end := s.Self
		if distance := math.Hypot(dx, dy); distance > 0 {
			end[0] += dx / distance * 40
			end[1] += dy / distance * 40
		}
		p.campaignDependency.ProbeTo = end
		p.routeKnown = false
	}
}

func (p *Planner) campaignDependencyGoal(s quake.Snapshot) (quake.Vec3, bool, bool) {
	d := p.campaignDependency
	if d == nil {
		return quake.Vec3{}, false, false
	}
	for _, mover := range s.Movers {
		if mover.Model == d.DoorModel && quake.Distance(mover.Origin, d.initial) > 1 && p.campaignDependencyPassable(s, d, mover) {
			p.rememberCampaignDoor(mover)
			p.campaignDependency = d.Parent
			if d.Parent != nil {
				d.Parent.started += s.Frame - d.started
				p.routeKnown = false
				return p.campaignDependencyGoal(s)
			}
			p.routeKnown = false
			return quake.Vec3{}, false, false
		}
	}
	p.World.Campaign.Dependency = d
	if d.State == "unit_activation_required" {
		// A remote map's coordinates are not a local navigation goal.
		// Unit travel/return needs an explicit itinerary and native proof.
		p.World.Campaign.State = "unit_activation_required"
		return quake.Vec3{}, false, true
	}
	if s.Frame-d.started > 300 {
		d.State = "activation_timeout"
		// Failed nested attempts remain explicit; don't loop between parents.
		if d.Parent != nil {
			return quake.Vec3{}, false, true
		}
		p.campaignDependency = nil
		p.campaignDependencyRetry = s.Frame + 100
		return quake.Vec3{}, false, true
	}
	p.World.Campaign.State = "unlock_exit_route"
	return d.Goal, true, true
}

// Native doors can retain a small lip after opening. Use the same bounded
// step clearance as movement, rather than waiting for a level hull sweep.
func (p *Planner) campaignDependencyPassable(s quake.Snapshot, d *CampaignDependency, mover quake.Mover) bool {
	if d.ProbeFrom == d.ProbeTo {
		return true
	}
	g := p.World.Geometry
	if g == nil {
		return false
	}
	if g.MoverHullClear(mover, d.ProbeFrom, d.ProbeTo) {
		return true
	}
	// A horizontal stair probe cannot validate a descending hatch corridor.
	if math.Abs(d.ProbeTo[2]-d.ProbeFrom[2]) > 2 {
		return false
	}
	dx, dy := d.ProbeTo[0]-d.ProbeFrom[0], d.ProbeTo[1]-d.ProbeFrom[1]
	distance := math.Hypot(dx, dy)
	if distance == 0 {
		return false
	}
	_, reason := g.DoorMoveBlockStep(s.Movers, d.ProbeFrom, dx, dy, distance)
	return reason == ""
}

// Remove edges intersecting the observed blocking door. Ordinary guards still
// validate all actual movement; this never makes a closed door passable.
func (p *Planner) campaignDependencyNavigator(s quake.Snapshot) *quake.Navigator {
	d := p.campaignDependency
	if p.campaignUnitTrip != nil {
		if s.Map == p.campaignUnitTrip.OriginMap {
			d = p.campaignUnitTrip.Dependency
		} else {
			return p.Nav
		}
	}
	if d == nil {
		d = p.campaignExitBlock
	}
	if d == nil || p.Nav == nil {
		return p.Nav
	}
	blockers := []quake.Mover{}
	for node := d; node != nil; node = node.Parent {
		mover := quake.Mover{Model: node.DoorModel, Origin: node.initial}
		for _, m := range s.Movers {
			if m.Model == node.DoorModel {
				mover = m
				break
			}
		}
		blockers = append(blockers, mover)
	}
	n := *p.Nav
	n.Edges = make([][]quake.Edge, len(p.Nav.Edges))
	for area, list := range p.Nav.Edges {
		for _, edge := range list {
			clear := true
			for _, mover := range blockers {
				if !p.World.Geometry.MoverHullClear(mover, edge.Start, edge.End) {
					clear = false
					break
				}
			}
			if clear {
				n.Edges[area] = append(n.Edges[area], edge)
			}
		}
	}
	return &n
}

func (p *Planner) campaignDependencyRoute(s quake.Snapshot, goal quake.Vec3) ([]quake.Waypoint, bool) {
	n := p.campaignDependencyNavigator(s)
	if n == nil {
		return nil, false
	}
	if route, ok := n.Route(s.Self, goal); ok {
		p.campaignDependency.State = "approach_activation"
		return route, true
	}
	if route, ok := p.checkedCampaignGroundRoute(s, goal); ok {
		p.campaignDependency.State = "approach_activation"
		return route, true
	}
	p.campaignDependency.State = "activation_route_unavailable"
	if len(p.campaignDependency.UnitConditions) > 0 {
		p.campaignDependency.State = "unit_activation_required"
	}
	return nil, false
}

func (p *Planner) campaignUnitConditions(model int) []quake.UnitCondition {
	conditions := p.World.Geometry.UnitDoorConditions(model)
	if len(conditions) == 0 {
		return nil
	}
	// Only explicitly configured campaign maps are inspected. Names and
	// target chains remain scoped to each map; flags bridge the maps.
	maps := []*quake.MapInfo{p.World.Geometry}
	seen := map[string]bool{p.World.Map: true}
	if p.campaignAssetRoot != "" {
		for _, name := range append(append([]string(nil), p.CampaignRoute...), p.CampaignUnitMaps...) {
			if seen[name] {
				continue
			}
			seen[name] = true
			if info, err := quake.LoadMap(p.campaignAssetRoot, name); err == nil {
				maps = append(maps, &info)
			}
		}
	}
	for i := range conditions {
		paths := quake.UnitMapPaths(p.World.Map, maps)
		for _, info := range maps {
			path, sameUnit := paths[info.Name]
			if !sameUnit {
				continue
			}
			for _, action := range info.UnitActions(conditions[i].RequiredFlags) {
				action.TravelMaps = path
				action.ReturnMaps = quake.UnitMapPaths(info.Name, maps)[p.World.Map]
				conditions[i].Activations = append(conditions[i].Activations, action)
			}
		}
	}
	return conditions
}
