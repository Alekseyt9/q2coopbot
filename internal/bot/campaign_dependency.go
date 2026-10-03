package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

type CampaignDependency struct {
	State          string                `json:"state"`
	DoorModel      int                   `json:"door_model"`
	Activation     quake.Activation      `json:"activation"`
	Goal           quake.Vec3            `json:"goal"`
	UnitConditions []quake.UnitCondition `json:"unit_conditions,omitempty"`
	initial        quake.Vec3
	started        int
}

// Discover goals only after an observed named door actually blocks movement.
func (p *Planner) discoverCampaignDependency(s quake.Snapshot, dx, dy float64) {
	if !p.Campaign || s.Teammate != nil || p.World.Goal != "reach_level_exit" || p.campaignDependency != nil || p.campaignUnitTrip != nil || s.Frame < p.campaignDependencyRetry {
		return
	}
	g := p.World.Geometry
	model, reason := g.DoorMoveBlock(s.Movers, s.Self, dx, dy)
	if reason != "dynamic_door_blocked" {
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
	for _, activation := range g.TouchActivationsForDoor(model) {
		bounds, _ := g.TouchBounds(activation.Trigger)
		exit := quake.MapExit{Min: bounds.Min, Max: bounds.Max, Center: quake.Vec3{(bounds.Min[0] + bounds.Max[0]) / 2, (bounds.Min[1] + bounds.Max[1]) / 2, (bounds.Min[2] + bounds.Max[2]) / 2}}
		at, ok := p.campaignExitContact(exit, s.Self)
		if !ok {
			continue
		}
		if cost := quake.Distance(s.Self, at); cost < best {
			best = cost
			p.campaignDependency = &CampaignDependency{State: "approach_activation", DoorModel: model, Activation: activation, Goal: at, UnitConditions: conditions, initial: live.Origin, started: s.Frame}
		}
	}
	if p.campaignDependency == nil && len(conditions) > 0 {
		p.campaignDependency = &CampaignDependency{State: "unit_activation_required", DoorModel: model, UnitConditions: conditions, initial: live.Origin, started: s.Frame}
	}
	if p.campaignDependency != nil {
		p.routeKnown = false
	}
}

func (p *Planner) campaignDependencyGoal(s quake.Snapshot) (quake.Vec3, bool, bool) {
	d := p.campaignDependency
	if d == nil {
		return quake.Vec3{}, false, false
	}
	for _, mover := range s.Movers {
		if mover.Model == d.DoorModel && quake.Distance(mover.Origin, d.initial) > 60 {
			p.campaignDependency = nil
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
		p.campaignDependency = nil
		p.campaignDependencyRetry = s.Frame + 100
		return quake.Vec3{}, false, true
	}
	p.World.Campaign.State = "unlock_exit_route"
	return d.Goal, true, true
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
	mover := quake.Mover{Model: d.DoorModel, Origin: d.initial}
	for _, m := range s.Movers {
		if m.Model == d.DoorModel {
			mover = m
			break
		}
	}
	n := *p.Nav
	n.Edges = make([][]quake.Edge, len(p.Nav.Edges))
	for area, list := range p.Nav.Edges {
		for _, edge := range list {
			if p.World.Geometry.MoverHullClear(mover, edge.Start, edge.End) {
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
