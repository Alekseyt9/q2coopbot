package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Only permanently open, non-toggle sliding doors can outlive PVS visibility.
// New contradictory observations revoke memory; map changes discard it.
func (p *Planner) rememberCampaignDoor(m quake.Mover) {
	if p.World.Geometry == nil {
		return
	}
	for _, e := range p.World.Geometry.Entities {
		if e.Model == m.Model && e.Class == "func_door" && e.Wait < 0 && e.SpawnFlags&32 == 0 {
			if p.campaignOpenedDoors == nil {
				p.campaignOpenedDoors = map[int]quake.Mover{}
			}
			p.campaignOpenedDoors[m.Model] = m
			return
		}
	}
}

func (p *Planner) campaignDoorMovers(s quake.Snapshot) []quake.Mover {
	if len(p.campaignOpenedDoors) == 0 {
		return s.Movers
	}
	movers := append([]quake.Mover(nil), s.Movers...)
	seen := map[int]bool{}
	for _, m := range s.Movers {
		seen[m.Model] = true
		if old, ok := p.campaignOpenedDoors[m.Model]; ok && (quake.Distance(old.Origin, m.Origin) > 1 || old.Angles != m.Angles) {
			delete(p.campaignOpenedDoors, m.Model)
			p.routeKnown = false
		}
	}
	for model, m := range p.campaignOpenedDoors {
		if !seen[model] {
			movers = append(movers, m)
		}
	}
	return movers
}

// Replan a short walk through confirmed open doors rather than retain an AAS
// detour. This is still constrained by full hull, support, movers and hazards.
func (p *Planner) campaignOpenedRoute(s quake.Snapshot, goal quake.Vec3) ([]quake.Waypoint, bool) {
	if !p.Campaign || p.campaignDependency != nil || p.campaignUnitTrip != nil || len(p.campaignOpenedDoors) == 0 || !s.OnGround || p.Nav == nil || p.World.Geometry == nil {
		return nil, false
	}
	g := p.World.Geometry
	distance := quake.Horizontal(s.Self, goal)
	if distance < 8 || distance > 256 || math.Abs(goal[2]-s.Self[2]) > 1 || !g.PlayerMoveClear(s.Self, goal) {
		return nil, false
	}
	movers := p.campaignDoorMovers(s)
	for _, m := range movers {
		if !g.MoverHullClear(m, s.Self, goal) {
			return nil, false
		}
	}
	dx, dy := goal[0]-s.Self[0], goal[1]-s.Self[1]
	if _, reason := g.DoorMoveBlockStep(movers, s.Self, dx, dy, distance); reason != "" {
		return nil, false
	}
	for step := 0.0; step <= distance; step += 8 {
		at := s.Self
		at[0] += dx / distance * step
		at[1] += dy / distance * step
		if _, ok := cornerFooting(g, at, 1); !ok || g.PlayerTouchesHazard(at) {
			return nil, false
		}
	}
	if _, ok := cornerFooting(g, goal, 1); !ok || g.PlayerTouchesHazard(goal) || g.LaserMoveHazard(s.Self, goal) {
		return nil, false
	}
	return []quake.Waypoint{{Position: goal, Kind: 2}}, true
}
