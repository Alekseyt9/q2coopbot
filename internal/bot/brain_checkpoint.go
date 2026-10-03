package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
	"slices"
	"sort"
)

// PlannerCheckpoint is portable observation memory, not a native game save.
// A future save barrier must bind it to the same native state before loading.
// Motor actions, visible entities, health and inventory are never restored.
type PlannerCheckpoint struct {
	Campaign      *CampaignCheckpoint  `json:"campaign,omitempty"`
	Version       int                  `json:"version"`
	Map           string               `json:"map"`
	CapturedFrame int                  `json:"captured_frame"`
	Goal          string               `json:"goal"`
	GoalPoint     *quake.Vec3          `json:"goal_point,omitempty"`
	DeathPoint    *quake.Vec3          `json:"death_point,omitempty"`
	Rendezvous    *quake.Vec3          `json:"rendezvous,omitempty"`
	Resources     []CheckpointResource `json:"resources,omitempty"`
}

type CampaignCheckpoint struct {
	Route            []string `json:"route,omitempty"`
	RouteIndex       int      `json:"route_index,omitempty"`
	PreparationSpent *int     `json:"preparation_spent_frames,omitempty"`
	Map              string   `json:"map"`
	Destination      string   `json:"destination"`
	NextMap          string   `json:"next_map"`
}

type CheckpointResource struct {
	Item        quake.Object `json:"item"`
	Age         int          `json:"age_frames"`
	Attempted   bool         `json:"attempted"`
	Unavailable bool         `json:"unavailable"`
}

func cloneCheckpointPoint(point *quake.Vec3) *quake.Vec3 {
	if point == nil {
		return nil
	}
	value := *point
	return &value
}

func (p *Planner) CaptureCheckpoint() (PlannerCheckpoint, error) {
	if p.campaignUnitTrip != nil {
		return PlannerCheckpoint{}, fmt.Errorf("checkpoint of an active campaign unit trip is not supported yet")
	}
	s := p.World.Snapshot
	state := PlannerCheckpoint{Version: 1, Map: s.Map, CapturedFrame: s.Frame, Goal: p.World.Goal, DeathPoint: cloneCheckpointPoint(p.deathPoint)}
	if p.Campaign {
		state.Campaign = &CampaignCheckpoint{Map: p.campaignMap, Destination: p.campaignDestination, NextMap: p.CampaignNextMap}
		state.Campaign.Route = append([]string(nil), p.CampaignRoute...)
		state.Campaign.RouteIndex = p.campaignRouteIndex
		if p.exitPreparation != nil {
			spent := p.exitPreparation.SpentFrames
			state.Campaign.PreparationSpent = &spent
		}
	}
	if s.Frame <= 0 || s.Map == "" || s.Health <= 0 || !s.OnGround || len(s.Projectiles) > 0 || p.jump != nil || p.elevator != nil || p.grenadeThrowPending(s) || isHandGrenade(s.Weapon) && s.GunFrame >= 1 && s.GunFrame <= 15 {
		return state, fmt.Errorf("planner checkpoint requires a living grounded idle motor state")
	}
	if p.hasGoal {
		state.GoalPoint = cloneCheckpointPoint(&p.goalPoint)
	}
	if p.respawnRegroup != nil {
		state.Rendezvous = cloneCheckpointPoint(&p.respawnRegroup.target)
	}
	for _, memory := range p.resources {
		age := s.Frame - memory.LastSeen
		if age < 0 || age > 600 {
			continue
		}
		item := memory.Item
		item.ClearShot = nil
		state.Resources = append(state.Resources, CheckpointResource{Item: item, Age: age, Attempted: memory.Attempted, Unavailable: memory.State == "unavailable"})
	}
	sort.Slice(state.Resources, func(i, j int) bool { return state.Resources[i].Item.ID < state.Resources[j].Item.ID })
	return state, state.validate(s.Map)
}

func (state PlannerCheckpoint) validate(mapName string) error {
	if state.Campaign != nil {
		c := state.Campaign
		if err := validateCampaignRoute(c.Route); err != nil {
			return err
		}
		if len(c.Route) == 0 && c.RouteIndex != 0 || len(c.Route) > 0 && (c.NextMap != "" || c.RouteIndex < 0 || c.RouteIndex >= len(c.Route) || c.Route[c.RouteIndex] != state.Map) {
			return fmt.Errorf("invalid campaign route progress")
		}
		if len(c.Route) > 0 && c.Map != "" && (c.Map != state.Map || c.RouteIndex+1 >= len(c.Route) || c.Destination != c.Route[c.RouteIndex+1]) {
			return fmt.Errorf("campaign exit differs from route")
		}
		if spent := state.Campaign.PreparationSpent; spent != nil && (*spent < 0 || *spent > exitPreparationFrames || state.Campaign.Map != state.Map) {
			return fmt.Errorf("invalid campaign preparation budget")
		}
		for _, name := range []string{state.Campaign.Map, state.Campaign.Destination, state.Campaign.NextMap} {
			if len(name) > 64 {
				return fmt.Errorf("invalid campaign checkpoint")
			}
			for _, ch := range name {
				if !(ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '_') {
					return fmt.Errorf("invalid campaign map")
				}
			}
		}
		if (state.Campaign.Map == "") != (state.Campaign.Destination == "") {
			return fmt.Errorf("incomplete campaign exit checkpoint")
		}
	}
	if state.Version != 1 || state.Map == "" || state.Map != mapName || state.CapturedFrame <= 0 || len(state.Resources) > 4096 || len(state.Goal) > 128 {
		return fmt.Errorf("planner checkpoint identity/schema mismatch")
	}
	finite := func(point quake.Vec3) bool {
		for _, v := range point {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 65536 {
				return false
			}
		}
		return true
	}
	for _, point := range []*quake.Vec3{state.GoalPoint, state.DeathPoint, state.Rendezvous} {
		if point != nil && !finite(*point) {
			return fmt.Errorf("invalid checkpoint point")
		}
	}
	ids := map[int]bool{}
	for _, memory := range state.Resources {
		if memory.Item.ID <= 0 || ids[memory.Item.ID] || !rememberedResource(memory.Item) || !finite(memory.Item.Origin) || memory.Age < 0 || memory.Age > 600 {
			return fmt.Errorf("invalid checkpoint resource")
		}
		ids[memory.Item.ID] = true
	}
	return nil
}

// Caller supplies the first fresh native snapshot AFTER load. Ages are rebased
// to its new network frame, so future observations cannot leak into the past.
// Use this on a fresh Planner; reusing an active motor is intentionally refused.
func (p *Planner) RestoreCheckpoint(state PlannerCheckpoint, fresh quake.Snapshot, root string) error {
	if err := state.validate(fresh.Map); err != nil {
		return err
	}
	if (state.Campaign != nil) != p.Campaign || state.Campaign != nil && (state.Campaign.NextMap != p.CampaignNextMap || !slices.Equal(state.Campaign.Route, p.CampaignRoute)) {
		return fmt.Errorf("campaign checkpoint/config mismatch")
	}
	if fresh.Frame <= 0 || fresh.Health <= 0 || !fresh.OnGround || p.observed || p.jump != nil || p.elevator != nil || p.grenadeThrow != nil || len(p.resources) > 0 {
		return fmt.Errorf("restore requires a fresh grounded planner")
	}
	resources := map[int]*ResourceMemory{}
	for _, memory := range state.Resources {
		// Negative rebased frames are allowed: they encode older observations.
		status := "unknown"
		if memory.Unavailable {
			status = "unavailable"
		}
		item := memory.Item
		item.ClearShot = nil
		resources[item.ID] = &ResourceMemory{Item: item, LastSeen: fresh.Frame - memory.Age, State: status, Attempted: memory.Attempted}
	}
	p.setMap(fresh.Map, root)
	if state.Campaign != nil {
		p.campaignMap = state.Campaign.Map
		p.campaignDestination = state.Campaign.Destination
		p.campaignRouteIndex = state.Campaign.RouteIndex
		if spent := state.Campaign.PreparationSpent; spent != nil {
			p.exitPreparation = &ExitPreparation{Map: fresh.Map, SpentFrames: *spent, lastFrame: fresh.Frame}
		}
	}
	p.World.Snapshot = fresh
	p.World.Goal = state.Goal
	p.deathPoint = cloneCheckpointPoint(state.DeathPoint)
	p.resources = resources
	if state.GoalPoint != nil {
		p.hasGoal = true
		p.goalPoint = *state.GoalPoint
	}
	if state.Rendezvous != nil && fresh.Teammate == nil {
		p.respawnRegroup = &respawnRegroup{target: *state.Rendezvous}
	}
	return nil
}
