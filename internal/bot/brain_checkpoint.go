package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
	"sort"
)

// PlannerCheckpoint is portable observation memory, not a native game save.
// A future save barrier must bind it to the same native state before loading.
// Motor actions, visible entities, health and inventory are never restored.
type PlannerCheckpoint struct {
	Version       int                  `json:"version"`
	Map           string               `json:"map"`
	CapturedFrame int                  `json:"captured_frame"`
	Goal          string               `json:"goal"`
	GoalPoint     *quake.Vec3          `json:"goal_point,omitempty"`
	DeathPoint    *quake.Vec3          `json:"death_point,omitempty"`
	Rendezvous    *quake.Vec3          `json:"rendezvous,omitempty"`
	Resources     []CheckpointResource `json:"resources,omitempty"`
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
	s := p.World.Snapshot
	state := PlannerCheckpoint{Version: 1, Map: s.Map, CapturedFrame: s.Frame, Goal: p.World.Goal, DeathPoint: cloneCheckpointPoint(p.deathPoint)}
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
