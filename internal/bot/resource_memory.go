package bot

import (
	"sort"
	"strings"

	"q2coopbot/internal/quake"
)

// ResourceMemory is observation history, never an assertion that an unseen
// pickup is still available. A failed visit requires a new sighting to retry.
type ResourceMemory struct {
	Item         quake.Object `json:"item"`
	LastSeen     int          `json:"last_seen"`
	State        string       `json:"state"`
	Attempted    bool         `json:"attempted"`
	missingSince int
}

func rememberedResource(item quake.Object) bool {
	return item.Class == "item_health" || strings.HasPrefix(item.Class, "item_armor_")
}

func (p *Planner) observeResources(s quake.Snapshot) {
	if p.resources == nil {
		p.resources = map[int]*ResourceMemory{}
	}
	seen := map[int]bool{}
	for _, item := range s.Pickups {
		if !rememberedResource(item) {
			delete(p.resources, item.ID)
			continue
		}
		seen[item.ID] = true
		p.resources[item.ID] = &ResourceMemory{Item: item, LastSeen: s.Frame, State: "observed"}
	}
	for id, r := range p.resources {
		if s.Frame < r.LastSeen || s.Frame-r.LastSeen > 600 {
			delete(p.resources, id)
			continue
		}
		if seen[id] {
			continue
		}
		if r.State != "unavailable" {
			r.State = "unknown"
		}
		// Only infer absence after sustained close, unobstructed observation.
		at := healthStand(r.Item.Origin)
		g := p.World.Geometry
		if quake.Distance(s.Self, at) <= 64 && g.HasCollision() && g.ClearShot(s.Self, at) && !g.DoorShotBlocked(s.Movers, s.Self, at) {
			if r.missingSince == 0 {
				r.missingSince = s.Frame
			}
			if s.Frame-r.missingSince >= 5 {
				r.State = "unavailable"
				r.Attempted = true
			}
		} else {
			r.missingSince = 0
		}
	}
	// Bound telemetry and memory even on maps with many entities.
	for len(p.resources) > 128 {
		oldest := -1
		for id, r := range p.resources {
			if oldest < 0 || r.LastSeen < p.resources[oldest].LastSeen || r.LastSeen == p.resources[oldest].LastSeen && id < oldest {
				oldest = id
			}
		}
		delete(p.resources, oldest)
	}
}

func (p *Planner) resourceMemory() []ResourceMemory {
	result := make([]ResourceMemory, 0, len(p.resources))
	for _, r := range p.resources {
		result = append(result, *r)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Item.ID < result[j].Item.ID })
	return result
}

func (p *Planner) rememberedCandidates(s quake.Snapshot) []quake.Object {
	var result []quake.Object
	for _, r := range p.resourceMemory() {
		if r.State != "unknown" || r.Attempted || s.Frame-r.LastSeen > 600 {
			continue
		}
		at := healthStand(r.Item.Origin)
		if s.Teammate == nil || quake.Distance(*s.Teammate, at) > 384 || quake.Distance(s.Self, at) > 300 || !s.OnGround {
			continue
		}
		if _, ok := p.resourceRoute(s.Self, at); ok {
			result = append(result, r.Item)
		}
	}
	return result
}

func (p *Planner) markResourceVisit(at quake.Vec3) {
	for _, r := range p.resources {
		if healthStand(r.Item.Origin) == at && r.State == "unknown" {
			r.Attempted = true
		}
	}
}

// Memory detours are limited to supported walking routes. Existing command
// guards still check doors and dynamic obstacles on every movement command.
func (p *Planner) resourceRoute(from, at quake.Vec3) (float64, bool) {
	g := p.World.Geometry
	if p.Nav == nil || !g.HasCollision() || !g.PlayerMoveClear(at, at) {
		return 0, false
	}
	if _, ok := g.GroundDrop(at, 18); !ok && !p.Nav.GroundedNear(at) {
		return 0, false
	}
	route, ok := p.Nav.Route(from, at)
	if !ok {
		return 0, false
	}
	cost, prev := 0.0, from
	for _, wp := range route {
		if wp.Kind != 2 {
			return 0, false
		}
		cost += quake.Distance(prev, wp.Position)
		prev = wp.Position
	}
	cost += quake.Distance(prev, at)
	return cost, cost <= 512
}
