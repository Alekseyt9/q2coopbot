package bot

import (
	"math"
	"strings"

	"q2coopbot/internal/quake"
)

// ResourceYield records proximity, not an inferred teammate inventory or need.
type ResourceYield struct {
	Entity int    `json:"entity"`
	Class  string `json:"class"`
	Reason string `json:"reason"`
}

func teammatePickupPriority(s quake.Snapshot, class string, at quake.Vec3) bool {
	// Coop weapons may stay for each player. Only compete for consumable supplies.
	if s.Teammate == nil || (!strings.HasPrefix(class, "ammo_") && !strings.HasPrefix(class, "item_armor_")) {
		return false
	}
	d := quake.Distance(*s.Teammate, at)
	return math.Abs((*s.Teammate)[2]-at[2]) <= 32 && d <= 96 && d+32 < quake.Distance(s.Self, at)
}

func (p *Planner) yieldPickup(s quake.Snapshot, id int, class string, at quake.Vec3) bool {
	if !teammatePickupPriority(s, class, at) || !p.World.Geometry.HasCollision() ||
		!p.World.Geometry.ClearShot(*s.Teammate, at) || p.World.Geometry.DoorShotBlocked(s.Movers, *s.Teammate, at) {
		return false
	}
	p.World.ResourceYield = &ResourceYield{Entity: id, Class: class, Reason: "teammate_closer"}
	return true
}
