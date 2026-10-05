package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// The protocol exposes position, not owner or fuse. React only to current
// observed grenades; do not invent a ballistic trajectory or explosion time.
func grenadeClearance(s quake.Snapshot, at quake.Vec3) float64 {
	d := math.Inf(1)
	for _, o := range s.Projectiles {
		if o.Class == "grenade" || o.Class == "hand_grenade" {
			d = math.Min(d, quake.Distance(at, o.Origin))
		}
	}
	return d
}

func (p *Planner) combatGrenadeEvade(cmd quake.UserCmd, profile *CombatSpacing) (quake.UserCmd, bool) {
	s := p.World.Snapshot
	if s.Health <= 0 || !s.OnGround || !p.World.Geometry.MovementComplete() || p.jump != nil || p.elevator != nil || p.button != nil || p.testSetupHold {
		return cmd, false
	}
	d := grenadeClearance(s, s.Self)
	if d >= 160 {
		return cmd, false
	}
	var best quake.Vec3
	score := d + 8
	for i := 0; i < 16; i++ {
		a := float64(i) * 2 * math.Pi / 16
		to := quake.Vec3{s.Self[0] + 24*math.Cos(a), s.Self[1] + 24*math.Sin(a), s.Self[2]}
		if grenadeClearance(s, to) <= score || !coverWalkClear(p.World, s.Self, to) {
			continue
		}
		if s.Teammate != nil && (quake.Horizontal(to, *s.Teammate) < 48 || quake.Distance(to, *s.Teammate) > combatLeash(profile)) {
			continue
		}
		// Reuse the already bounded detour budget, otherwise avoid moving
		// closer to any nearby observed monster while evading a grenade.
		floors := map[int]float64(nil)
		target := -1
		if p.cornerEscape != nil && p.cornerEscape.threatFloors != nil {
			floors, target = p.cornerEscape.threatFloors, p.cornerEscape.target
		}
		if !cornerDetourThreatClear(s, s.Self, to, target, 0, floors) {
			continue
		}
		best, score = to, grenadeClearance(s, to)
	}
	if score <= d+8 {
		return cmd, false
	}
	p.cornerEscape = nil // replan from the new position after this maneuver
	p.World.Command.MoveSource = "combat_grenade_evade"
	p.World.Command.MovePoint = &best
	p.World.Command.MoveLimitReason = ""
	return worldMove(cmd, s, (best[0]-s.Self[0])/24, (best[1]-s.Self[1])/24, 240, p.World.Command.AimSource == "enemy"), true
}
