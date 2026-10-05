package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"strings"
)

// The native tongue starts 24 units ahead of the monster, then reaches 256.
// Keep another 8 units for this slow local maneuver. This is not a guarantee
// against an approaching monster or arbitrary incoming player momentum.
const parasiteFiringDistance = 256 + 24 + 8

func combatLeash(profile *CombatSpacing) float64 {
	if profile == nil {
		return 220
	}
	return min(384, max(220, profile.Minimum+32))
}

// CombatSpacing is an observed-state constraint for System1, not a promise of
// safety or a monster-health estimate. Preferred distances are policy values.
type CombatSpacing struct {
	Target         int     `json:"target"`
	Enemy          string  `json:"enemy"`
	Weapon         string  `json:"weapon"`
	Distance       float64 `json:"distance"`
	Minimum        float64 `json:"minimum"`
	PreferredMax   float64 `json:"preferred_max"`
	VisibleThreats int     `json:"visible_threats"`
	NeedSpace      bool    `json:"need_space"`
	RangeConflict  bool    `json:"weapon_range_conflict,omitempty"`
}

func combatSpacing(s quake.Snapshot) *CombatSpacing {
	var chosen *CombatSpacing
	count := 0
	for _, e := range s.Enemies {
		if e.ClearShot == nil || !*e.ClearShot {
			continue
		}
		count++
		low, high := combatDistanceBand(s.Weapon, e.Class)
		d := quake.Distance(s.Self, e.Origin)
		candidate := &CombatSpacing{Target: e.ID, Enemy: e.Class, Weapon: s.Weapon, Distance: d, Minimum: low, PreferredMax: high, NeedSpace: d < low}
		candidate.RangeConflict = high < low
		if chosen == nil || candidate.NeedSpace && (!chosen.NeedSpace || d/low < chosen.Distance/chosen.Minimum) || !candidate.NeedSpace && !chosen.NeedSpace && d < chosen.Distance {
			chosen = candidate
		}
	}
	if chosen != nil {
		chosen.VisibleThreats = count
	}
	return chosen
}

func combatDistanceBand(weapon, class string) (float64, float64) {
	weapon = strings.ToLower(weapon)
	low, high := 96.0, 512.0
	switch {
	case strings.Contains(weapon, "shotg") || strings.Contains(weapon, "shotgun"):
		low, high = 80, 192
	case isRailgun(weapon):
		low, high = 160, 900
	case strings.Contains(weapon, "rocket") || strings.Contains(weapon, "launch"):
		low, high = 192, 640
	}
	switch class {
	case "monster_parasite":
		low = math.Max(low, 320)
	case "monster_mutant":
		low = math.Max(low, 256)
	case "monster_berserk":
		low = math.Max(low, 192)
	case "monster_tank", "monster_supertank", "monster_boss2", "monster_jorg", "monster_makron":
		low = math.Max(low, 384)
	}
	if (strings.Contains(weapon, "shotg") || strings.Contains(weapon, "shotgun")) && high < low {
		// No overlap between effective shotgun range and this threat's spacing.
		// Expose that conflict instead of inventing an effective distant range.
		return low, high
	}
	return low, math.Max(high, low+96)
}
func (p *Planner) combatRetreat(cmd quake.UserCmd, profile *CombatSpacing) quake.UserCmd {
	s := p.World.Snapshot
	p.World.Command.MoveLimitReason = "combat_retreat_blocked"
	if profile == nil || !profile.NeedSpace || !s.OnGround || (s.Teammate == nil && !p.Campaign) || !p.World.Geometry.MovementComplete() || p.elevator != nil || p.button != nil || p.jump != nil {
		return cmd
	}
	if s.Teammate != nil && quake.Distance(s.Self, *s.Teammate) > combatLeash(profile) {
		return cmd
	}
	var enemy *quake.Object
	for i := range s.Enemies {
		if s.Enemies[i].ID == profile.Target {
			enemy = &s.Enemies[i]
			break
		}
	}
	if enemy == nil {
		return cmd
	}
	if escaped, ok := p.continueCornerEscape(cmd, profile, *enemy); ok {
		return escaped
	}
	if s.Teammate != nil {
		if reposition := p.combatFiringPosition(cmd, profile, *enemy); p.World.Command.MoveSource == "combat_firing_position" {
			return reposition
		}
	}
	dx, dy := s.Self[0]-enemy.Origin[0], s.Self[1]-enemy.Origin[1]
	d := math.Hypot(dx, dy)
	if d < 1 {
		return cmd
	}
	dx /= d
	dy /= d
	speeds := []float64{80}
	if p.urgentRetreat.target == enemy.ID && s.Frame <= p.urgentRetreat.until && d < parasiteFiringDistance || p.cornerUrgency.target == enemy.ID && s.Frame <= p.cornerUrgency.until && d < profile.Minimum {
		speeds = []float64{160, 80}
		p.World.Command.RetreatUrgent = true
	}
	for _, speed := range speeds {
		step := speed * 0.2
		for _, angle := range []float64{0, math.Pi / 4, -math.Pi / 4, math.Pi / 2, -math.Pi / 2} {
			x, y := dx*math.Cos(angle)-dy*math.Sin(angle), dx*math.Sin(angle)+dy*math.Cos(angle)
			next := quake.Vec3{s.Self[0] + step*x, s.Self[1] + step*y, s.Self[2]}
			if s.Teammate != nil && profile.Enemy == "monster_parasite" && quake.Horizontal(s.Self, enemy.Origin) >= parasiteFiringDistance && p.World.Command.AimSource == "enemy" {
				candidate := s
				candidate.Self = next
				to := enemy.AimPoint()
				if isRailgun(s.Weapon) {
					to = railEnd(candidate.EyePoint(), to)
				}
				if teammateBlocksShot(candidate.EyePoint(), to, *s.Teammate) {
					continue
				}
			}
			if s.Teammate != nil && ((quake.Horizontal(next, *s.Teammate) < 48 && quake.Horizontal(next, *s.Teammate) < quake.Horizontal(s.Self, *s.Teammate)+1) || quake.Distance(next, *s.Teammate) > combatLeash(profile)) {
				continue
			}
			groupSafe := true
			for _, e := range s.Enemies {
				if quake.Distance(s.Self, e.Origin) < 650 && quake.Distance(next, e.Origin) < quake.Distance(s.Self, e.Origin)-2 {
					groupSafe = false
					break
				}
			}
			if !groupSafe {
				continue
			}
			if p.World.Geometry.GroundMoveHazardStep(nil, s.Self, x, y, step) != "" {
				continue
			}
			if _, hazard := p.World.Geometry.DoorMoveBlockStep(s.Movers, s.Self, x, y, step); hazard != "" {
				continue
			}
			moverBlocked := false
			for _, mover := range s.Movers {
				if !p.World.Geometry.MoverHullClear(mover, s.Self, next) {
					moverBlocked = true
					break
				}
			}
			if moverBlocked {
				continue
			}
			p.World.Command.MoveSource = "combat_retreat"
			p.World.Command.MoveLimitReason = ""
			return worldMove(cmd, s, x, y, speed, p.World.Command.AimSource == "enemy")
		}
	}
	return p.combatCornerEscape(cmd, profile, *enemy)
}

// A short local search can trade preferred spacing for a clear firing lane.
// Only Parasite currently has a verified attack reach for this tradeoff.
func (p *Planner) combatFiringPosition(cmd quake.UserCmd, profile *CombatSpacing, enemy quake.Object) quake.UserCmd {
	s := p.World.Snapshot
	if profile.Enemy != "monster_parasite" || quake.Horizontal(s.Self, enemy.Origin) < parasiteFiringDistance || p.World.Command.LimitReason != "friendly_line_of_fire" || p.World.Command.AimEntity != enemy.ID {
		return cmd
	}
	// Keep the established coarse search first. Near a wall, the safe firing
	// lane can fall between its 22.5-degree rays; refine only when it fails.
	for _, directions := range []int{16, 64} {
		for radius := 8.0; radius <= 64; radius += 8 {
			for direction := 0; direction < directions; direction++ {
				if directions == 64 && direction%4 == 0 {
					continue
				}
				angle := float64(direction) * 2 * math.Pi / float64(directions)
				x, y := math.Cos(angle), math.Sin(angle)
				next := s.Self
				safe := true
				for step := 8.0; step <= radius; step += 8 {
					if p.World.Geometry.GroundMoveHazardStep(nil, next, x, y, 8) != "" {
						safe = false
						break
					}
					if _, hazard := p.World.Geometry.DoorMoveBlockStep(s.Movers, next, x, y, 8); hazard != "" {
						safe = false
						break
					}
					next[0] += 8 * x
					next[1] += 8 * y
					if quake.Horizontal(next, enemy.Origin) < parasiteFiringDistance || quake.Horizontal(next, *s.Teammate) < 48 || quake.Distance(next, *s.Teammate) > combatLeash(profile) {
						safe = false
						break
					}
					for _, other := range s.Enemies {
						if other.ID != enemy.ID && quake.Distance(s.Self, other.Origin) < 650 && quake.Distance(next, other.Origin) < quake.Distance(s.Self, other.Origin)-2 {
							safe = false
							break
						}
					}
					if !safe {
						break
					}
				}
				candidate := s
				candidate.Self = next
				from, to := candidate.EyePoint(), enemy.AimPoint()
				if isRailgun(s.Weapon) {
					to = railEnd(from, to)
				}
				if !safe || teammateBlocksShot(from, to, *s.Teammate) || !p.World.Geometry.ClearShot(from, enemy.AimPoint()) || p.World.Geometry.DoorShotBlocked(s.Movers, from, enemy.AimPoint()) {
					continue
				}
				p.World.Command.MoveSource = "combat_firing_position"
				p.World.Command.MoveLimitReason = ""
				return worldMove(cmd, s, x, y, 80, false)
			}
		}
	}
	return cmd
}
