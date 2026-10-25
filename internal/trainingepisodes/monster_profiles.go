package trainingepisodes

import (
	"math"
	"q2coopbot/internal/quake"
)

const nativeProfileKind = "campaign-native-profile-combat-v1"
const nativeProfileGeometry = "stock_monster_hulls_habitats_v1"

type MonsterProfile struct {
	Min     quake.Vec3 `json:"min"`
	Max     quake.Vec3 `json:"max"`
	Health  int        `json:"health"`
	Habitat string     `json:"habitat"`
}

// Stock SP_monster_* spawn hulls/health, not corpse or ducking hulls.
// boss3_stand and insane are noncombat scenery, not substitute enemies.
var monsterProfiles = map[string]MonsterProfile{
	"monster_soldier_light":  {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 20, "ground"},
	"monster_soldier":        {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 30, "ground"},
	"monster_soldier_ss":     {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 40, "ground"},
	"monster_infantry":       {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 100, "ground"},
	"monster_gunner":         {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 175, "ground"},
	"monster_parasite":       {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 24}, 175, "ground"},
	"monster_berserk":        {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 240, "ground"},
	"monster_brain":          {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 300, "ground"},
	"monster_chick":          {quake.Vec3{-16, -16, 0}, quake.Vec3{16, 16, 56}, 175, "ground"},
	"monster_medic":          {quake.Vec3{-24, -24, -24}, quake.Vec3{24, 24, 32}, 300, "ground"},
	"monster_mutant":         {quake.Vec3{-32, -32, -24}, quake.Vec3{32, 32, 48}, 300, "ground"},
	"monster_gladiator":      {quake.Vec3{-32, -32, -24}, quake.Vec3{32, 32, 64}, 400, "ground"},
	"monster_tank":           {quake.Vec3{-32, -32, -16}, quake.Vec3{32, 32, 72}, 750, "ground"},
	"monster_tank_commander": {quake.Vec3{-32, -32, -16}, quake.Vec3{32, 32, 72}, 1000, "ground"},
	"monster_supertank":      {quake.Vec3{-64, -64, 0}, quake.Vec3{64, 64, 112}, 1500, "ground"},
	"monster_jorg":           {quake.Vec3{-80, -80, 0}, quake.Vec3{80, 80, 140}, 3000, "ground"},
	"monster_makron":         {quake.Vec3{-30, -30, 0}, quake.Vec3{30, 30, 90}, 3000, "ground"},
	"monster_flyer":          {quake.Vec3{-16, -16, -24}, quake.Vec3{16, 16, 32}, 50, "air"},
	"monster_floater":        {quake.Vec3{-24, -24, -24}, quake.Vec3{24, 24, 32}, 200, "air"},
	"monster_hover":          {quake.Vec3{-24, -24, -24}, quake.Vec3{24, 24, 32}, 240, "air"},
	"monster_boss2":          {quake.Vec3{-56, -56, 0}, quake.Vec3{56, 56, 80}, 2000, "air"},
	"monster_flipper":        {quake.Vec3{-16, -16, 0}, quake.Vec3{16, 16, 32}, 50, "water"},
}

// StockMonsterProfiles exposes a copy for catalog/audit tooling.
func StockMonsterProfiles() map[string]MonsterProfile {
	result := make(map[string]MonsterProfile, len(monsterProfiles))
	for class, profile := range monsterProfiles {
		result[class] = profile
	}
	return result
}

func groundForHull(world *quake.MapInfo, p quake.Vec3, minZ, maxDrop float64) (quake.Vec3, bool) {
	probe := p
	probe[2] += minZ + 24 + 48
	drop, ok := world.GroundDrop(probe, maxDrop)
	if !ok {
		return p, false
	}
	p[2] = math.Round((probe[2]-drop-24-minZ)*8)/8 + .25
	return p, true
}

func profileBounds(p quake.Vec3, profile MonsterProfile) (quake.Vec3, quake.Vec3) {
	lo, hi := p, p
	for a := range p {
		lo[a] += profile.Min[a]
		hi[a] += profile.Max[a]
	}
	return lo, hi
}

func boundsOverlap(a, b, c, d quake.Vec3, margin float64) bool {
	for axis := range a {
		if b[axis]+margin <= c[axis] || d[axis]+margin <= a[axis] {
			return false
		}
	}
	return true
}

func checkProfileStart(v Instance, world *quake.MapInfo) string {
	points := []quake.Vec3{v.Player}
	profiles := []MonsterProfile{{Min: quake.Vec3{-16, -16, -24}, Max: quake.Vec3{16, 16, 32}, Habitat: "ground"}}
	for _, m := range v.Monsters {
		p, ok := monsterProfiles[m.Class]
		if !ok {
			return "unknown_native_profile"
		}
		points = append(points, m.Position)
		profiles = append(profiles, p)
	}
	for i, p := range points {
		profile := profiles[i]
		lo, hi := profileBounds(p, profile)
		if clear, valid := world.ProjectileBoxClear(lo, hi); !valid || !clear {
			return "static_profile_hull_collision"
		}
		if profile.Habitat == "ground" {
			probe := p
			probe[2] += profile.Min[2] + 24
			if drop, ok := world.GroundDrop(probe, 2); !ok || drop > 1 {
				return "unsupported_profile_floor"
			}
		}
		for _, x := range []float64{lo[0] + .125, hi[0] - .125} {
			for _, y := range []float64{lo[1] + .125, hi[1] - .125} {
				for _, z := range []float64{lo[2] + .125, hi[2] - .125} {
					contents, ok := world.StaticPointContents(quake.Vec3{x, y, z})
					if !ok {
						return "unknown_profile_contents"
					}
					if contents&(8|16) != 0 {
						return "lethal_profile_contents"
					}
					if profile.Habitat == "water" && contents&32 == 0 {
						return "swimmer_outside_water"
					}
				}
			}
		}
		if i == 0 {
			trace := world.TracePlayer(p, p, false)
			if !trace.Valid || trace.StartSolid || trace.AllSolid {
				return "native_profile_playerclip"
			}
			for _, dx := range []float64{-15, 15} {
				for _, dy := range []float64{-15, 15} {
					foot := p
					foot[0] += dx
					foot[1] += dy
					if drop, ok := world.GroundDrop(foot, 2); !ok || drop > 1 {
						return "uneven_profile_player_floor"
					}
				}
			}
			raised := p
			raised[2] += 10
			trace = world.TracePlayer(p, raised, false)
			if !trace.Valid || trace.StartSolid || trace.AllSolid || trace.Fraction < 1 {
				return "player_clip_collision"
			}
			top := hi
			top[2] += 10
			if clear, valid := world.ProjectileBoxClear(lo, top); !clear || !valid {
				return "teleport_headroom"
			}
		}
		for _, e := range world.Entities {
			if e.Class == "misc_explobox" || e.Class == "misc_insane" || e.Class == "misc_deadsoldier" {
				prop := MonsterProfile{Min: quake.Vec3{-16, -16, -24}, Max: quake.Vec3{16, 16, 32}}
				if e.Class == "misc_explobox" {
					prop.Min[2] = 0
					prop.Max[2] = 40
				}
				a, b := profileBounds(e.Origin, prop)
				if boundsOverlap(lo, hi, a, b, 2) {
					return "profile_solid_prop_overlap"
				}
			}
			switch e.Class {
			case "func_wall", "func_door", "func_door_secret", "func_door_rotating", "func_plat", "func_train", "func_rotating", "func_button", "func_explosive", "func_object", "func_conveyor", "trigger_hurt", "trigger_changelevel", "trigger_teleport", "trigger_push", "trigger_gravity", "trigger_monsterjump":
			default:
				continue
			}
			b, ok := world.TouchBounds(e)
			if ok && boundsOverlap(lo, hi, b.Min, b.Max, 0) {
				return "inline_profile_hull_overlap"
			}
		}
		for j := 0; j < i; j++ {
			a, b := profileBounds(points[j], profiles[j])
			if boundsOverlap(lo, hi, a, b, 8) {
				return "profile_actor_overlap"
			}
		}
		if i >= 2 {
			for _, spawn := range world.Entities {
				if spawn.Class == "info_player_start" || spawn.Class == "info_player_coop" {
					a, b := profileBounds(spawn.Origin, profiles[0])
					b[2] += 10
					if boundsOverlap(lo, hi, a, b, 8) {
						return "profile_native_spawn_overlap"
					}
				}
			}
		}
	}
	if len(v.Monsters) == 0 {
		return "missing_profile_target"
	}
	eye, target := v.Player, v.Monsters[0].Position
	eye[2] += 22
	target[2] += (profiles[1].Min[2] + profiles[1].Max[2]) / 2
	a, ak := world.StaticArea(eye)
	b, bk := world.StaticArea(target)
	if !ak || !bk || a != b {
		return "profile_primary_area_portal_dependency"
	}
	if possible, known := world.PointPVS(eye, target); known && !possible {
		return "profile_primary_point_pvs"
	}
	if !world.ClearShot(eye, target) || world.DoorShotBlocked(nil, eye, target) {
		return "profile_primary_occluded"
	}
	// The observation contract uses Object.AimPoint, rather than bbox center.
	// Both must be clear before an observed-reset fixture can be admitted.
	aim := v.Monsters[0].Position
	aim[2] += math.Min(math.Max(22, profiles[1].Min[2]+8), profiles[1].Max[2]-8)
	if !world.ClearShot(eye, aim) || world.DoorShotBlocked(nil, eye, aim) {
		return "profile_observed_aim_occluded"
	}
	return ""
}
