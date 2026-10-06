// Package policy defines the low-level combat boundary. It has no server reward
// or hidden monster state and does not implement training or tactical assistance.
package policy

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"q2coopbot/internal/quake"
)

const ObservationVersion = "combat_observation_v3"
const ActionVersion = "combat_action_v1"

// Identity prevents a delayed decision from crossing a frame or connection.
type Identity struct {
	Life       int    `json:"life"`
	Map        string `json:"map"`
	Connection int    `json:"connection"`
	Spawncount int    `json:"spawncount"`
	Actor      int    `json:"actor"`
	Frame      int    `json:"frame"`
}

type Enemy struct {
	ModelPath       string      `json:"observed_model,omitempty"`
	Skin            *uint32     `json:"observed_skin,omitempty"`
	Animation       *int        `json:"observed_animation_frame,omitempty"`
	Angles          *quake.Vec3 `json:"observed_angles,omitempty"`
	Solid           *uint16     `json:"observed_solid,omitempty"`
	Distance        float64     `json:"distance"`
	MotionDirection *quake.Vec3 `json:"motion_direction"`
	Track           *int        `json:"observed_track"`
	Velocity        *quake.Vec3 `json:"observed_velocity"`
	ID              int         `json:"id"`
	Class           string      `json:"class"`
	Relative        quake.Vec3  `json:"relative"`
	ClearShot       *bool       `json:"clear_shot"`
}

// Nil masks distinguish unavailable information from an observed zero.
// Tracks describe uninterrupted visible observations, not server generations.
// Client/BSP-derived features are separate from offline server reward telemetry.
type Observation struct {
	Composition        *[]MonsterCount        `json:"visible_monster_composition,omitempty"`
	Geometry           *LocalGeometry         `json:"local_geometry"`
	Projectiles        *[]Enemy               `json:"visible_projectiles"`
	Pickups            *[]NearbyObject        `json:"visible_pickups"`
	Props              *[]NearbyObject        `json:"visible_props"`
	Movers             *[]NearbyMover         `json:"visible_movers"`
	Beams              *[]NearbyBeam          `json:"visible_beams"`
	History            []HistoryFrame         `json:"history"`
	AgeMS              int64                  `json:"observation_age_ms"`
	Version            string                 `json:"version"`
	Identity           Identity               `json:"identity"`
	Health             int16                  `json:"health"`
	Armor              int16                  `json:"armor"`
	Position           quake.Vec3             `json:"position"`
	Velocity           quake.Vec3             `json:"velocity"`
	OnGround           bool                   `json:"on_ground"`
	Ducked             bool                   `json:"ducked"`
	ViewAngles         [3]int16               `json:"view_angles"`
	KickAngles         *quake.Vec3            `json:"kick_angles_degrees,omitempty"`
	Weapon             string                 `json:"weapon"`
	Ammo               int16                  `json:"ammo"`
	GunFrame           int                    `json:"gun_frame"`
	Inventory          *[]quake.InventoryItem `json:"inventory"`
	InventoryAgeFrames *int                   `json:"inventory_age_frames"`
	Teammate           *quake.Vec3            `json:"teammate_relative"`
	Enemies            []Enemy                `json:"enemies"`
	PreviousCommand    quake.UserCmd          `json:"previous_applied_command"`
}

type Action struct {
	Version    string   `json:"version"`
	Identity   Identity `json:"identity"`
	Forward    float64  `json:"forward"`
	Side       float64  `json:"side"`
	YawDelta   float64  `json:"yaw_delta_degrees"`
	PitchDelta float64  `json:"pitch_delta_degrees"`
	Attack     bool     `json:"attack"`
	Vertical   string   `json:"vertical"` // release, jump, crouch
	Weapon     string   `json:"weapon"`   // empty keeps the current weapon
}

// Provider supplies actions directly, without calling the rules controller.
// No learned provider is shipped yet.
type Provider interface {
	Decide(Observation) (Action, error)
	Version() string
}

func Observe(s quake.Snapshot, id Identity, previous quake.UserCmd) Observation {
	o := Observation{Version: ObservationVersion, Identity: id, Health: s.Health, Armor: s.Armor,
		Position: s.Self, Velocity: s.SelfVelocity, OnGround: s.OnGround, Ducked: s.Ducked,
		ViewAngles: s.ViewAngles, Weapon: s.Weapon, Ammo: s.Ammo, GunFrame: s.GunFrame,
		PreviousCommand: previous, Enemies: []Enemy{}}
	if s.KickAnglesKnown {
		kick := s.KickAngles
		o.KickAngles = &kick
	}
	if s.InventoryKnown {
		items := append([]quake.InventoryItem{}, s.Inventory...)
		o.Inventory, o.InventoryAgeFrames = &items, &s.InventoryAgeFrames
	}
	if s.Teammate != nil {
		v := relative(*s.Teammate, s.Self)
		o.Teammate = &v
	}
	counts := map[string]int{}
	visible := append([]quake.Object(nil), s.Enemies...)
	sort.SliceStable(visible, func(i, j int) bool {
		return quake.Distance(visible[i].Origin, s.Self) < quake.Distance(visible[j].Origin, s.Self)
	})
	for _, e := range visible {
		// PVS alone is not visibility. Do not expose positions behind a wall.
		if e.ClearShot == nil || !*e.ClearShot {
			continue
		}
		if len(o.Enemies) == 8 {
			counts[MonsterType(e.Class, e.ModelPath)]++
			continue
		}
		counts[MonsterType(e.Class, e.ModelPath)]++
		solid := e.Solid
		enemy := Enemy{ModelPath: e.ModelPath, Solid: &solid, ID: e.ID, Class: e.Class, Relative: relative(e.Origin, s.Self), Distance: quake.Distance(e.Origin, s.Self), ClearShot: e.ClearShot}
		if e.ModelPath != "" {
			skin, frame, angles := e.Skin, e.Frame, e.Angles
			enemy.Skin = &skin
			enemy.Animation = &frame
			enemy.Angles = &angles
		}
		o.Enemies = append(o.Enemies, enemy)
	}
	composition := []MonsterCount{}
	for _, kind := range monsterTypes {
		if counts[kind] > 0 {
			composition = append(composition, MonsterCount{kind, counts[kind]})
		}
	}
	o.Composition = &composition
	return o
}

func relative(a, b quake.Vec3) quake.Vec3 { return quake.Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }
func degrees(v int16) float64             { return float64(v) * 360 / 65536 }
func delta(to, from int16) float64        { return degrees(int16(uint16(to) - uint16(from))) }

// FromCommand labels the teacher's command. This is not a prediction by a net.
func FromCommand(o Observation, cmd quake.UserCmd, deltaAngles [3]int16, weapon string) Action {
	vertical := "release"
	if cmd.Up > 0 {
		vertical = "jump"
	} else if cmd.Up < 0 {
		vertical = "crouch"
	}
	return Action{Version: ActionVersion, Identity: o.Identity,
		Forward: float64(cmd.Forward) / 400, Side: float64(cmd.Side) / 400,
		YawDelta:   delta(int16(uint16(cmd.Yaw)+uint16(deltaAngles[1])), o.ViewAngles[1]),
		PitchDelta: delta(int16(uint16(cmd.Pitch)+uint16(deltaAngles[0])), o.ViewAngles[0]),
		Attack:     cmd.Buttons&1 != 0, Vertical: vertical, Weapon: strings.TrimPrefix(weapon, "use ")}
}

// Command validates and converts a decision; it never aims or selects a tactic.
// Collision, shot and freshness guards still belong to the caller. This adapter
// alone is not approval to enable learned control in live gameplay.
func Command(o Observation, a Action, deltaAngles [3]int16) (quake.UserCmd, error) {
	if o.Version != ObservationVersion || a.Version != ActionVersion || a.Identity != o.Identity || o.Identity.Frame <= 0 || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		return quake.UserCmd{}, fmt.Errorf("policy version, identity or life mismatch")
	}
	for _, v := range []float64{a.Forward, a.Side, a.YawDelta, a.PitchDelta} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return quake.UserCmd{}, fmt.Errorf("nonfinite policy action")
		}
	}
	if math.Abs(a.Forward) > 1 || math.Abs(a.Side) > 1 || math.Abs(a.YawDelta) > 180 || math.Abs(a.PitchDelta) > 180 {
		return quake.UserCmd{}, fmt.Errorf("policy action outside v1 bounds")
	}
	if a.Vertical != "release" && a.Vertical != "jump" && a.Vertical != "crouch" {
		return quake.UserCmd{}, fmt.Errorf("unknown vertical action")
	}
	if a.Weapon != "" {
		allowed := false
		for _, name := range []string{"Blaster", "Shotgun", "Super Shotgun", "Machinegun", "Chaingun", "Grenade Launcher", "Rocket Launcher", "HyperBlaster", "Railgun", "BFG10K", "Grenades"} {
			if a.Weapon == name && o.Inventory != nil && o.InventoryAgeFrames != nil && *o.InventoryAgeFrames >= 0 && *o.InventoryAgeFrames <= 20 {
				for _, item := range *o.Inventory {
					if item.Name == name && item.Count > 0 {
						allowed = true
					}
				}
			}
		}
		if !allowed {
			return quake.UserCmd{}, fmt.Errorf("weapon unavailable or inventory stale")
		}
	}
	pitch := math.Max(-89, math.Min(89, degrees(o.ViewAngles[0])+a.PitchDelta))
	cmd := quake.UserCmd{Msec: 100, Forward: int16(math.Round(a.Forward * 400)), Side: int16(math.Round(a.Side * 400)),
		Yaw:   int16(uint16(o.ViewAngles[1]) + uint16(int16(math.Round(a.YawDelta*65536/360))) - uint16(deltaAngles[1])),
		Pitch: int16(uint16(int16(math.Round(pitch*65536/360))) - uint16(deltaAngles[0]))}
	if a.Attack {
		cmd.Buttons = 1
	}
	if a.Vertical == "jump" {
		cmd.Up = 400
	} else if a.Vertical == "crouch" {
		cmd.Up = -400
	}
	return cmd, nil
}

// ControlChanged excludes server timing/light metadata from control changes.
func ControlChanged(a, b quake.UserCmd) bool {
	return a.Pitch != b.Pitch || a.Yaw != b.Yaw || a.Roll != b.Roll ||
		a.Forward != b.Forward || a.Side != b.Side || a.Up != b.Up ||
		a.Buttons != b.Buttons || a.Impulse != b.Impulse
}

type Capture struct {
	TeacherAimSource string        `json:"teacher_aim_source,omitempty"`
	TeacherAimEntity int           `json:"teacher_aim_entity,omitempty"`
	TeacherPrimitive string        `json:"teacher_primitive,omitempty"`
	ClientSequence   uint32        `json:"client_sequence"`
	Selection        *Selection    `json:"selection,omitempty"`
	CommandAtUnixNS  int64         `json:"command_at_unix_ns"`
	Provider         string        `json:"provider"`
	Observation      Observation   `json:"observation"`
	Proposed         Action        `json:"proposed"`
	Applied          Action        `json:"applied"`
	ProposedCommand  quake.UserCmd `json:"proposed_command"`
	AppliedCommand   quake.UserCmd `json:"applied_command"`
	Changed          bool          `json:"command_changed"`
	LimitReason      string        `json:"limit_reason,omitempty"`
	MoveLimitReason  string        `json:"move_limit_reason,omitempty"`
	LabelQuality     string        `json:"label_quality"`
}

type Intervention struct {
	Component string `json:"component"`
	Reason    string `json:"reason"`
}

type Selection struct {
	InferenceBudgetExceeded bool           `json:"inference_budget_exceeded,omitempty"`
	CombatContinuation      bool           `json:"combat_continuation,omitempty"`
	Sample                  *Sample        `json:"sample,omitempty"`
	Mode                    string         `json:"mode"`
	Owner                   string         `json:"owner"`
	ProviderVersion         string         `json:"provider_version,omitempty"`
	Candidate               *Action        `json:"candidate,omitempty"`
	CandidateCommand        *quake.UserCmd `json:"candidate_command,omitempty"`
	GuardedCommand          *quake.UserCmd `json:"guarded_command,omitempty"`
	Interventions           []Intervention `json:"interventions,omitempty"`
	Fallback                string         `json:"fallback,omitempty"`
	ElapsedUS               int64          `json:"elapsed_us"`
}
