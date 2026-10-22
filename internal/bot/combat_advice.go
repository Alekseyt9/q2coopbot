package bot

import (
	"encoding/json"
	"time"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// CombatAdvice is a hypothetical planner-stage recommendation, never an applied
// or native-matched command. The reactive expert does not replay route/cover state.
type CombatAdvice struct {
	Version   string          `json:"version"`
	Scope     string          `json:"scope"`
	Identity  policy.Identity `json:"identity"`
	Available bool            `json:"available"`
	Reason    string          `json:"reason,omitempty"`
	Command   quake.UserCmd   `json:"recommended_command"`
	Action    policy.Action   `json:"recommended_action"`
	Decision  CommandDecision `json:"decision"`
}

// Only commands actually sent by the learner update this history. Hypothetical
// pauses must not reset the burst while the learner continues holding attack.
type combatAdviceHistory struct {
	last  policy.Identity
	burst machinegunBurst
}

func (h *combatAdviceHistory) observe(o policy.Observation, cmd quake.UserCmd) {
	if !policy.SameLife(h.last, o.Identity) || o.Identity.Frame != h.last.Frame+1 {
		h.burst = machinegunBurst{}
	}
	h.last = o.Identity
	if !isMachinegun(o.Weapon) || o.Health <= 0 {
		h.burst = machinegunBurst{}
		return
	}
	b := &h.burst
	b.mapName, b.lastFrame = o.Identity.Map, o.Identity.Frame
	if cmd.Buttons&1 == 0 {
		if o.Identity.Frame-b.lastAttack >= 2 {
			b.start = 0
		}
		return
	}
	if b.start == 0 {
		b.start = o.Identity.Frame
	}
	b.lastAttack = o.Identity.Frame
}

func (p *Planner) reactiveCombatAdvice(o policy.Observation, prev quake.UserCmd, now time.Time, h combatAdviceHistory) CombatAdvice {
	a := CombatAdvice{Version: "reactive_combat_advice_v2", Identity: o.Identity,
		Scope: "hypothetical reactive planner-stage advice; not applied or native-matched; not full stateful rules replay; no weapon-selection label"}
	if !p.testCombatOnly || p.World.Goal != "combat_fixture" || p.testSetupHold {
		a.Reason = "outside_isolated_combat"
		return a
	}
	if o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 || o.Identity.Frame != p.World.Snapshot.Frame {
		a.Reason = "invalid_observation"
		return a
	}
	if o.Weapon != "Blaster" && !directHitscanWeapon(o.Weapon) {
		a.Reason = "unsupported_equipment"
		return a
	}
	if p.World.Geometry == nil || p.World.GeometryStatus == "unavailable" || p.World.GeometryStatus == "incomplete" {
		a.Reason = "geometry_unavailable"
		return a
	}
	// BSP/Nav data are immutable queries. All observed mutable slices, maps and
	// pointers are copied, rather than shallow-copying the live Planner.
	w := p.World
	w.Geometry = nil
	b, err := json.Marshal(w)
	if err != nil {
		a.Reason = "world_copy_failed"
		return a
	}
	var copied World
	if err := json.Unmarshal(b, &copied); err != nil {
		a.Reason = "world_copy_failed"
		return a
	}
	copied.Geometry = p.World.Geometry
	q := Planner{World: copied, Nav: p.Nav, GameClock: p.GameClock, Campaign: p.Campaign,
		testCombatOnly: true, TestDisableProbe: true, TestDisableHandGrenade: true,
		TestDisableProjectileLead: p.TestDisableProjectileLead,
		urgentRetreat:             p.urgentRetreat, cornerUrgency: p.cornerUrgency,
		shotTeammateMotion: p.shotTeammateMotion}
	q.enemyMotion = make(map[int]enemyMotion, len(p.enemyMotion))
	for id, motion := range p.enemyMotion {
		q.enemyMotion[id] = motion
	}
	if policy.SameLife(h.last, o.Identity) && h.last.Frame+1 == o.Identity.Frame {
		q.machinegunBurst = h.burst
	}
	a.Command = q.commandAt(prev, now)
	a.Command.Msec = 100
	a.Decision = q.World.Command
	a.Action = policy.FromCommand(o, a.Command, p.World.Snapshot.DeltaAngles, "")
	if q.World.Command.AimEntity > 0 {
		mask, err := policy.TargetAvailability(o)
		if err != nil {
			a.Reason = "target_mask_invalid"
			return a
		}
		for i, e := range policy.TargetEnemies(o) {
			if i+1 < len(mask) && mask[i+1] && e.ID == q.World.Command.AimEntity && e.Track != nil && *e.Track > 0 {
				a.Action.TargetEntity, a.Action.TargetTrack = e.ID, *e.Track
				a.Available = true
				return a
			}
		}
		a.Reason = "expert_target_unavailable_to_policy"
		return a
	}
	if q.World.Command.AimSource != "none" {
		a.Reason = "non_monster_expert_aim"
		return a
	}
	a.Available = true
	return a
}
