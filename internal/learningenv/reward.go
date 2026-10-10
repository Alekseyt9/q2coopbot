package learningenv

import (
	"fmt"
	"math"
	"strings"

	"q2coopbot/internal/policy"
)

const RewardVersion = "combat_reward_v1"
const KillRewardVersion = "combat_reward_v2"
const AimRewardVersion = "combat_reward_v3"
const ManeuverRewardVersion = "combat_reward_v4"
const ActionQualityRewardVersion = "combat_reward_v6"
const TargetSequenceRewardVersion = "combat_reward_v7"
const SelectedAimRewardVersion = "combat_reward_v8"
const NativeWasteRewardVersion = "combat_reward_v9"
const NavigationRewardVersion = "combat_reward_v10"
const CoopRewardVersion = "combat_reward_v11"

// RewardConfig is an explicit experimental objective, not a measured success
// metric. Server effects never enter the policy observation.
type RewardConfig struct {
	Version             string  `json:"version"`
	MonsterDamage       float64 `json:"monster_damage"`
	ReceivedDamage      float64 `json:"received_damage"`
	SelfDamage          float64 `json:"self_damage_extra"`
	FriendlyDamage      float64 `json:"friendly_damage"`
	Death               float64 `json:"death"`
	Tick                float64 `json:"tick"`
	MonsterKill         float64 `json:"monster_kill,omitempty"`
	AimPotential        float64 `json:"aim_potential,omitempty"`
	AimGamma            float64 `json:"aim_gamma,omitempty"`
	AimKickAngles       bool    `json:"aim_kick_angles,omitempty"`
	SpacingPotential    float64 `json:"spacing_potential,omitempty"`
	ParasiteRange       float64 `json:"parasite_range,omitempty"`
	BlasterMiss         float64 `json:"blaster_miss,omitempty"`
	MachinegunMiss      float64 `json:"machinegun_miss,omitempty"`
	OffTargetAttack     float64 `json:"off_target_attack,omitempty"`
	TurnAway            float64 `json:"turn_away,omitempty"`
	StalledMovement     float64 `json:"stalled_movement,omitempty"`
	TargetChurn         float64 `json:"target_churn,omitempty"`
	NavigationPotential float64 `json:"navigation_potential,omitempty"`
	PeerDeath           float64 `json:"peer_death,omitempty"`
}

func (c RewardConfig) HasKillReward() bool {
	return c.Version == KillRewardVersion || c.HasPotentialReward()
}
func (c RewardConfig) HasPotentialReward() bool {
	return c.Version == AimRewardVersion || c.Version == ManeuverRewardVersion || c.Version == MissRewardVersion || c.HasActionQualityCosts()
}

func (c RewardConfig) HasActionQualityCosts() bool {
	return c.Version == ActionQualityRewardVersion || c.Version == TargetSequenceRewardVersion || c.Version == SelectedAimRewardVersion || c.Version == NativeWasteRewardVersion || c.HasNavigationReward()
}

func (c RewardConfig) HasNavigationReward() bool {
	return c.Version == NavigationRewardVersion || c.Version == CoopRewardVersion
}

func (c RewardConfig) HasProjectileMissCost() bool {
	return c.Version == MissRewardVersion || c.Version == NativeWasteRewardVersion
}

func (c RewardConfig) Validate() error {
	if c.Version != RewardVersion && !c.HasKillReward() {
		return fmt.Errorf("unsupported reward version")
	}
	for _, v := range []float64{c.MonsterDamage, c.ReceivedDamage, c.SelfDamage, c.FriendlyDamage, c.Death, c.Tick, c.MonsterKill, c.AimPotential, c.AimGamma, c.SpacingPotential, c.ParasiteRange, c.BlasterMiss, c.MachinegunMiss, c.OffTargetAttack, c.TurnAway, c.StalledMovement, c.TargetChurn, c.NavigationPotential, c.PeerDeath} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("nonfinite reward coefficient")
		}
	}
	if c.Version == CoopRewardVersion {
		if c.PeerDeath >= 0 || c.PeerDeath < -10 {
			return fmt.Errorf("v11 requires peer death cost in [-10,0)")
		}
	} else if c.PeerDeath != 0 {
		return fmt.Errorf("peer death cost requires reward v11")
	}
	if c.HasNavigationReward() {
		if c.NavigationPotential <= 0 || c.NavigationPotential > .5 {
			return fmt.Errorf("v10 requires navigation potential in (0,.5]")
		}
	} else if c.NavigationPotential != 0 {
		return fmt.Errorf("navigation shaping requires reward v10")
	}
	if c.Version == RewardVersion && c.MonsterKill != 0 || c.HasKillReward() && c.MonsterKill <= 0 {
		return fmt.Errorf("v1 forbids kill bonus; v2 requires positive kill bonus")
	}
	if c.HasPotentialReward() {
		if c.AimPotential <= 0 || c.AimPotential > 1 || c.AimGamma <= 0 || c.AimGamma >= 1 {
			return fmt.Errorf("v3 requires bounded positive aim potential and discount in (0,1)")
		}
	} else if c.AimPotential != 0 || c.AimGamma != 0 || c.AimKickAngles {
		return fmt.Errorf("aim shaping requires reward v3")
	}
	if c.HasProjectileMissCost() {
		if c.BlasterMiss >= 0 || c.BlasterMiss < -0.1 {
			return fmt.Errorf("v5 requires bounded negative blaster miss cost")
		}
	} else if c.BlasterMiss != 0 {
		return fmt.Errorf("blaster miss cost requires reward v5")
	}
	if c.Version == NativeWasteRewardVersion {
		if c.MachinegunMiss >= 0 || c.MachinegunMiss < -.1 {
			return fmt.Errorf("v9 requires bounded negative MG miss cost")
		}
	} else if c.MachinegunMiss != 0 {
		return fmt.Errorf("MG miss cost requires reward v9")
	}
	for _, cost := range []float64{c.OffTargetAttack, c.TurnAway, c.StalledMovement} {
		if c.HasActionQualityCosts() {
			if cost >= 0 || cost < -.05 {
				return fmt.Errorf("v6 requires action quality costs in [-.05,0)")
			}
		} else if cost != 0 {
			return fmt.Errorf("action quality costs require reward v6")
		}
	}
	if c.Version == TargetSequenceRewardVersion {
		if c.TargetChurn >= 0 || c.TargetChurn < -.05 {
			return fmt.Errorf("v7 requires target churn cost in [-.05,0)")
		}
	} else if c.TargetChurn != 0 {
		return fmt.Errorf("target churn cost requires reward v7")
	}
	if c.Version == ManeuverRewardVersion || c.Version == MissRewardVersion || c.HasActionQualityCosts() {
		if c.SpacingPotential <= 0 || c.SpacingPotential > 2 || c.ParasiteRange < 280 || c.ParasiteRange > 512 {
			return fmt.Errorf("invalid maneuver potential scale/range")
		}
	} else if c.SpacingPotential != 0 || c.ParasiteRange != 0 {
		return fmt.Errorf("spacing shaping requires reward v4")
	}
	if c.MonsterDamage <= 0 || c.ReceivedDamage >= 0 || c.SelfDamage > 0 || c.FriendlyDamage >= 0 || c.Death >= 0 || c.Tick >= 0 {
		return fmt.Errorf("reward requires positive useful damage and negative damage/death/time costs")
	}
	return nil
}

type Reward struct {
	Navigation   *NavigationRewardEvidence `json:"navigation_reference,omitempty"`
	AimReference *AimReferenceEvidence     `json:"aim_reference,omitempty"`
	TargetCycle  *TargetCycleEvidence      `json:"target_cycle,omitempty"`
	Version      string                    `json:"version"`
	Worker       string                    `json:"worker"`
	Episode      string                    `json:"episode"`
	Step         int                       `json:"step"`
	Available    bool                      `json:"available"`
	Reason       string                    `json:"reason,omitempty"`
	Score        *float64                  `json:"score"`
	Components   map[string]float64        `json:"components"`
}

func (c RewardConfig) Evaluate(s *Step, o ServerOutcome) Reward {
	if c.Version == TargetSequenceRewardVersion {
		return Reward{Version: c.Version, Worker: s.Worker, Episode: s.Episode, Step: s.Index, Reason: "sequence_context_required"}
	}
	return c.evaluate(s, o)
}

func (c RewardConfig) evaluate(s *Step, o ServerOutcome) Reward {
	r := Reward{Version: c.Version, Worker: s.Worker, Episode: s.Episode, Step: s.Index}
	deny := func(reason string) Reward { r.Reason = reason; return r }
	if err := c.Validate(); err != nil {
		return deny("invalid_reward_config")
	}
	if o.Worker != s.Worker || o.Episode != s.Episode || o.Step != s.Index {
		return deny("outcome_identity_mismatch")
	}
	if s.Observation.Identity.Life != 1 {
		return deny("after_first_life")
	}
	// A control handoff ends the learned segment after its final command has
	// already executed. Keep its verified effect window in v2, including kills.
	completeHandoff := c.HasKillReward() && s.Truncated && s.Reason == "control_handoff"
	if s.Truncated && !completeHandoff || s.Next == nil || !sameWorld(s) || s.Next.Identity.Frame != s.Observation.Identity.Frame+1 || s.Observation.Health <= 0 || s.Observation.AgeMS < 0 || s.Observation.AgeMS > 300 || s.Next.AgeMS < 0 || s.Next.AgeMS > 300 {
		return deny("incomplete_transition")
	}
	if s.Action.Identity != s.Observation.Identity || s.AppliedAction.Identity != s.Observation.Identity || s.Action.Version != policy.ActionVersion || s.AppliedAction.Version != policy.ActionVersion {
		return deny("action_identity_mismatch")
	}
	if s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Native == nil || s.Native.Actor != s.Observation.Identity.Actor || s.Native.Spawncount != s.Observation.Identity.Spawncount || s.Native.Sequence != s.ClientSequence || s.Native.BeginFrame != s.Observation.Identity.Frame || s.Native.EndFrame != s.Next.Identity.Frame {
		return deny("unproven_synchronous_execution")
	}
	if !o.Available || o.Version != "server_step_effects_v1" {
		return deny("unavailable_native_effects")
	}
	if c.HasProjectileMissCost() {
		if o.ProjectileMisses == nil || o.ProjectileMisses.Version != "native_projectile_miss_v1" {
			return deny("unavailable_projectile_miss_evidence")
		}
		seen := map[uint32]bool{}
		for _, m := range o.ProjectileMisses.Misses {
			if m.Shot == 0 || m.Entity <= 0 || seen[m.Shot] || (m.Outcome != "geometry" && m.Outcome != "sky") || !sameNativeWindow(&m.End, s.Native) || m.Launch.Spawncount != m.End.Spawncount || m.Launch.Actor != m.End.Actor || m.Launch.BeginFrame > m.End.BeginFrame || m.Launch.EndFrame != m.Launch.BeginFrame+1 || m.Launch.Sequence == 0 || m.Launch.Sequence > m.End.Sequence {
				return deny("invalid_projectile_miss_evidence")
			}
			seen[m.Shot] = true
		}
	}
	if c.Version == NativeWasteRewardVersion {
		if o.HitscanMisses == nil || o.HitscanMisses.Version != "native_hitscan_miss_v1" {
			return deny("unavailable_hitscan_miss_evidence")
		}
		seen := map[uint32]bool{}
		for _, m := range o.HitscanMisses.Misses {
			if m.Shot == 0 || seen[m.Shot] || !sameNativeWindow(&m.Window, s.Native) || s.Native.Actor != s.Observation.Identity.Actor ||
				(m.Outcome != "geometry" && m.Outcome != "sky" && m.Outcome != "no_contact" && m.Outcome != "corpse") ||
				s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Execution.RecoveryCommands != 0 {
				return deny("invalid_hitscan_miss_evidence")
			}
			seen[m.Shot] = true
		}
	}
	if o.MonsterHealthDamage < 0 || o.ReceivedHealthDamage < 0 || o.SelfHealthDamage < 0 || o.TeammateHealthDamage < 0 || o.Deaths < 0 || o.Deaths > 1 || o.SelfHealthDamage > o.ReceivedHealthDamage {
		return deny("invalid_native_effects")
	}
	goalTerminal := s.Terminal && s.Reason == "combat_goal_complete" && s.Next.Health > 0
	jointTerminal := s.Terminal && s.Reason == "coop_participant_death"
	peerLost := false
	if jointTerminal || s.JointTerminal != nil || len(o.JointDeathEvents) > 0 {
		if c.Version != CoopRewardVersion || !jointTerminal {
			return deny("joint_terminal_requires_coop_reward")
		}
		var err error
		peerLost, err = verifyJointDeathReward(s, o.JointDeathEvents)
		if err != nil {
			return deny("invalid_joint_death_evidence")
		}
	}
	deathTerminal := s.Terminal && !goalTerminal && (!jointTerminal || s.Next.Health <= 0)
	if s.Reason == "combat_goal_complete" && !goalTerminal || deathTerminal != (s.Next.Health <= 0) || (deathTerminal && o.Deaths != 1) || (!deathTerminal && o.Deaths != 0) {
		return deny("death_evidence_mismatch")
	}
	if c.HasKillReward() {
		// Count lethal native effects caused by this actor, including delayed
		// projectiles. A disappearing entity or damage to a corpse is no kill.
		killed := map[int]bool{}
		for _, e := range o.Events {
			if e.Attacker == s.Observation.Identity.Actor && e.Target != e.Attacker && strings.HasPrefix(e.TargetClass, "monster_") && e.HealthBefore > 0 && e.HealthAfter <= 0 {
				if e.Target <= 0 || e.Map != s.Observation.Identity.Map || e.Spawncount != s.Observation.Identity.Spawncount || e.Take <= 0 || e.HealthBefore-e.Take != e.HealthAfter || killed[e.Target] {
					return deny("invalid_monster_kill_evidence")
				}
				killed[e.Target] = true
			}
		}
		if o.MonsterKills < 0 || len(killed) != o.MonsterKills {
			return deny("monster_kill_evidence_mismatch")
		}
	}
	r.Components = map[string]float64{
		"monster_damage":    float64(o.MonsterHealthDamage) * c.MonsterDamage,
		"received_damage":   float64(o.ReceivedHealthDamage) * c.ReceivedDamage,
		"self_damage_extra": float64(o.SelfHealthDamage) * c.SelfDamage,
		"friendly_damage":   float64(o.TeammateHealthDamage) * c.FriendlyDamage,
		"death":             float64(o.Deaths) * c.Death, "tick": c.Tick,
	}
	if c.HasKillReward() {
		r.Components["monster_kill"] = float64(o.MonsterKills) * c.MonsterKill
	}
	score := 0.0
	// Fixed order ensures byte-stable floating-point totals across exports.
	for _, name := range []string{"monster_damage", "received_damage", "self_damage_extra", "friendly_damage", "death", "tick"} {
		score += r.Components[name]
	}
	if c.HasKillReward() {
		score += r.Components["monster_kill"]
	}
	if c.Version == CoopRewardVersion {
		r.Components["peer_death"] = 0
		if peerLost && !deathTerminal {
			r.Components["peer_death"] = c.PeerDeath
		}
		score += r.Components["peer_death"]
	}
	if c.HasPotentialReward() {
		if c.Version == SelectedAimRewardVersion || c.Version == NativeWasteRewardVersion || c.HasNavigationReward() {
			shaping, evidence, err := c.selectedAimShaping(s, s.Terminal || completeHandoff)
			if err != nil {
				r.Components = nil
				return deny("invalid_aim_observation")
			}
			r.AimReference = evidence
			r.Components["aim_potential"] = shaping
			score += shaping
		} else {
			before, err := aimPotentialWithKick(s.Observation, c.AimPotential, c.AimKickAngles)
			if err != nil {
				r.Components = nil
				return deny("invalid_aim_observation")
			}
			after := 0.0
			if !s.Terminal && !completeHandoff {
				after, err = aimPotentialWithKick(*s.Next, c.AimPotential, c.AimKickAngles)
				if err != nil {
					r.Components = nil
					return deny("invalid_aim_observation")
				}
			}
			r.Components["aim_potential"] = c.AimGamma*after - before
			score += r.Components["aim_potential"]
		}
	}
	if c.Version == ManeuverRewardVersion || c.Version == MissRewardVersion || c.HasActionQualityCosts() {
		before, err := spacingPotential(s.Observation, c.SpacingPotential, c.ParasiteRange)
		if err != nil {
			r.Components = nil
			return deny("invalid_spacing_observation")
		}
		after := 0.0
		if !s.Terminal && !completeHandoff {
			after, err = spacingPotential(*s.Next, c.SpacingPotential, c.ParasiteRange)
		}
		if err != nil {
			r.Components = nil
			return deny("invalid_spacing_observation")
		}
		r.Components["spacing_potential"] = c.AimGamma*after - before
		score += r.Components["spacing_potential"]
	}
	if c.HasProjectileMissCost() {
		r.Components["blaster_miss"] = float64(len(o.ProjectileMisses.Misses)) * c.BlasterMiss
		score += r.Components["blaster_miss"]
	}
	if c.Version == NativeWasteRewardVersion {
		r.Components["machinegun_miss"] = float64(len(o.HitscanMisses.Misses)) * c.MachinegunMiss
		score += r.Components["machinegun_miss"]
	}
	if c.HasActionQualityCosts() {
		components, err := c.actionQualityCosts(s)
		if err != nil {
			r.Components = nil
			return deny("invalid_action_quality_observation")
		}
		for _, name := range []string{"off_target_attack", "turn_away", "stalled_movement"} {
			cost := components[name]
			r.Components[name] = cost
			score += cost
		}
	}
	if c.HasNavigationReward() {
		shaping, evidence, err := c.navigationShaping(s, s.Terminal || completeHandoff)
		if err != nil {
			r.Components = nil
			return deny("invalid_navigation_observation")
		}
		r.Navigation = evidence
		r.Components["navigation_potential"] = shaping
		score += shaping
	}
	if math.IsNaN(score) || math.IsInf(score, 0) {
		r.Components = nil
		return deny("reward_overflow")
	}
	r.Score = &score
	r.Available = true
	return r
}

// Nearest visible Parasite horizontal origin range only. The preferred range
// is an experimental maneuver prior, not a live command or safety guarantee.
func spacingPotential(o policy.Observation, scale, preferred float64) (float64, error) {
	nearest := math.Inf(1)
	for _, e := range o.Enemies {
		if e.Class != "monster_parasite" || e.ClearShot == nil || !*e.ClearShot {
			continue
		}
		d := math.Hypot(e.Relative[0], e.Relative[1])
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return 0, fmt.Errorf("nonfinite spacing observation")
		}
		nearest = math.Min(nearest, d)
	}
	return -scale * math.Max(0, 1-nearest/preferred), nil
}

// Potential is [-scale,0]; nearest observed bbox target, no server truth.
// Each complete segment sums to -Phi(start)+gamma^T Phi(end). Death/handoff
// has Phi(end)=0; arbitrary truncations retain normal critic bootstrap.
func aimPotential(o policy.Observation, scale float64) (float64, error) {
	return aimPotentialWithKick(o, scale, false)
}

// UDP camera punch includes weapon recoil and damage/fall shake. It is an
// observed aim proxy, not hidden weapon state or exact bullet direction.
func aimPotentialWithKick(o policy.Observation, scale float64, useKick bool) (float64, error) {
	best := math.Inf(1)
	result := 0.0
	yaw := float64(o.ViewAngles[1]) * 2 * math.Pi / 65536
	pitch := float64(o.ViewAngles[0]) * 2 * math.Pi / 65536
	if useKick {
		if o.KickAngles == nil {
			return 0, fmt.Errorf("observed kick angles unavailable")
		}
		for _, angle := range *o.KickAngles {
			if math.IsNaN(angle) || math.IsInf(angle, 0) {
				return 0, fmt.Errorf("nonfinite observed kick angle")
			}
		}
		yaw += (*o.KickAngles)[1] * math.Pi / 180
		pitch += (*o.KickAngles)[0] * math.Pi / 180
	}
	for _, e := range o.Enemies {
		if math.IsNaN(e.Distance) || math.IsInf(e.Distance, 0) || e.Distance < 0 {
			return 0, fmt.Errorf("invalid aim distance")
		}
		r, known, err := policy.ObservedAimDirection(o, e)
		if err != nil {
			return 0, err
		}
		if !known || e.Distance >= best {
			continue
		}
		best = e.Distance
		dot := math.Cos(pitch)*math.Cos(yaw)*r[0] + math.Cos(pitch)*math.Sin(yaw)*r[1] - math.Sin(pitch)*r[2]
		result = -scale * (1 - math.Max(-1, math.Min(1, dot))) / 2
	}
	return result, nil
}
