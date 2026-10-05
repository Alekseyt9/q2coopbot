package learningenv

import (
	"fmt"
	"math"

	"q2coopbot/internal/policy"
)

const RewardVersion = "combat_reward_v1"

// RewardConfig is an explicit experimental objective, not a measured success
// metric. Server effects never enter the policy observation.
type RewardConfig struct {
	Version        string  `json:"version"`
	MonsterDamage  float64 `json:"monster_damage"`
	ReceivedDamage float64 `json:"received_damage"`
	SelfDamage     float64 `json:"self_damage_extra"`
	FriendlyDamage float64 `json:"friendly_damage"`
	Death          float64 `json:"death"`
	Tick           float64 `json:"tick"`
}

func (c RewardConfig) Validate() error {
	if c.Version != RewardVersion {
		return fmt.Errorf("unsupported reward version")
	}
	for _, v := range []float64{c.MonsterDamage, c.ReceivedDamage, c.SelfDamage, c.FriendlyDamage, c.Death, c.Tick} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("nonfinite reward coefficient")
		}
	}
	if c.MonsterDamage <= 0 || c.ReceivedDamage >= 0 || c.SelfDamage > 0 || c.FriendlyDamage >= 0 || c.Death >= 0 || c.Tick >= 0 {
		return fmt.Errorf("reward requires positive useful damage and negative damage/death/time costs")
	}
	return nil
}

type Reward struct {
	Version    string             `json:"version"`
	Worker     string             `json:"worker"`
	Episode    string             `json:"episode"`
	Step       int                `json:"step"`
	Available  bool               `json:"available"`
	Reason     string             `json:"reason,omitempty"`
	Score      *float64           `json:"score"`
	Components map[string]float64 `json:"components"`
}

func (c RewardConfig) Evaluate(s *Step, o ServerOutcome) Reward {
	r := Reward{Version: RewardVersion, Worker: s.Worker, Episode: s.Episode, Step: s.Index}
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
	if s.Truncated || s.Next == nil || !sameWorld(s) || s.Next.Identity.Frame != s.Observation.Identity.Frame+1 || s.Observation.Health <= 0 || s.Observation.AgeMS < 0 || s.Observation.AgeMS > 300 || s.Next.AgeMS < 0 || s.Next.AgeMS > 300 {
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
	if o.MonsterHealthDamage < 0 || o.ReceivedHealthDamage < 0 || o.SelfHealthDamage < 0 || o.TeammateHealthDamage < 0 || o.Deaths < 0 || o.Deaths > 1 || o.SelfHealthDamage > o.ReceivedHealthDamage {
		return deny("invalid_native_effects")
	}
	if s.Terminal != (s.Next.Health <= 0) || (s.Terminal && o.Deaths != 1) || (!s.Terminal && o.Deaths != 0) {
		return deny("death_evidence_mismatch")
	}
	r.Components = map[string]float64{
		"monster_damage":    float64(o.MonsterHealthDamage) * c.MonsterDamage,
		"received_damage":   float64(o.ReceivedHealthDamage) * c.ReceivedDamage,
		"self_damage_extra": float64(o.SelfHealthDamage) * c.SelfDamage,
		"friendly_damage":   float64(o.TeammateHealthDamage) * c.FriendlyDamage,
		"death":             float64(o.Deaths) * c.Death, "tick": c.Tick,
	}
	score := 0.0
	// Fixed order ensures byte-stable floating-point totals across exports.
	for _, name := range []string{"monster_damage", "received_damage", "self_damage_extra", "friendly_damage", "death", "tick"} {
		score += r.Components[name]
	}
	if math.IsNaN(score) || math.IsInf(score, 0) {
		r.Components = nil
		return deny("reward_overflow")
	}
	r.Score = &score
	r.Available = true
	return r
}
