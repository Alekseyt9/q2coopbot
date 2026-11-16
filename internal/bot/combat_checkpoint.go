package bot

import (
	"encoding/json"
	"fmt"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const combatCheckpointVersion = "hjepa_predecision_checkpoint_v1"

// combatCheckpoint describes client memory immediately BEFORE Root is observed
// by the policy. The native save independently supplies the world at Root.
// Previous is the preceding sent command, not a queued command for Root.
type combatCheckpoint struct {
	Version            string                             `json:"version"`
	Root               policy.Identity                    `json:"decision_root"`
	Memory             policy.ObservationMemoryCheckpoint `json:"observation_memory"`
	Sampling           policy.HJEPASamplingCheckpoint     `json:"sampling"`
	Previous           quake.UserCmd                      `json:"previous_command"`
	LastHealth         int16                              `json:"last_health"`
	WeaponRequest      *combatCheckpointWeaponRequest     `json:"weapon_request,omitempty"`
	Engagement         combatCheckpointEngagement         `json:"engagement"`
	PreviousTarget     *policy.TargetIntent               `json:"previous_target,omitempty"`
	Contract           string                             `json:"execution_contract"`
	ActionQualityGuard bool                               `json:"action_quality_guard"`
	CheckedWallSlide   bool                               `json:"checked_wall_slide"`
}

type combatCheckpointWeaponRequest struct {
	Identity policy.Identity `json:"identity"`
	Name     string          `json:"name"`
}

type combatCheckpointEngagement struct {
	Identity policy.Identity `json:"identity"`
	Known    map[int]string  `json:"known"`
	LastSeen int             `json:"last_seen"`
}

// Diagnostic actions, campaign navigation and held options require their own
// state contracts. Reject those configurations instead of silently omitting state.
func (b *combatControl) checkpointCompatible() bool {
	return b.mode == "learned" && !b.campaignEvaluation && !b.diagnosticNominalAim &&
		b.grenadeReleaseProbeFrame == 0 && b.rocketAimProbeFrame == 0 && b.rocketAimProbePoint == nil &&
		b.forcedDuckProbeFrame == 0 && b.stoppedCrouchProbeFrame == 0 && b.checkedGroundRequestProbeFrame == 0 &&
		!b.checkedGroundRequestProbeAllFrames && !b.stoppedCrouchProbeAllFrames
}

func (b *combatControl) captureCombatCheckpoint(root policy.Identity, previous quake.UserCmd) (combatCheckpoint, error) {
	p, ok := b.provider.(*policy.HJEPA)
	if !ok || !b.checkpointCompatible() || !policy.SameLife(b.last, root) || b.last.Frame+1 != root.Frame ||
		b.life != root.Life || b.lastHealth <= 0 || b.observation == nil || b.observation.Identity != b.last {
		return combatCheckpoint{}, fmt.Errorf("combat checkpoint requires a living H-JEPA pre-decision prefix")
	}
	memory, err := policy.CaptureObservationMemory(&b.history, &b.threatMemory, b.last)
	if err != nil {
		return combatCheckpoint{}, err
	}
	sampling, err := p.CaptureSamplingCheckpoint(root)
	if err != nil {
		return combatCheckpoint{}, err
	}
	s := combatCheckpoint{Version: combatCheckpointVersion, Root: root, Memory: memory, Sampling: sampling,
		Previous: previous, LastHealth: b.lastHealth, Contract: b.executionContract,
		ActionQualityGuard: b.actionQualityGuard, CheckedWallSlide: b.checkedWallSlide,
		Engagement:     combatCheckpointEngagement{Identity: b.engagement.identity, Known: b.engagement.known, LastSeen: b.engagement.lastSeen},
		PreviousTarget: b.previousTarget}
	if b.weaponRequests.name != "" {
		s.WeaponRequest = &combatCheckpointWeaponRequest{Identity: b.weaponRequests.identity, Name: b.weaponRequests.name}
	}
	// Copy pointer/map fields, so continued source execution cannot change the save.
	data, err := json.Marshal(s)
	if err != nil {
		return combatCheckpoint{}, err
	}
	var result combatCheckpoint
	if err = json.Unmarshal(data, &result); err != nil {
		return combatCheckpoint{}, err
	}
	if err = result.validate(); err != nil {
		return combatCheckpoint{}, err
	}
	return result, nil
}

func (s combatCheckpoint) validate() error {
	prior := s.Root
	prior.Frame--
	if s.Version != combatCheckpointVersion || s.Root.Frame < 2 || s.LastHealth <= 0 || s.Memory.Anchor != prior ||
		s.Sampling.Source.Map != s.Root.Map || s.Sampling.Source.Actor != s.Root.Actor || s.Sampling.Source.Life != s.Root.Life ||
		s.Engagement.Identity != prior || len(s.Engagement.Known) > 256 ||
		s.Engagement.LastSeen > prior.Frame || s.Engagement.LastSeen < prior.Frame-policy.ThreatMemoryFrames {
		return fmt.Errorf("combat checkpoint root/prefix/engagement mismatch")
	}
	if _, _, err := policy.RestoreObservationMemory(s.Memory, prior); err != nil {
		return err
	}
	if s.Memory.Frames[len(s.Memory.Frames)-1].Health != s.LastHealth {
		return fmt.Errorf("combat checkpoint health prefix mismatch")
	}
	for id, class := range s.Engagement.Known {
		if id <= 0 || class == "" || len(class) > 128 {
			return fmt.Errorf("combat checkpoint invalid observed combat identity")
		}
	}
	if r := s.WeaponRequest; r != nil {
		var h weaponRequestHistory
		h.record(r.Identity, r.Name)
		if h.name == "" || !policy.SameLife(r.Identity, prior) || r.Identity.Frame > prior.Frame || r.Identity.Frame < -1000000 {
			return fmt.Errorf("combat checkpoint invalid previous weapon request")
		}
	}
	if t := s.PreviousTarget; t != nil && (t.Identity != prior || t.Entity <= 0 || t.Track <= 0 || t.Track > s.Memory.NextTrack) {
		return fmt.Errorf("combat checkpoint invalid previous target")
	}
	return nil
}

// restoreCombatCheckpoint prepares a fresh controller for its first decision.
// Native root remains positive, while its last-observed prefix may end at zero.
// Current entities, inventory, HP and cached observations are not restored here.
func (b *combatControl) restoreCombatCheckpoint(s combatCheckpoint, native policy.Identity) (quake.UserCmd, error) {
	if err := s.validate(); err != nil {
		return quake.UserCmd{}, err
	}
	p, ok := b.provider.(*policy.HJEPA)
	if !ok || !b.checkpointCompatible() || b.last != (policy.Identity{}) || b.observation != nil || b.life != 0 ||
		b.executionContract != s.Contract || b.actionQualityGuard != s.ActionQualityGuard || b.checkedWallSlide != s.CheckedWallSlide ||
		native.Life != s.Root.Life {
		return quake.UserCmd{}, fmt.Errorf("combat checkpoint requires fresh controller and matching execution contract")
	}
	history, threats, err := policy.RestoreObservationMemoryBeforeDecision(s.Memory, native)
	if err != nil {
		return quake.UserCmd{}, err
	}
	prior := native
	prior.Frame--
	rebase := func(id policy.Identity) policy.Identity {
		r := native
		r.Frame -= s.Root.Frame - id.Frame
		return r
	}
	engagement := combatEngagement{identity: prior, known: make(map[int]string), lastSeen: native.Frame - (s.Root.Frame - s.Engagement.LastSeen)}
	for id, class := range s.Engagement.Known {
		engagement.known[id] = class
	}
	var request weaponRequestHistory
	if s.WeaponRequest != nil {
		request = weaponRequestHistory{identity: rebase(s.WeaponRequest.Identity), name: s.WeaponRequest.Name}
	}
	var target *policy.TargetIntent
	if s.PreviousTarget != nil {
		value := *s.PreviousTarget
		value.Identity = prior
		target = &value
	}
	// Mutate the provider only after all other validations have succeeded.
	if err := p.RestoreSamplingCheckpoint(s.Sampling, native); err != nil {
		return quake.UserCmd{}, err
	}
	b.history, b.threatMemory = history, threats
	b.last, b.lastHealth, b.life = prior, s.LastHealth, native.Life
	b.engagement, b.weaponRequests, b.previousTarget = engagement, request, target
	return s.Previous, nil
}
