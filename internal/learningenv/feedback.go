package learningenv

import "q2coopbot/internal/policy"

const FeedbackVersion = "combat_feedback_v1"

// Feedback travels separately from the action observation. Native effects are
// training labels, not hidden state available to the Go combat controller.
type FeedbackEvent struct {
	Version string              `json:"version"`
	Kind    string              `json:"kind"`
	Initial *policy.Observation `json:"initial_observation,omitempty"`
	Reset   *ResetProof         `json:"reset_proof,omitempty"`
	Step    *Step               `json:"step,omitempty"`
	Reward  *Reward             `json:"reward,omitempty"`
	Effects *ServerOutcome      `json:"effects,omitempty"`
}
