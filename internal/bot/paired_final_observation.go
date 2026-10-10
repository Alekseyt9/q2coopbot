package bot

import (
	"encoding/json"
	"fmt"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"time"
)

// Final observation is not a sent/applied command or an actor training sample.
func (c *Client) writePairedFinalObservation(now time.Time) error {
	if c.traceFile == nil || !c.combatCapture {
		return fmt.Errorf("paired stop requires combat trace capture")
	}
	o := c.combatObservation(now)
	if o.Identity.Frame != c.testPairStopFrame || o.Identity.Life != 1 {
		return fmt.Errorf("invalid paired final observation identity")
	}
	a := policy.Action{Version: policy.ActionVersion, Identity: o.Identity, Vertical: "release"}
	capture := policy.Capture{Observation: o, Applied: a, Proposed: a, Provider: "terminal_observer"}
	row := struct {
		Terminal         bool           `json:"terminal_observation_only"`
		Capture          policy.Capture `json:"combat_policy"`
		Connection       int            `json:"connection"`
		Map              string         `json:"map"`
		Spawncount       int            `json:"spawncount"`
		Frame            int            `json:"frame"`
		ObservationFrame int            `json:"observation_frame"`
		SelfEntity       int            `json:"self_entity"`
		Self             quake.Vec3     `json:"self"`
		Health           int16          `json:"health"`
	}{true, capture, o.Identity.Connection, o.Identity.Map, o.Identity.Spawncount, o.Identity.Frame, o.Identity.Frame, o.Identity.Actor, o.Position, o.Health}
	return json.NewEncoder(c.traceFile).Encode(row)
}
