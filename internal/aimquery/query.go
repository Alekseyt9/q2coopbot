// Package aimquery constructs an unexecuted nominal aim annotation from
// observed client state. It does not modify the applied action or its reward.
package aimquery

import (
	"fmt"
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"strings"
)

const Version = "observed_bbox_aim_query_v1"

type Label struct {
	InputScale float64       `json:"planar_input_scale,omitempty"`
	Version    string        `json:"version"`
	Target     int           `json:"target"`
	Action     policy.Action `json:"desired_action"`
	Scope      string        `json:"scope"`
}

// Query requires a confirmed clear observed target and known protocol bbox.
// Nominal aim ignores lead and recoil; it is not a claim of expert optimality.
func Query(o policy.Observation) (*Label, error) {
	if o.Version != policy.ObservationVersion || o.Identity.Frame <= 0 || o.Identity.Life != 1 || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		return nil, fmt.Errorf("invalid query observation")
	}
	weapon := strings.ToLower(o.Weapon)
	if weapon != "blaster" && weapon != "machinegun" && !strings.Contains(weapon, "/v_blast/") && !strings.Contains(weapon, "/v_machn/") {
		return nil, nil
	}
	best := 650.0
	target := 0
	var direction quake.Vec3
	for _, e := range o.Enemies {
		if !strings.HasPrefix(e.Class, "monster_") || e.ID <= 0 {
			continue
		}
		d := math.Hypot(math.Hypot(e.Relative[0], e.Relative[1]), e.Relative[2])
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return nil, fmt.Errorf("nonfinite enemy position")
		}
		r, known, err := policy.ObservedAimDirection(o, e)
		if err != nil {
			return nil, err
		}
		if !known || d >= best || math.Hypot(r[0], r[1]) < 1e-9 {
			continue
		}
		best, target, direction = d, e.ID, r
	}
	if target == 0 {
		return nil, nil
	}
	cmd := quake.UserCmd{Yaw: quake.YawTo(quake.Vec3{}, direction, 0), Pitch: quake.PitchTo(quake.Vec3{}, direction, 0), Msec: 100}
	a := policy.FromCommand(o, cmd, [3]int16{}, "")
	if _, err := policy.Command(o, a, [3]int16{}); err != nil {
		return nil, err
	}
	return &Label{Version: Version, Target: target, Action: a, Scope: "Counterfactual nominal aim only; not executed. Observed clear target/bbox/current eye. No firing, movement, lead, recoil, hit credit or target-optimality claim."}, nil
}
