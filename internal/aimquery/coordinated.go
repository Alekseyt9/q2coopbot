package aimquery

import (
	"math"

	"q2coopbot/internal/policy"
)

const CoordinatedVersion = "observed_bbox_aim_world_input_query_v1"

// QueryCoordinated changes the nominal aim while rotating the executed planar
// input into the new view basis. This is an offline annotation, not a tactical
// movement demonstration. Unsupported movement states receive no label.
func QueryCoordinated(o policy.Observation, applied policy.Action) (*Label, error) {
	if _, err := policy.Command(o, applied, [3]int16{}); err != nil {
		return nil, err
	}
	if !o.OnGround || o.Ducked || o.ViewAngles[2] != 0 || applied.Vertical != "release" {
		return nil, nil
	}
	q, err := Query(o)
	if err != nil || q == nil {
		return q, err
	}
	oldCmd, _ := policy.Command(o, applied, [3]int16{})
	newCmd, err := policy.Command(o, q.Action, [3]int16{})
	if err != nil {
		return nil, err
	}
	// PM_AirMove uses the horizontal components of AngleVectors, with
	// pitch divided by three, and right=(sin(yaw),-cos(yaw)) at zero roll.
	angle := func(v int16) float64 { return float64(v) * 2 * math.Pi / 65536 }
	d := angle(oldCmd.Yaw) - angle(newCmd.Yaw)
	f := float64(oldCmd.Forward) / 400 * math.Cos(angle(oldCmd.Pitch)/3)
	s := float64(oldCmd.Side) / 400
	q.Action.Forward = (f*math.Cos(d) + s*math.Sin(d)) / math.Cos(angle(newCmd.Pitch)/3)
	q.Action.Side = -f*math.Sin(d) + s*math.Cos(d)
	// Keep direction when the transformed vector exceeds the action square.
	// Amplitude can change; record that rather than claiming exact invariance.
	scale := math.Max(1, math.Max(math.Abs(q.Action.Forward), math.Abs(q.Action.Side)))
	q.Action.Forward /= scale
	q.Action.Side /= scale
	q.Version = CoordinatedVersion
	q.InputScale = 1 / scale
	q.Scope = "Unexecuted nominal aim plus planar input re-expression in final view frame. Preserves executed input direction before command rounding; amplitude scales if bounds require it. Grounded, standing, zero roll, vertical release only. No optimal movement, lead, recoil, hit or trajectory claim."
	if _, err := policy.Command(o, q.Action, [3]int16{}); err != nil {
		return nil, err
	}
	return q, nil
}
