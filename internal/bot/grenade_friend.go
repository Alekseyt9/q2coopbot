package bot

import "q2coopbot/internal/quake"

type GrenadeFriendMotion struct {
	Frame    int        `json:"frame"`
	Entity   int        `json:"entity"`
	Velocity quake.Vec3 `json:"velocity"`
}

// A constant-motion hazard, never proof of safety. Restrict extrapolation to
// 500ms, sweep relative motion, and inflate the observed player's body by8.
// Unknown histories/turns and later reachability retain the broad risk guard.
func grenadeFriendContact(start, velocity quake.Vec3, gravity float64, trace func(quake.Vec3, quake.Vec3) quake.PointTrace, friend, motion quake.Vec3) float64 {
	t, _ := grenadeMovingBodyContact(start, velocity, gravity, trace, grenadeBody{friend, quake.Vec3{-24, -24, -32}, quake.Vec3{24, 24, 40}}, motion)
	return t
}

// Relative sweep of a short constant-motion hypothesis. Inflated observed
// bounds can detect risk; neither the hit nor its absence certifies safety.
func grenadeMovingBodyContact(start, velocity quake.Vec3, gravity float64, trace func(quake.Vec3, quake.Vec3) quake.PointTrace, body grenadeBody, motion quake.Vec3) (float64, quake.Vec3) {
	previous := start
	tick := 0
	contact := 0.0
	position := quake.Vec3{}
	simulateGrenade(start, velocity, 0.6, gravity, trace, nil, func(point quake.Vec3, _ int) {
		tick++
		if contact > 0 || tick > 5 {
			return
		}
		from, to := previous, point
		for i := range from {
			from[i] -= motion[i] * float64(tick-1) * 0.1
			to[i] -= motion[i] * float64(tick) * 0.1
		}
		if fraction, ok := grenadeBodyHit(from, to, body); ok {
			contact = float64(tick) * 0.1
			for i := range position {
				position[i] = previous[i] + fraction*(point[i]-previous[i])
			}
		}
		previous = point
	})
	return contact, position
}
