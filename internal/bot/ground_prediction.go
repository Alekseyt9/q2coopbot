package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// GroundPrediction assumes baseq2 dry, flat, nonslick ground, no currents,
// ladders, contacts or custom physics. It is diagnostic, not a collision proof.
type GroundPrediction struct {
	Displacement        quake.Vec3 `json:"displacement"`
	Velocity            quake.Vec3 `json:"velocity"`
	NeutralStopDistance float64    `json:"neutral_stop_distance"`
	Model               string     `json:"model"`
	CommandPath         string     `json:"command_path,omitempty"`
	NeutralPath         string     `json:"neutral_path,omitempty"`
	CommandThenStopPath string     `json:"command_then_stop_path,omitempty"`
}

// Diagnose the emitted command and the full neutral stop, including inertia
// after that command. Static sampling is not proof against dynamic contacts.
func diagnoseGroundStep(s quake.Snapshot, cmd quake.UserCmd, geometry *quake.MapInfo) *GroundPrediction {
	p := predictGroundStep(s, cmd)
	if p == nil {
		return nil
	}
	end := s.Self
	for axis := range end {
		end[axis] += p.Displacement[axis]
	}
	p.CommandPath = geometry.GroundPathStatus(s.Self, end, s.Ducked)
	stopEnd := func(origin, velocity quake.Vec3, distance float64) quake.Vec3 {
		speed := math.Hypot(velocity[0], velocity[1])
		if speed > 0 {
			origin[0] += velocity[0] / speed * distance
			origin[1] += velocity[1] / speed * distance
		}
		return origin
	}
	p.NeutralPath = geometry.GroundPathStatus(s.Self, stopEnd(s.Self, s.SelfVelocity, p.NeutralStopDistance), s.Ducked)
	next := s
	next.Self, next.SelfVelocity = end, p.Velocity
	neutral := cmd
	neutral.Forward, neutral.Side = 0, 0
	remaining := predictGroundStep(next, neutral)
	p.CommandThenStopPath = p.CommandPath
	if p.CommandPath == "static_sampled_clear" && remaining != nil {
		p.CommandThenStopPath = geometry.GroundPathStatus(end, stopEnd(end, p.Velocity, remaining.NeutralStopDistance), s.Ducked)
	}
	return p
}

func predictGroundStep(s quake.Snapshot, cmd quake.UserCmd) *GroundPrediction {
	if !s.OnGround || s.Health <= 0 || cmd.Up != 0 || cmd.Msec == 0 || cmd.Msec > 100 || math.Abs(s.SelfVelocity[2]) > 1 {
		return nil
	}
	dt := float64(cmd.Msec) / 1000
	v := s.SelfVelocity
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil
		}
	}
	friction := func(v quake.Vec3) quake.Vec3 {
		speed := math.Hypot(v[0], v[1])
		if speed < 1 {
			return quake.Vec3{}
		}
		scale := max(0, speed-max(speed, 100)*6*dt) / speed
		return quake.Vec3{v[0] * scale, v[1] * scale, 0}
	}
	stop := 0.0
	for neutral, n := v, 0; ; n++ {
		// Bound elapsed simulation time rather than command count: 100 short
		// commands can end before the player stops and understate the path.
		if n >= int(math.Ceil(10/dt)) {
			return nil
		}
		neutral = friction(neutral)
		speed := math.Hypot(neutral[0], neutral[1])
		if math.IsNaN(speed) || math.IsInf(speed, 0) {
			return nil
		}
		stop += speed * dt
		if speed < 1 {
			break
		}
	}
	v = friction(v)
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	pitch = max(-89*math.Pi/180, min(89*math.Pi/180, pitch)) / 3
	wish := quake.Vec3{math.Cos(pitch)*math.Cos(yaw)*float64(cmd.Forward) + math.Sin(yaw)*float64(cmd.Side), math.Cos(pitch)*math.Sin(yaw)*float64(cmd.Forward) - math.Cos(yaw)*float64(cmd.Side), 0}
	speed := math.Hypot(wish[0], wish[1])
	if speed > 0 {
		x, y := wish[0]/speed, wish[1]/speed
		limit := 300.0
		if s.Ducked {
			limit = 100
		}
		speed = min(speed, limit)
		add := speed - (v[0]*x + v[1]*y)
		if add > 0 {
			accel := min(add, 10*dt*speed)
			v[0] += x * accel
			v[1] += y * accel
		}
	}
	return &GroundPrediction{Displacement: quake.Vec3{v[0] * dt, v[1] * dt, 0}, Velocity: v, NeutralStopDistance: stop, Model: "baseq2_dry_flat_no_contacts"}
}
