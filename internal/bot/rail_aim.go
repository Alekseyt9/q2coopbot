package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"strings"
)

type railAim struct {
	class                  string
	solid                  uint16
	mapName                string
	entity, frame, settled int
	point                  quake.Vec3
}

func isRailgun(weapon string) bool {
	return strings.Contains(strings.ToLower(weapon), "/v_rail/") || strings.EqualFold(weapon, "Railgun")
}

// Smooth only acquisition of the rail aim, then require two precise consecutive
// commands. Movement continues while aiming; no fixed delay is added per shot.
func (p *Planner) railCommand(s quake.Snapshot, e quake.Object, prev quake.UserCmd, cmd quake.UserCmd) (quake.UserCmd, bool) {
	point := e.AimPoint()
	if p.railAim.mapName != s.Map || p.railAim.entity != e.ID || p.railAim.class != e.Class || p.railAim.solid != e.Solid || p.railAim.frame+1 != s.Frame || quake.Distance(point, p.railAim.point) > 64 {
		p.railAim = railAim{mapName: s.Map, entity: e.ID, class: e.Class, solid: e.Solid}
	}
	turn := func(from, to int16) int16 {
		delta := int(int16(uint16(to) - uint16(from)))
		delta = max(-8192, min(8192, delta))
		return int16(uint16(from) + uint16(int16(delta)))
	}
	cmd.Yaw = turn(prev.Yaw, cmd.Yaw)
	cmd.Pitch = turn(prev.Pitch, cmd.Pitch)
	yaw := quake.YawTo(s.EyePoint(), point, s.DeltaAngles[1])
	pitch := quake.PitchTo(s.EyePoint(), point, s.DeltaAngles[0])
	error := math.Max(math.Abs(float64(int16(uint16(yaw)-uint16(cmd.Yaw)))), math.Abs(float64(int16(uint16(pitch)-uint16(cmd.Pitch))))) * 360 / 65536
	if error <= 0.5 {
		p.railAim.settled++
	} else {
		p.railAim.settled = 0
	}
	p.railAim.frame = s.Frame
	p.railAim.point = point
	p.World.Command.AimErrorDegrees = error
	return cmd, p.railAim.settled >= 2
}

func railEnd(from, to quake.Vec3) quake.Vec3 {
	d := quake.Distance(from, to)
	if d < 1 {
		return to
	}
	for i := range to {
		to[i] = from[i] + (to[i]-from[i])*8192/d
	}
	return to
}
