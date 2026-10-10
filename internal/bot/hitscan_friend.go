package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// hitscanTeammateRisk bounds stock Machinegun/Shotgun/Super Shotgun paths, including a water
// transition. It does not grant safety from unseen players or arbitrary future
// movement. Geometry only: it never changes aim, weapon or adds an attack.
func hitscanTeammateRisk(s quake.Snapshot, cmd quake.UserCmd) bool {
	if !directHitscanWeapon(s.Weapon) {
		return false
	}
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	horizontal, vertical, recoil := 500.0, 500.0, 0.0 // stock player's Shotgun_Fire, not monster defaults
	if machinegunWeapon(s.Weapon) {
		horizontal, vertical = 300, 500
		// Coop machinegun_shots is hidden; all kicks0..-13.5 fit around -6.75.
		// Native new yaw kick is random +/-0.7, not the previous rendered kick.
		pitch -= 6.75 * math.Pi / 180
		recoil = (6.75 + 0.7) * math.Pi / 180
	}
	if superShotgunWeapon(s.Weapon) {
		horizontal, vertical = 1000, 500
		// Both native pellet groups: v_angle yaw -5/+5; kick does not
		// change their firing direction. Include doubled underwater spread.
		recoil = 5 * math.Pi / 180
	}
	spread := math.Hypot(horizontal, vertical)
	angle := recoil + math.Atan(spread/8192) + math.Atan(2*spread/8192)
	length := math.Hypot(8192, spread) + math.Hypot(8192, 2*spread)
	axis := quake.Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
	from := s.EyePoint()
	// Body sphere contains [-16,16]XY/[-24,32]Z. Muzzle offset is8 right/8 down.
	// Bound both players' displacement until this usercmd is applied. Actual
	// partner motion, falling speed and network delay beyond this horizon remain
	// outside this guard; the companion runs synchronously from fresh snapshots.
	delay := math.Max(.1, float64(cmd.Msec)/1000)
	ownSpeed := math.Max(600, math.Sqrt(s.SelfVelocity[0]*s.SelfVelocity[0]+s.SelfVelocity[1]*s.SelfVelocity[1]+s.SelfVelocity[2]*s.SelfVelocity[2]))
	radius := math.Sqrt(16*16+16*16+32*32) + math.Hypot(8, 8) + 4 + (ownSpeed+600)*delay
	near := func(point quake.Vec3, age int) bool {
		for _, v := range point {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return true
			}
		}
		return finiteConeNearPoint(from, axis, length, math.Tan(angle), point, radius+float64(age)*60)
	}
	if s.Teammate != nil {
		return near(*s.Teammate, 0)
	}
	if s.LastTeammate != nil && s.TeammateAgeFrames != nil && *s.TeammateAgeFrames >= 0 && *s.TeammateAgeFrames <= 10 {
		return near(*s.LastTeammate, *s.TeammateAgeFrames)
	}
	return false
}

// Exact distance to the filled finite circular cone in axial/radial space.
func finiteConeNearPoint(from, axis quake.Vec3, length, slope float64, point quake.Vec3, radius float64) bool {
	d := quake.Vec3{point[0] - from[0], point[1] - from[1], point[2] - from[2]}
	x := d[0]*axis[0] + d[1]*axis[1] + d[2]*axis[2]
	r := math.Sqrt(math.Max(0, d[0]*d[0]+d[1]*d[1]+d[2]*d[2]-x*x))
	if x >= 0 && x <= length && r <= slope*x {
		return true
	}
	// Lateral cone surface, clipped at apex and end plane.
	t := math.Max(0, math.Min(length, (x+slope*r)/(1+slope*slope)))
	distance2 := (x-t)*(x-t) + (r-slope*t)*(r-slope*t)
	// The filled end disk can be closer than its circumference.
	radial := math.Max(0, r-slope*length)
	distance2 = math.Min(distance2, (x-length)*(x-length)+radial*radial)
	return distance2 <= radius*radius
}
