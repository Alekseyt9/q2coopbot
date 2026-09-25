package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// CommandDecision records which controller supplied each part of a usercmd.
type CommandDecision struct {
	MoveSource      string `json:"move_source"`
	AimSource       string `json:"aim_source"`
	Skill           string `json:"skill,omitempty"`
	LimitReason     string `json:"limit_reason,omitempty"`
	MoveLimitReason string `json:"move_limit_reason,omitempty"`
}

// worldMove converts an XY direction into forward/side commands in the final
// view frame. Quake II's positive sidemove points to the player's right, whose
// XY basis is (sin(yaw), -cos(yaw)). Forward's XY length shrinks with pitch.
func worldMove(cmd quake.UserCmd, s quake.Snapshot, dx, dy, speed float64, keepAim bool) quake.UserCmd {
	distance := math.Hypot(dx, dy)
	if distance < 0.001 || speed <= 0 {
		return cmd
	}
	if !keepAim {
		cmd.Yaw = quake.YawTo(s.Self, quake.Vec3{s.Self[0] + dx, s.Self[1] + dy, s.Self[2]}, s.DeltaAngles[1])
	}
	x, y := dx/distance, dy/distance
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	pitch = math.Max(-89*math.Pi/180, math.Min(89*math.Pi/180, pitch))
	forward := speed * (x*math.Cos(yaw) + y*math.Sin(yaw)) / math.Cos(pitch)
	side := speed * (x*math.Sin(yaw) - y*math.Cos(yaw))
	maxInput := math.Max(math.Abs(forward), math.Abs(side))
	if maxInput > 400 {
		forward *= 400 / maxInput
		side *= 400 / maxInput
	}
	cmd.Forward = int16(math.Round(forward))
	cmd.Side = int16(math.Round(side))
	return cmd
}

// teammateBlocksShot conservatively tests the visible player's body against
// the intended shot segment. Static BSP ClearShot cannot see players.
func teammateBlocksShot(from, target, teammate quake.Vec3) bool {
	dx, dy := target[0]-from[0], target[1]-from[1]
	length2 := dx*dx + dy*dy
	if length2 < 1 {
		return false
	}
	t := ((teammate[0]-from[0])*dx + (teammate[1]-from[1])*dy) / length2
	if t <= 0 || t >= 1 {
		return false
	}
	x, y := from[0]+t*dx, from[1]+t*dy
	if math.Hypot(x-teammate[0], y-teammate[1]) > 24 {
		return false
	}
	z := from[2] + t*(target[2]-from[2])
	return z >= teammate[2]-28 && z <= teammate[2]+36
}
