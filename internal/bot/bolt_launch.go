package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Stock Blaster muzzle is 24 forward, 8 right, viewheight-8. PM_CheckDuck
// precedes weapon firing, so command posture matters, not just snapshot posture.
func boltLaunchPaths(s quake.Snapshot, cmd quake.UserCmd) (quake.Vec3, []quake.Vec3, float64) {
	yaw := float64(int16(uint16(cmd.Yaw)+uint16(s.DeltaAngles[1]))) * 2 * math.Pi / 65536
	pitch := float64(int16(uint16(cmd.Pitch)+uint16(s.DeltaAngles[0]))) * 2 * math.Pi / 65536
	f := quake.Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
	heights := []float64{22}
	if cmd.Up < 0 {
		heights = []float64{-2}
	} else if s.Ducked {
		// Standing may fail under a ceiling; cover both release outcomes.
		heights = []float64{-2, 22}
	}
	starts := make([]quake.Vec3, 0, len(heights))
	for _, height := range heights {
		starts = append(starts, quake.Vec3{s.Self[0] + 24*f[0] + 8*math.Sin(yaw), s.Self[1] + 24*f[1] - 8*math.Cos(yaw), s.Self[2] + height - 8 + 24*f[2]})
	}
	speed := math.Max(320, math.Sqrt(s.SelfVelocity[0]*s.SelfVelocity[0]+s.SelfVelocity[1]*s.SelfVelocity[1]+s.SelfVelocity[2]*s.SelfVelocity[2]))
	return f, starts, 4 + speed*math.Max(.1, float64(cmd.Msec)/1000)
}

func (p *Planner) guardBoltTeammate(s quake.Snapshot, cmd quake.UserCmd) quake.UserCmd {
	if s.Weapon != "Blaster" || s.Health <= 0 || cmd.Buttons&1 == 0 {
		return cmd
	}
	axis, starts, motion := boltLaunchPaths(s, cmd)
	point, age := s.Teammate, 0
	if point == nil && s.LastTeammate != nil && s.TeammateAgeFrames != nil && *s.TeammateAgeFrames >= 0 && *s.TeammateAgeFrames <= 10 {
		point, age = s.LastTeammate, *s.TeammateAgeFrames
	}
	if point == nil {
		return cmd
	}
	// Covers current body and command application displacement. Existing
	// velocity-based flight guard remains; unseen players/future turns aren't proved.
	radius := math.Sqrt(16*16+16*16+32*32) + motion + 320*math.Max(.1, float64(cmd.Msec)/1000) + float64(age)*32
	for _, start := range starts {
		end := quake.Vec3{start[0] + 1000*axis[0], start[1] + 1000*axis[1], start[2] + 1000*axis[2]}
		if finiteConeNearPoint(start, axis, 1000, 0, *point, radius) || p.teammateEntersProjectile(s, start, end) {
			cmd.Buttons &^= 1
			p.World.Command.LimitReason = "bolt_partner_launch_guard"
			return cmd
		}
	}
	return cmd
}
