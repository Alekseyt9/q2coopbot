package bot

import "q2coopbot/internal/quake"

// Explicit harness action only; the companion combat policy never calls this.
func scenarioShootTeammate(s quake.Snapshot, g *quake.MapInfo) quake.UserCmd {
	cmd := quake.UserCmd{}
	if s.Teammate == nil || s.Health <= 0 || !g.HasCollision() {
		return cmd
	}
	from, to := s.Self, *s.Teammate
	from[2] += 22
	if !g.ClearShot(from, to) || g.DoorShotBlocked(s.Movers, from, to) {
		return cmd
	}
	cmd.Pitch = quake.PitchTo(from, to, s.DeltaAngles[0])
	cmd.Yaw = quake.YawTo(from, to, s.DeltaAngles[1])
	cmd.Buttons = 1
	return cmd
}
