package bot

import (
	"fmt"
	"math"

	"q2coopbot/internal/quake"
)

// A shared server-frame origin removes per-client join timing from paired trials.
// Zero preserves the older clock relative to the initial teleport command.
func (c *Client) testScenarioAge(frame int) int {
	if c.testScenarioFrameOrigin > 0 {
		return frame - c.testScenarioFrameOrigin
	}
	return frame - c.testTeleportSentFrame
}

// Test walking uses normal usercmd physics; only the initial placement teleports.
func validateTestWalk(cfg Config) (quake.Vec3, error) {
	if cfg.TestWalkTarget == "" && cfg.TestWalkAfterFrames == 0 && cfg.TestWalkFrames == 0 {
		return quake.Vec3{}, nil
	}
	if !cfg.FramePaced || !cfg.Idle || cfg.TestTeleport == "" || cfg.TestWalkAfterFrames < 25 ||
		cfg.TestWalkFrames < 1 || cfg.TestWalkFrames > 1000 || cfg.TestTeleportAfter != "" || cfg.TestLineCross {
		return quake.Vec3{}, fmt.Errorf("test.walk_target requires frame pacing, idle, initial teleport, delay >=25, 1..1000 frames and no other movement scenario")
	}
	return parseTestTeleport(cfg.TestWalkTarget)
}

func testWalkCommand(s quake.Snapshot, target quake.Vec3, geometry *quake.MapInfo, nav *quake.Navigator) quake.UserCmd {
	cmd := quake.UserCmd{}
	if s.Health <= 0 || !s.OnGround || geometry == nil || !geometry.MovementComplete() {
		return cmd
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	distance := math.Hypot(dx, dy)
	if distance <= 12 || geometry.GroundMoveHazardStep(nav, s.Self, dx, dy, math.Min(30, distance)) != "" {
		return cmd
	}
	cmd.Pitch = -s.DeltaAngles[0]
	return worldMove(cmd, s, dx, dy, math.Min(300, distance*10), false)
}
