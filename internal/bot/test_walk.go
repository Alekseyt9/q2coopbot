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

func (c *Client) testWalkAge(frame int) int {
	if c.testCombatBarrier {
		if !c.testCombatGo {
			return -1
		}
		return frame - c.testCombatGoFrame
	}
	return c.testScenarioAge(frame)
}

// Test walking uses normal usercmd physics; only the initial placement teleports.
func validateTestWalk(cfg Config) (quake.Vec3, error) {
	if cfg.TestRunInSpeed < 0 || cfg.TestRunInSpeed > 300 || cfg.TestRunInSpeed > 0 && !cfg.TestWalkRunIn {
		return quake.Vec3{}, fmt.Errorf("test.run_in_speed requires walk_run_in and a speed in 1..300")
	}
	if cfg.TestWalkRunIn {
		combat := cfg.TestCombatBarrier && cfg.TestWalkAfterFrames == 0
		movement := !cfg.TestCombatBarrier && cfg.TestSetupHoldFrames > 0 && cfg.TestWalkAfterFrames == cfg.TestSetupHoldFrames
		if !cfg.FramePaced || cfg.Idle || (!combat && !movement) || cfg.TestTeleport == "" || cfg.TestWalkTarget == "" || cfg.TestWalkFrames < 1 || cfg.TestWalkFrames > 5 || cfg.TestWalkRoute || cfg.TestTeleportAfter != "" || cfg.TestLineCross {
			return quake.Vec3{}, fmt.Errorf("test.walk_run_in requires active frame pacing, combat release or setup hold, initial teleport and 1..5 straight walk frames")
		}
		return parseTestTeleport(cfg.TestWalkTarget)
	}
	if cfg.TestWalkRoute && cfg.TestWalkTarget == "" {
		return quake.Vec3{}, fmt.Errorf("test.walk_route requires test.walk_target")
	}
	if cfg.TestWalkTarget == "" && cfg.TestWalkAfterFrames == 0 && cfg.TestWalkFrames == 0 {
		return quake.Vec3{}, nil
	}
	if !cfg.FramePaced || !cfg.Idle || cfg.TestTeleport == "" || cfg.TestWalkAfterFrames < 25 ||
		cfg.TestWalkFrames < 1 || cfg.TestWalkFrames > 1000 || cfg.TestTeleportAfter != "" || cfg.TestLineCross {
		return quake.Vec3{}, fmt.Errorf("test.walk_target requires frame pacing, idle, initial teleport, delay >=25, 1..1000 frames and no other movement scenario")
	}
	return parseTestTeleport(cfg.TestWalkTarget)
}

type testWalkPath struct {
	reason    string
	route     []quake.Waypoint
	next      int
	ready     bool
	mapName   string
	lastFrame int
}

// A bounded prelude accelerates through native usercmd physics, without
// substituting an invented velocity into the planner's observation.
func testRunInCommand(s quake.Snapshot, target quake.Vec3, speed float64, g *quake.MapInfo, nav *quake.Navigator) quake.UserCmd {
	if s.Health <= 0 || !s.OnGround || g == nil || !g.MovementComplete() {
		return quake.UserCmd{}
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	if g.GroundMoveHazardStep(nav, s.Self, dx, dy, speed/10) != "" {
		return quake.UserCmd{}
	}
	return worldMove(quake.UserCmd{Pitch: -s.DeltaAngles[0]}, s, dx, dy, speed, false)
}

func (p *testWalkPath) command(s quake.Snapshot, target quake.Vec3, geometry *quake.MapInfo, nav *quake.Navigator) quake.UserCmd {
	if p.mapName != s.Map || s.Frame < p.lastFrame {
		*p = testWalkPath{mapName: s.Map, lastFrame: s.Frame}
	}
	p.reason = ""
	if nav == nil {
		p.reason = "aas_unavailable"
		return quake.UserCmd{}
	}
	p.lastFrame = s.Frame
	if !p.ready {
		if nav.ExactAreaFor(target) < 0 {
			p.reason = "route_target_outside_aas"
			return quake.UserCmd{}
		}
		if nav.ExactAreaFor(s.Self) < 0 {
			p.reason = "route_start_outside_aas"
			return quake.UserCmd{}
		}
		var ok bool
		p.route, ok = nav.SearchRoute(s.Self, target)
		if !ok {
			p.reason = "route_unavailable"
			return quake.UserCmd{}
		}
		p.ready = true
	}
	for p.next < len(p.route) && quake.Horizontal(s.Self, p.route[p.next].Position) <= 24 {
		p.next++
	}
	if p.next < len(p.route) {
		cmd, reason := testWalkDiagnostic(s, p.route[p.next].Position, geometry, nav)
		p.reason = reason
		return cmd
	}
	cmd, reason := testWalkDiagnostic(s, target, geometry, nav)
	p.reason = reason
	return cmd
}

func testWalkCommand(s quake.Snapshot, target quake.Vec3, geometry *quake.MapInfo, nav *quake.Navigator) quake.UserCmd {
	cmd, _ := testWalkDiagnostic(s, target, geometry, nav)
	return cmd
}

func testWalkDiagnostic(s quake.Snapshot, target quake.Vec3, geometry *quake.MapInfo, nav *quake.Navigator) (quake.UserCmd, string) {
	cmd := quake.UserCmd{}
	if s.Health <= 0 {
		return cmd, "actor_dead"
	}
	if !s.OnGround {
		return cmd, "not_grounded"
	}
	if geometry == nil || !geometry.MovementComplete() {
		return cmd, "geometry_unavailable"
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	distance := math.Hypot(dx, dy)
	if distance <= 12 {
		return cmd, "within_horizontal_tolerance"
	}
	if reason := geometry.GroundMoveHazardStep(nav, s.Self, dx, dy, math.Min(30, distance)); reason != "" {
		// The actor may stand on a stopped platform rather than static BSP.
		// Use the same exact mover/support checks as the companion's bypass.
		clear := false
		if reason == "no_ground_support" {
			p := &Planner{World: World{Geometry: geometry}}
			next := s.Self
			next[0] += dx / distance * math.Min(30, distance)
			next[1] += dy / distance * math.Min(30, distance)
			for _, mover := range s.Movers {
				model, ok := geometry.Model(mover.Model)
				if !ok || quake.Distance(mover.Origin, model.Origin) > .125 {
					continue
				}
				if drop, ok := geometry.MoverFooting(mover, s.Self, .5); ok && drop <= .5 && p.elevatorBypassClear(s, mover, s.Self, next) {
					clear = true
					break
				}
			}
		}
		if !clear {
			return cmd, reason
		}
	}
	cmd.Pitch = -s.DeltaAngles[0]
	return worldMove(cmd, s, dx, dy, math.Min(300, distance*10), false), ""
}
