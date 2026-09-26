package bot

import (
	"fmt"
	"q2coopbot/internal/harness"
	"time"
)

func configureSession(cfg *Config) (*harness.Session, *harness.SessionRunner, error) {
	if cfg.TestSession == "" {
		if cfg.TestSessionRole != "" {
			return nil, nil, fmt.Errorf("session_role requires test.session")
		}
		return nil, nil, nil
	}
	if !cfg.FramePaced || cfg.GameFrames != 0 || cfg.Duration <= 0 || cfg.TestScenario != "" || cfg.TestChangeMap != "" || cfg.TestTeleport != "" || cfg.TestTeleportAfter != "" || cfg.TestWalkTarget != "" || cfg.TestSpawnSoldier != "" || cfg.TestLineCross || cfg.TestGapFrames != 0 || cfg.TestHoldPosition || cfg.TestJumpAfterTeleportFrames != 0 || cfg.TestJumpAgainAfterTeleportFrames != 0 {
		return nil, nil, fmt.Errorf("test.session requires frame pacing, wall duration, no game_frames and no other scenario/movement controls")
	}
	if cfg.TestSessionRole != "actor" && cfg.TestSessionRole != "observer" {
		return nil, nil, fmt.Errorf("session_role must be actor or observer")
	}
	if cfg.TestScenarioFrameOrigin != 0 || cfg.ExitOnReconnect || cfg.TestGroundEdgeProbe || cfg.TestDoorProbe || cfg.TestDoorPassProbe || cfg.TestButtonProbe || cfg.TestButtonAutoGoal || cfg.TestTeleportReturn != "" {
		return nil, nil, fmt.Errorf("test.session cannot combine with legacy setup, reconnect exit or gameplay probes")
	}
	actor := cfg.TestSessionRole == "actor"
	if actor != cfg.Idle || actor && cfg.TestRCONPassword == "" {
		return nil, nil, fmt.Errorf("session actor requires idle and RCON; observer must run bot policy")
	}
	s, err := harness.LoadSession(cfg.TestSession)
	if err != nil {
		return nil, nil, err
	}
	first := s.Phases[0].Scenario
	if s.ReadinessBarrier && cfg.TestScenarioResult == "" {
		return nil, nil, fmt.Errorf("readiness barrier requires scenario_result in a fresh run directory")
	}
	position := first.BotOrigin
	if actor {
		position = first.ActorOrigin
	}
	cfg.TestTeleportMap = first.Map
	cfg.TestTeleport = fmt.Sprintf("%g,%g,%g", position[0], position[1], position[2])
	cfg.TestSetupHoldFrames = 1
	if !actor {
		return &s, nil, nil
	}
	r, err := harness.NewSession(s)
	return &s, r, err
}

func (c *Client) prepareSessionPhase() error {
	if c.sessionDefinition == nil || !c.begun || !c.frameReady {
		return nil
	}
	mapName := c.planner.World.Map
	if c.sessionMap != "" && c.sessionGeneration == c.spawncount {
		if c.sessionMap != mapName {
			return fmt.Errorf("session map changed without generation")
		}
		return nil
	}
	index := c.sessionPhase
	if c.sessionMap != "" {
		index++
	}
	if index >= len(c.sessionDefinition.Phases) || c.sessionDefinition.Phases[index].Scenario.Map != mapName {
		return fmt.Errorf("unexpected session map %s at phase %d", mapName, index)
	}
	phase := c.sessionDefinition.Phases[index].Scenario
	if !c.sessionDefinition.ReadinessBarrier && c.latestFrame >= phase.StartFrame {
		return fmt.Errorf("session placement too late: frame %d, phase start %d", c.latestFrame, phase.StartFrame)
	}
	c.sessionPhase, c.sessionMap, c.sessionGeneration = index, mapName, c.spawncount
	c.testTeleportMap = phase.Map
	c.testTeleportPosition = phase.BotOrigin
	if c.session != nil {
		c.testTeleportPosition = phase.ActorOrigin
	}
	c.testTeleportSent = false
	c.sessionReadySent, c.sessionStartFrame = false, 0
	c.sessionSetupAt = time.Now()
	c.scenarioPath = testWalkPath{}
	return nil
}

func (c *Client) sessionStatus() *harness.SessionStatus {
	if c.session == nil {
		return nil
	}
	s := c.session.Status
	return &s
}
