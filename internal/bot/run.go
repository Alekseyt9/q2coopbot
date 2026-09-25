package bot

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"q2coopbot/internal/quake"
)

// Config contains runtime settings for one UDP companion session.
type Config struct {
	Host, Name, GameDir, AASDir       string
	WorldFile, TracePath, StopFile    string
	System1Model, System2Model        string
	TestChangeMap, TestRCONPassword   string
	Port, GameFrames, TestChangeAfter int
	Duration                          time.Duration
	FramePaced, Idle, ExitOnReconnect bool
}

// The test server accepts the destination and entry as separate RCON arguments.
// Its command joins them after console macro expansion, which would consume '$'.
func transitionMapArgument(destination, previous string) (string, error) {
	validMap := regexp.MustCompile(`^[A-Za-z0-9_]+$`)
	if !validMap.MatchString(destination) || !validMap.MatchString(previous) {
		return "", fmt.Errorf("transition requires simple destination and previous map names")
	}
	return destination + " " + previous, nil
}

func Run(ctx context.Context, cfg Config) error {
	if cfg.GameDir == "" {
		return fmt.Errorf("--game-dir is required")
	}
	if cfg.GameFrames < 0 || cfg.GameFrames > 0 && !cfg.FramePaced {
		return fmt.Errorf("--game-frames requires --frame-paced and a non-negative value")
	}
	if cfg.TracePath != "" && !cfg.FramePaced {
		return fmt.Errorf("--trace-jsonl requires --frame-paced")
	}
	if cfg.TestChangeMap != "" {
		validMap := regexp.MustCompile(`^[A-Za-z0-9_]+$`)
		if !cfg.FramePaced || cfg.GameFrames <= cfg.TestChangeAfter || cfg.TestChangeAfter < 1 || !validMap.MatchString(cfg.TestChangeMap) {
			return fmt.Errorf("test map change requires --frame-paced, --game-frames greater than --test-change-after-frames, and a simple map name")
		}
		if cfg.TestRCONPassword == "" {
			return fmt.Errorf("Q2COOPBOT_TEST_RCON is required for test map change")
		}
	}
	if cfg.AASDir == "" {
		cfg.AASDir = filepath.Join(cfg.GameDir, "maps")
	}
	address, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("%s:%d", cfg.Host, cfg.Port))
	if err != nil {
		return err
	}
	conn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := &Client{
		conn: conn, address: address, qport: uint16(rand.Intn(65535) + 1), seq: 1,
		decoder: quake.NewDecoder(), planner: &Planner{AASDir: cfg.AASDir, GameClock: cfg.FramePaced},
		root: cfg.GameDir, worldFile: cfg.WorldFile, stopFile: cfg.StopFile, name: cfg.Name,
		idle: cfg.Idle, duration: cfg.Duration, framePaced: cfg.FramePaced, gameFrames: cfg.GameFrames,
		exitOnReconnect: cfg.ExitOnReconnect, testChangeMap: cfg.TestChangeMap,
		testChangeAfter: cfg.TestChangeAfter, testRconPassword: cfg.TestRCONPassword,
	}
	if cfg.TracePath != "" {
		client.traceFile, err = os.Create(cfg.TracePath)
		if err != nil {
			return err
		}
		defer client.traceFile.Close()
	}
	if cfg.System2Model != "" {
		client.strategist = NewStrategist(cfg.System2Model)
	}
	if cfg.System1Model != "" {
		client.tactician = NewTactician(cfg.System1Model)
	}
	if err := client.run(ctx); err != nil {
		return err
	}
	if cfg.WorldFile != "" {
		_ = client.writeWorld()
	}
	gameFPS := 0.0
	if client.framePaced && !client.firstMoveAt.IsZero() && client.lastMoveFrame > client.firstMoveFrame {
		gameFPS = float64(client.lastMoveFrame-client.firstMoveFrame) / time.Since(client.firstMoveAt).Seconds()
	}
	log.Printf("finished connected=%t begun=%t frames=%d moves=%d attacks=%d frame_paced=%t game_frames=%d frame_gaps=%d server_suppressed=%d game_fps=%.2f wall_s=%.2f map_changes=%d last_map=%s transition_timeout=%t decode_errors=%d last_error=%q", client.connected, client.begun, client.frames, client.moves, client.attacks, client.framePaced, max(0, client.lastMoveFrame-client.firstMoveFrame), client.frameGaps, client.suppressedFrames, gameFPS, time.Since(client.start).Seconds(), client.mapChanges, client.lastObservedMap, client.testChangeTimedOut, client.decoder.Errors, client.decoder.LastError)
	return nil
}
