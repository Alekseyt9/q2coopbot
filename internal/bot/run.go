package bot

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"q2coopbot/internal/quake"
)

// Config contains runtime settings for one UDP companion session.
type Config struct {
	Host, Name, GameDir, AASDir       string
	WorldFile, TracePath, StopFile    string
	System1Model, System2Model        string
	TestChangeMap, TestRCONPassword   string
	TestTeleportMap, TestTeleport     string
	Port, GameFrames, TestChangeAfter int
	Duration                          time.Duration
	FramePaced, Idle, ExitOnReconnect bool
}

func parseTestTeleport(value string) (quake.Vec3, error) {
	var position quake.Vec3
	parts := strings.Split(value, ",")
	if len(parts) != 3 {
		return position, fmt.Errorf("test teleport position must be x,y,z")
	}
	for i, part := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) || v < -32768 || v > 32767 {
			return position, fmt.Errorf("invalid test teleport coordinate %q", part)
		}
		position[i] = v
	}
	return position, nil
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
	var teleportPosition quake.Vec3
	if cfg.TestTeleportMap != "" || cfg.TestTeleport != "" {
		if !cfg.FramePaced || !regexp.MustCompile(`^[A-Za-z0-9_]+$`).MatchString(cfg.TestTeleportMap) {
			return fmt.Errorf("test teleport requires --frame-paced and a simple --test-teleport-map")
		}
		var err error
		teleportPosition, err = parseTestTeleport(cfg.TestTeleport)
		if err != nil {
			return err
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
		testTeleportMap: cfg.TestTeleportMap, testTeleportPosition: teleportPosition,
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
