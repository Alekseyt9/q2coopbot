package bot

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// ConfigFile is the user-facing JSON format. Test-only controls are kept in a
// separate section so ordinary companion settings stay easy to read.
type ConfigFile struct {
	Server struct {
		Host string `json:"host"`
		Port int    `json:"port"`
	} `json:"server"`
	Client struct {
		Name    string `json:"name"`
		GameDir string `json:"game_dir"`
		AASDir  string `json:"aas_dir"`
	} `json:"client"`
	Run struct {
		Duration   string `json:"duration"`
		FramePaced bool   `json:"frame_paced"`
		GameFrames int    `json:"game_frames"`
	} `json:"run"`
	Models struct {
		System1 string `json:"system1"`
		System2 string `json:"system2"`
	} `json:"models"`
	Output struct {
		WorldJSON  string `json:"world_json"`
		TraceJSONL string `json:"trace_jsonl"`
		StopFile   string `json:"stop_file"`
	} `json:"output"`
	Test struct {
		Idle                      bool   `json:"idle"`
		ExitOnReconnect           bool   `json:"exit_on_reconnect"`
		ChangeMap                 string `json:"change_map"`
		ChangeAfterFrames         int    `json:"change_after_frames"`
		TeleportMap               string `json:"teleport_map"`
		Teleport                  string `json:"teleport"`
		TeleportAfter             string `json:"teleport_after"`
		TeleportAfterFrames       int    `json:"teleport_after_frames"`
		TeleportReturn            string `json:"teleport_return"`
		TeleportReturnAfterFrames int    `json:"teleport_return_after_frames"`
		JumpAfterTeleportFrames   int    `json:"jump_after_teleport_frames"`
		SpawnMap                  string `json:"spawn_map"`
		SpawnSoldier              string `json:"spawn_soldier"`
		SpawnClass                string `json:"spawn_class"`
		ObservationGapStart       int    `json:"observation_gap_start"`
		ObservationGapFrames      int    `json:"observation_gap_frames"`
		LineCross                 bool   `json:"line_cross"`
		HoldPosition              bool   `json:"hold_position"`
		GroundEdgeProbe           bool   `json:"ground_edge_probe"`
		NoAAS                     bool   `json:"no_aas"`
		DoorProbe                 bool   `json:"door_probe"`
		DoorPassProbe             bool   `json:"door_pass_probe"`
		ButtonProbe               bool   `json:"button_probe"`
		ButtonAutoGoal            bool   `json:"button_auto_goal"`
		NoBSP                     bool   `json:"no_bsp"`
		PartialBSP                bool   `json:"partial_bsp"`
		HideDoor53                bool   `json:"hide_door_53"`
	} `json:"test"`
}

func LoadConfig(path string) (Config, error) {
	var cfg Config
	absPath, err := filepath.Abs(path)
	if err != nil {
		return cfg, err
	}
	f, err := os.Open(absPath)
	if err != nil {
		return cfg, fmt.Errorf("open config: %w", err)
	}
	defer f.Close()
	decoder := json.NewDecoder(f)
	decoder.DisallowUnknownFields()
	var file ConfigFile
	if err := decoder.Decode(&file); err != nil {
		return cfg, fmt.Errorf("decode config: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		if err == nil {
			return cfg, fmt.Errorf("config contains more than one JSON value")
		}
		return cfg, fmt.Errorf("trailing config data: %w", err)
	}
	resolve := func(value string) string {
		if value == "" {
			return ""
		}
		if filepath.IsAbs(value) {
			return filepath.Clean(value)
		}
		return filepath.Join(filepath.Dir(absPath), value)
	}
	cfg.Host, cfg.Port, cfg.Name = file.Server.Host, file.Server.Port, file.Client.Name
	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.Port == 0 {
		cfg.Port = 27910
	}
	if cfg.Name == "" {
		cfg.Name = "GoCoopMate"
	}
	cfg.GameDir, cfg.AASDir = resolve(file.Client.GameDir), resolve(file.Client.AASDir)
	cfg.WorldFile, cfg.TracePath, cfg.StopFile = resolve(file.Output.WorldJSON), resolve(file.Output.TraceJSONL), resolve(file.Output.StopFile)
	cfg.FramePaced, cfg.GameFrames = file.Run.FramePaced, file.Run.GameFrames
	if file.Run.Duration != "" {
		cfg.Duration, err = time.ParseDuration(file.Run.Duration)
		if err != nil || cfg.Duration < 0 {
			return Config{}, fmt.Errorf("invalid run.duration %q: expected a non-negative Go duration", file.Run.Duration)
		}
	}
	cfg.System1Model, cfg.System2Model = file.Models.System1, file.Models.System2
	cfg.Idle, cfg.ExitOnReconnect = file.Test.Idle, file.Test.ExitOnReconnect
	cfg.TestChangeMap, cfg.TestChangeAfter = file.Test.ChangeMap, file.Test.ChangeAfterFrames
	if cfg.TestChangeAfter == 0 {
		cfg.TestChangeAfter = 20
	}
	cfg.TestTeleportMap, cfg.TestTeleport = file.Test.TeleportMap, file.Test.Teleport
	cfg.TestTeleportAfter, cfg.TestTeleportAfterFrames = file.Test.TeleportAfter, file.Test.TeleportAfterFrames
	cfg.TestTeleportReturn, cfg.TestTeleportReturnAfterFrames = file.Test.TeleportReturn, file.Test.TeleportReturnAfterFrames
	cfg.TestJumpAfterTeleportFrames = file.Test.JumpAfterTeleportFrames
	cfg.TestSpawnMap, cfg.TestSpawnSoldier, cfg.TestSpawnClass = file.Test.SpawnMap, file.Test.SpawnSoldier, file.Test.SpawnClass
	cfg.TestGapStart, cfg.TestGapFrames = file.Test.ObservationGapStart, file.Test.ObservationGapFrames
	cfg.TestLineCross, cfg.TestHoldPosition = file.Test.LineCross, file.Test.HoldPosition
	cfg.TestGroundEdgeProbe = file.Test.GroundEdgeProbe
	cfg.TestNoAAS = file.Test.NoAAS
	cfg.TestDoorProbe = file.Test.DoorProbe
	cfg.TestDoorPassProbe = file.Test.DoorPassProbe
	cfg.TestButtonProbe = file.Test.ButtonProbe
	cfg.TestButtonAutoGoal = file.Test.ButtonAutoGoal
	cfg.TestNoBSP, cfg.TestPartialBSP = file.Test.NoBSP, file.Test.PartialBSP
	cfg.TestHideDoor53 = file.Test.HideDoor53
	return cfg, nil
}
