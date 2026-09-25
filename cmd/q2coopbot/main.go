package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"

	"q2coopbot/internal/bot"
)

func main() {
	var cfg bot.Config
	flag.StringVar(&cfg.Host, "host", "127.0.0.1", "server host")
	flag.IntVar(&cfg.Port, "port", 27910, "server UDP port")
	flag.StringVar(&cfg.Name, "name", "GoCoopMate", "client name")
	flag.StringVar(&cfg.GameDir, "game-dir", "", "baseq2 directory containing maps/*.aas")
	flag.StringVar(&cfg.AASDir, "aas-dir", "", "directory containing campaign AAS files; default game-dir/maps")
	flag.DurationVar(&cfg.Duration, "duration", 0, "session duration; 0 runs until Ctrl-C")
	flag.BoolVar(&cfg.FramePaced, "frame-paced", false, "send one 100 ms usercmd per received game frame (for accelerated local tests)")
	flag.IntVar(&cfg.GameFrames, "game-frames", 0, "stop after this many observed game frames in frame-paced mode; 0 disables")
	flag.StringVar(&cfg.WorldFile, "world-json", "", "current world state output")
	flag.StringVar(&cfg.TracePath, "trace-jsonl", "", "write one observation and usercmd per received game frame")
	flag.StringVar(&cfg.StopFile, "stop-file", "", "disconnect when file appears")
	flag.BoolVar(&cfg.Idle, "idle", false, "send neutral movement commands as a stationary test player")
	flag.BoolVar(&cfg.ExitOnReconnect, "test-exit-on-reconnect", false, "test only: leave when server changes map")
	flag.StringVar(&cfg.TestChangeMap, "test-change-map", "", "test only: switch the server to this map through local RCON")
	flag.IntVar(&cfg.TestChangeAfter, "test-change-after-frames", 20, "test only: game frames before requesting map change")
	flag.StringVar(&cfg.System2Model, "system2-model", "", "optional Ollama strategy model, called asynchronously every 2 seconds")
	flag.StringVar(&cfg.System1Model, "system1-model", "", "optional Ollama tactical model, called asynchronously every 1 second")
	flag.Parse()
	cfg.TestRCONPassword = os.Getenv("Q2COOPBOT_TEST_RCON")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := bot.Run(ctx, cfg); err != nil {
		log.Fatal(err)
	}
}
