package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"
	"q2coopbot/internal/harness/netfault"
	"strings"
)

func run() error {
	path := flag.String("config", "", "report config JSON")
	flag.Parse()
	if *path == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: q2netfault-report --config config.json")
	}
	f, err := os.Open(*path)
	if err != nil {
		return err
	}
	defer f.Close()
	var cfg struct {
		Network              netfault.Config `json:"network"`
		Events               string          `json:"events"`
		Output               string          `json:"output"`
		RecoveryMS           int             `json:"recovery_ms"`
		RequireUpstreamLoss  bool            `json:"require_upstream_loss"`
		BotTrace             string          `json:"bot_trace"`
		ActorTrace           string          `json:"actor_trace"`
		FollowRecoveryFrames int             `json:"follow_recovery_frames"`
		ServerLog            string          `json:"server_log"`
	}
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err = d.Decode(&cfg); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing config")
	}
	if cfg.Events == "" || cfg.Output == "" {
		return fmt.Errorf("events and output required")
	}
	if cfg.ServerLog != "" && cfg.BotTrace == "" {
		return fmt.Errorf("server_log requires bot_trace")
	}
	if (cfg.ActorTrace != "" || cfg.FollowRecoveryFrames != 0) && (cfg.ActorTrace == "" || cfg.BotTrace == "" || cfg.FollowRecoveryFrames < 1 || cfg.FollowRecoveryFrames > 1000) {
		return fmt.Errorf("follow check requires actor_trace, bot_trace and follow_recovery_frames 1..1000")
	}
	resolve := func(p string) string {
		if filepath.IsAbs(p) {
			return filepath.Clean(p)
		}
		return filepath.Join(filepath.Dir(*path), p)
	}
	outputPath, err := filepath.Abs(resolve(cfg.Output))
	if err != nil {
		return err
	}
	inputs := []string{resolve(cfg.Events), *path}
	if cfg.BotTrace != "" {
		inputs = append(inputs, resolve(cfg.BotTrace))
	}
	if cfg.ActorTrace != "" {
		inputs = append(inputs, resolve(cfg.ActorTrace))
	}
	if cfg.ServerLog != "" {
		inputs = append(inputs, resolve(cfg.ServerLog))
	}
	for _, input := range inputs {
		inputPath, err := filepath.Abs(input)
		if err != nil {
			return err
		}
		if strings.EqualFold(inputPath, outputPath) {
			return fmt.Errorf("output cannot replace an input")
		}
		outInfo, outErr := os.Stat(outputPath)
		inInfo, inErr := os.Stat(inputPath)
		if outErr == nil && inErr == nil && os.SameFile(outInfo, inInfo) {
			return fmt.Errorf("output aliases an input")
		}
	}
	rows, err := netfault.ReadEvents(resolve(cfg.Events))
	var r struct {
		netfault.Report
		Game     *harness.NetworkFramesReport `json:"game,omitempty"`
		Follow   *harness.NetworkFollowReport `json:"follow,omitempty"`
		Commands *harness.NetworkCommandProof `json:"observer_commands,omitempty"`
	}
	if err != nil {
		r.Report = netfault.Report{State: "trace_invalid", Reason: err.Error()}
	} else {
		r.Report = netfault.Analyze(cfg.Network, rows, cfg.RecoveryMS, cfg.RequireUpstreamLoss)
	}
	if r.Accepted && cfg.BotTrace != "" {
		bot, readErr := harness.ReadTrace(resolve(cfg.BotTrace))
		if readErr != nil {
			r.Accepted = false
			r.State = "trace_invalid"
			r.Reason = readErr.Error()
		} else {
			game := harness.VerifyNetworkFrames(rows, bot)
			r.Game = &game
			if !game.Accepted {
				r.Accepted = false
				r.State = game.State
				r.Reason = game.Reason
			}
			if r.Accepted && cfg.ServerLog != "" {
				applied, appliedErr := harness.ReadAppliedCommands(resolve(cfg.ServerLog))
				if appliedErr != nil {
					r.Accepted = false
					r.State = "trace_invalid"
					r.Reason = appliedErr.Error()
				} else {
					proof := harness.VerifyNetworkCommands(rows, bot, applied)
					r.Commands = &proof
					if !proof.Accepted {
						r.Accepted = false
						r.State = "trace_invalid"
						r.Reason = proof.Reason
					}
				}
			}
			if r.Accepted && cfg.ActorTrace != "" {
				actor, actorErr := harness.ReadTrace(resolve(cfg.ActorTrace))
				if actorErr != nil {
					r.Accepted = false
					r.State = "trace_invalid"
					r.Reason = actorErr.Error()
				} else {
					follow := harness.VerifyNetworkFollow(rows, bot, actor, cfg.FollowRecoveryFrames)
					r.Follow = &follow
					if !follow.Accepted {
						r.Accepted = false
						r.State = follow.State
						r.Reason = follow.Reason
					}
				}
			}
		}
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(resolve(cfg.Output), data, 0600); err != nil {
		return err
	}
	if !r.Accepted {
		return fmt.Errorf("network report rejected: %s", r.Reason)
	}
	fmt.Println("Network transport accepted:", resolve(cfg.Output))
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
