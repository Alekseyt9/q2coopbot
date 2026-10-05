// q2learning-relay streams verified first-life transitions from the existing
// live harness logs. It never drives the UDP client or edits observations.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func readJSON(path string, dst any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(data) > 16384 {
		return fmt.Errorf("relay config too large")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing relay config data")
	}
	return nil
}

func run() (resultErr error) {
	trace := flag.String("trace", "", "Growing capture JSONL")
	server := flag.String("server-log", "", "Growing native server log")
	provider := flag.String("provider", "", "Remote provider JSON with episode/seed")
	rewardPath := flag.String("reward-config", "", "Explicit reward JSON")
	resetPath := flag.String("reset-expectation", "", "Observed fixture JSON with seed")
	out := flag.String("out", "", "Fresh relay output directory")
	stop := flag.String("stop-file", "", "Worker completion marker")
	worker := flag.String("worker", "", "Worker identity")
	episode := flag.String("episode", "", "Episode identity")
	gameFrames := flag.Int("game-frames", 0, "Known harness frame limit; zero uses worker completion marker")
	flag.Parse()
	if *trace == "" || *server == "" || *provider == "" || *rewardPath == "" || *resetPath == "" || *out == "" || *stop == "" || *worker == "" || *episode == "" {
		return fmt.Errorf("relay paths and identities required")
	}
	if err := os.Mkdir(*out, 0755); err != nil {
		return err
	}
	summary := struct {
		Version          string `json:"version"`
		Accepted         bool   `json:"accepted"`
		Steps            int    `json:"steps"`
		RewardSteps      int    `json:"reward_steps"`
		Terminal         bool   `json:"terminal"`
		Truncated        bool   `json:"truncated"`
		Reason           string `json:"reason,omitempty"`
		Error            string `json:"error,omitempty"`
		FirstDeliveredNS int64  `json:"first_delivered_unix_ns"`
		LastDeliveredNS  int64  `json:"last_delivered_unix_ns"`
	}{Version: learningenv.FeedbackVersion}
	defer func() {
		if resultErr != nil {
			summary.Error = resultErr.Error()
			summary.Accepted = false
		}
		data, _ := json.MarshalIndent(summary, "", "  ")
		os.WriteFile(filepath.Join(*out, "report.json"), data, 0644)
	}()
	var coefficients learningenv.RewardConfig
	if err := readJSON(*rewardPath, &coefficients); err != nil {
		return err
	}
	if err := coefficients.Validate(); err != nil {
		return err
	}
	var expected learningenv.ResetExpectation
	if err := readJSON(*resetPath, &expected); err != nil {
		return err
	}
	if expected.Seed == nil {
		return fmt.Errorf("relay requires episode seed")
	}
	var remoteConfig policy.RemoteConfig
	if err := readJSON(*provider, &remoteConfig); err != nil {
		return err
	}
	if remoteConfig.Seed != *expected.Seed || remoteConfig.Episode != *worker+"-"+*episode {
		return fmt.Errorf("relay worker/episode/seed mismatch")
	}
	p, err := policy.LoadProvider(*provider, true)
	if err != nil {
		return err
	}
	remote, ok := p.(*policy.Remote)
	if !ok {
		return fmt.Errorf("relay requires remote provider")
	}
	defer remote.Close()
	journal, err := os.Create(filepath.Join(*out, "feedback.jsonl"))
	if err != nil {
		return err
	}
	defer journal.Close()
	encoder := json.NewEncoder(journal)
	send := func(event learningenv.FeedbackEvent) error {
		if err := remote.Notify(event); err != nil {
			return err
		}
		now := time.Now().UnixNano()
		if summary.FirstDeliveredNS == 0 {
			summary.FirstDeliveredNS = now
		}
		summary.LastDeliveredNS = now
		return encoder.Encode(event)
	}
	var traceLog, serverLog learningenv.AppendLog
	native := learningenv.LiveTelemetry{ClientName: "SoloRetreatBot"}
	assembler := learningenv.Assembler{Worker: *worker, Episode: *episode}
	var queue []*learningenv.Step
	closed, resetSent := false, false
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		lines, err := serverLog.Read(*server)
		if err != nil {
			return err
		}
		for _, line := range lines {
			if err := native.Push(line); err != nil {
				return err
			}
		}
		if !closed {
			lines, err := traceLog.Read(*trace)
			if err != nil {
				return err
			}
			for _, line := range lines {
				var row struct {
					Capture       *policy.Capture `json:"combat_policy"`
					Sequence      uint32          `json:"client_sequence"`
					Command       quake.UserCmd   `json:"sent_command"`
					RelativeFrame int             `json:"relative_frame"`
				}
				if err := json.Unmarshal([]byte(line), &row); err != nil {
					return err
				}
				if row.Capture == nil {
					return fmt.Errorf("live capture missing")
				}
				if row.Capture.ClientSequence != row.Sequence {
					return fmt.Errorf("live capture sequence mismatch")
				}
				if row.Capture.AppliedCommand != row.Command {
					return fmt.Errorf("live sent command mismatch")
				}
				s, _, err := assembler.Push(*row.Capture)
				if err != nil {
					return err
				}
				if s != nil {
					if s.Observation.Identity.Life != 1 {
						return fmt.Errorf("relay crossed first-life boundary")
					}
					queue = append(queue, s)
					if s.Terminal || s.Truncated {
						closed = true
						break
					}
				}
				if *gameFrames > 0 && row.RelativeFrame >= *gameFrames {
					s, _ := assembler.Close("game_frame_limit")
					if s != nil {
						queue = append(queue, s)
					}
					closed = true
					break
				}
			}
			if _, err := os.Stat(*stop); err == nil && !closed {
				if len(traceLog.Partial) != 0 {
					return fmt.Errorf("incomplete final capture line")
				}
				s, _ := assembler.Close("game_frame_limit")
				if s != nil {
					queue = append(queue, s)
				}
				closed = true
			}
		}
		for len(queue) > 0 {
			s := queue[0]
			effects, ready, err := native.Enrich(s)
			if err != nil {
				return err
			}
			if !ready {
				break
			}
			if !resetSent {
				if native.Release == nil {
					break
				}
				id := s.Observation.Identity
				if native.Release.Seed != *expected.Seed || native.Release.Spawncount != id.Spawncount || native.Release.Frame != id.Frame {
					return fmt.Errorf("live reset release mismatch")
				}
				proof := learningenv.VerifyReset(s.Observation, expected)
				if err := proof.Error(); err != nil {
					return err
				}
				if s.Observation.Inventory == nil || s.Observation.InventoryAgeFrames == nil || *s.Observation.InventoryAgeFrames < 0 || *s.Observation.InventoryAgeFrames > 2 {
					return fmt.Errorf("live initial inventory unavailable")
				}
				proof.NativeBarrier = native.Release
				if err := send(learningenv.FeedbackEvent{Version: learningenv.FeedbackVersion, Kind: "reset", Initial: &s.Observation, Reset: &proof}); err != nil {
					return err
				}
				resetSent = true
			}
			reward := coefficients.Evaluate(s, effects)
			if err := send(learningenv.FeedbackEvent{Version: learningenv.FeedbackVersion, Kind: "step", Step: s, Reward: &reward, Effects: &effects}); err != nil {
				return err
			}
			summary.Steps++
			if reward.Available {
				summary.RewardSteps++
			}
			queue = queue[1:]
			if s.Terminal || s.Truncated {
				summary.Terminal = s.Terminal
				summary.Truncated = s.Truncated
				summary.Reason = s.Reason
				summary.Accepted = true
				return nil
			}
		}
		if closed && len(queue) == 0 {
			return fmt.Errorf("relay ended without terminal/truncated step")
		}
		time.Sleep(5 * time.Millisecond)
	}
	return fmt.Errorf("live feedback watchdog expired")
}
