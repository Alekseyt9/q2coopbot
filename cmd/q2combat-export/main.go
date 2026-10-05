package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"

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

func run() error {
	input := flag.String("trace", "", "Client trace JSONL")
	out := flag.String("out", "", "Fresh output directory")
	worker := flag.String("worker", "", "Worker identity")
	episode := flag.String("episode", "", "Episode identity")
	endReason := flag.String("end-reason", "trace_end", "Reason for truncating the final command")
	serverLog := flag.String("server-log", "", "Optional native log for separate server damage windows")
	clientName := flag.String("client-name", "GoCoopMate", "Name selected by native sv_test_trace_client")
	requireExecution := flag.Bool("require-execution", false, "Reject export without exact server dispatch for every sent command")
	resetFile := flag.String("reset-expectation", "", "Optional fixture JSON; reject mismatched first usable observation")
	synchronous := flag.Bool("synchronous", false, "Require native one-command/one-tick phases and seeded single-client barrier")
	rewardFile := flag.String("reward-config", "", "Optional explicit experimental reward JSON; requires synchronous proof")
	flag.Parse()
	if *input == "" || *out == "" || *worker == "" || *episode == "" {
		return fmt.Errorf("trace, out, worker and episode required")
	}
	var joiner *learningenv.DamageJoiner
	var executions *learningenv.ExecutionIndex
	var applied []harness.AppliedCommand
	var resetExpectation *learningenv.ResetExpectation
	var native *learningenv.NativeSteps
	var rewardConfig *learningenv.RewardConfig
	if *rewardFile != "" {
		if !*synchronous {
			return fmt.Errorf("reward export requires synchronous native proof")
		}
		data, err := os.ReadFile(*rewardFile)
		if err != nil {
			return err
		}
		if len(data) > 16384 {
			return fmt.Errorf("reward config too large")
		}
		d := json.NewDecoder(bytes.NewReader(data))
		d.DisallowUnknownFields()
		rewardConfig = &learningenv.RewardConfig{}
		if err := d.Decode(rewardConfig); err != nil {
			return err
		}
		var extra any
		if err := d.Decode(&extra); err != io.EOF {
			return fmt.Errorf("trailing reward config data")
		}
		if err := rewardConfig.Validate(); err != nil {
			return err
		}
	}
	if *resetFile != "" {
		data, err := os.ReadFile(*resetFile)
		if err != nil {
			return err
		}
		if len(data) > 16384 {
			return fmt.Errorf("reset expectation too large")
		}
		d := json.NewDecoder(bytes.NewReader(data))
		d.DisallowUnknownFields()
		resetExpectation = &learningenv.ResetExpectation{}
		if err := d.Decode(resetExpectation); err != nil {
			return err
		}
		var extra any
		if err := d.Decode(&extra); err != io.EOF {
			return fmt.Errorf("trailing reset expectation data")
		}
	}
	if *requireExecution && *serverLog == "" {
		return fmt.Errorf("server-log required for execution proof")
	}
	if *synchronous && (!*requireExecution || resetExpectation == nil || resetExpectation.Seed == nil) {
		return fmt.Errorf("synchronous export requires execution proof and reset expectation with per-episode seed")
	}
	if *serverLog != "" {
		f, err := os.Open(*serverLog)
		if err != nil {
			return err
		}
		events, parseErr := learningenv.ReadDamageEvents(f)
		f.Close()
		if parseErr != nil {
			return parseErr
		}
		joiner = &learningenv.DamageJoiner{Events: events}
		applied, err = harness.ReadAppliedCommandsForClient(*serverLog, *clientName)
		if err != nil {
			return err
		}
		if len(applied) > 0 {
			executions = learningenv.NewExecutionIndex(applied)
		}
		if *synchronous {
			f, err := os.Open(*serverLog)
			if err != nil {
				return err
			}
			native, err = learningenv.ReadNativeSteps(f, events)
			f.Close()
			if err != nil {
				return err
			}
		}
	}
	if err := os.Mkdir(*out, 0755); err != nil {
		return err
	}
	f, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer f.Close()
	steps, err := os.Create(filepath.Join(*out, "steps.jsonl"))
	if err != nil {
		return err
	}
	defer steps.Close()
	outcomes, err := os.Create(filepath.Join(*out, "outcomes.jsonl"))
	if err != nil {
		return err
	}
	defer outcomes.Close()
	stepWriter, outcomeWriter := bufio.NewWriter(steps), bufio.NewWriter(outcomes)
	se, oe := json.NewEncoder(stepWriter), json.NewEncoder(outcomeWriter)
	var serverWriter *bufio.Writer
	var serverEncoder *json.Encoder
	if joiner != nil {
		f, err := os.Create(filepath.Join(*out, "server_outcomes.jsonl"))
		if err != nil {
			return err
		}
		defer f.Close()
		serverWriter = bufio.NewWriter(f)
		serverEncoder = json.NewEncoder(serverWriter)
	}
	a := learningenv.Assembler{Worker: *worker, Episode: *episode}
	var rewardWriter *bufio.Writer
	var rewardEncoder *json.Encoder
	if rewardConfig != nil {
		f, err := os.Create(filepath.Join(*out, "rewards.jsonl"))
		if err != nil {
			return err
		}
		defer f.Close()
		rewardWriter = bufio.NewWriter(f)
		rewardEncoder = json.NewEncoder(rewardWriter)
		data, err := json.MarshalIndent(rewardConfig, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*out, "reward-config.json"), data, 0644); err != nil {
			return err
		}
	}
	rewardSteps := 0
	rewardSum := 0.0
	rewardMasks := map[string]int{}
	count, terminals, truncated := 0, 0, 0
	serverWindows := 0
	exclusiveWindows := 0
	var sent []harness.Trace
	var initial *policy.Observation
	var resetProof *learningenv.ResetProof
	emit := func(s *learningenv.Step, o *learningenv.Outcome) error {
		if s == nil {
			return nil
		}
		if initial == nil {
			copy := s.Observation
			initial = &copy
			if resetExpectation != nil {
				p := learningenv.VerifyReset(copy, *resetExpectation)
				resetProof = &p
				if err := p.Error(); err != nil {
					return err
				}
				if native != nil {
					if native.Release.Spawncount != copy.Identity.Spawncount || native.Release.Frame != copy.Identity.Frame || native.Release.Seed != *resetExpectation.Seed {
						return fmt.Errorf("native reset release does not match first observation or episode seed")
					}
					if copy.Inventory == nil || copy.InventoryAgeFrames == nil || *copy.InventoryAgeFrames < 0 || *copy.InventoryAgeFrames > 2 {
						return fmt.Errorf("native reset inventory observation unavailable/stale")
					}
					resetProof.NativeBarrier = native.Release
				}
			}
		}
		if executions != nil {
			e := executions.Match(s)
			s.Execution = &e
			if e.WindowExclusive {
				exclusiveWindows++
			}
		}
		if native != nil {
			p, err := native.Match(s)
			if err != nil {
				return err
			}
			s.Native = p
			if !s.Execution.Matched || s.Next != nil && !s.Execution.WindowExclusive {
				return fmt.Errorf("native step dispatch not exclusive/aligned")
			}
		}
		if err := se.Encode(s); err != nil {
			return err
		}
		if err := oe.Encode(o); err != nil {
			return err
		}
		if joiner != nil {
			var outcome learningenv.ServerOutcome
			if native != nil {
				outcome = joiner.JoinNative(s, s.Native)
			} else {
				outcome = joiner.Join(s)
			}
			if outcome.Available {
				serverWindows++
			}
			if err := serverEncoder.Encode(outcome); err != nil {
				return err
			}
			if rewardConfig != nil {
				r := rewardConfig.Evaluate(s, outcome)
				if r.Available {
					rewardSteps++
					rewardSum += *r.Score
				} else {
					rewardMasks[r.Reason]++
				}
				if err := rewardEncoder.Encode(r); err != nil {
					return err
				}
			}
		}
		count++
		if s.Terminal {
			terminals++
		}
		if s.Truncated {
			truncated++
		}
		return nil
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 65536), 8*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		var row struct {
			Capture  *policy.Capture `json:"combat_policy"`
			Command  quake.UserCmd   `json:"sent_command"`
			Sequence uint32          `json:"client_sequence"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if row.Capture == nil || row.Capture.AppliedCommand != row.Command {
			return fmt.Errorf("line %d: missing/mismatched capture", line)
		}
		if row.Capture.ClientSequence != 0 && row.Capture.ClientSequence != row.Sequence {
			return fmt.Errorf("line %d: capture sequence mismatch", line)
		}
		row.Capture.ClientSequence = row.Sequence
		id := row.Capture.Observation.Identity
		sent = append(sent, harness.Trace{Connection: id.Connection, Generation: id.Spawncount, ClientSequence: row.Sequence, Command: row.Command})
		s, o, err := a.Push(*row.Capture)
		if err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if err := emit(s, o); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	proof := harness.VerifyAppliedCommands(sent, applied)
	if native != nil && len(native.Steps) != len(sent) {
		return fmt.Errorf("native pulse count differs from full sent trace")
	}
	if *requireExecution && !proof.Accepted {
		return fmt.Errorf("server execution proof failed: %s generation=%d sequence=%d", proof.Reason, proof.Generation, proof.Sequence)
	}
	s, o := a.Close(*endReason)
	if err := emit(s, o); err != nil {
		return err
	}
	if err := stepWriter.Flush(); err != nil {
		return err
	}
	if err := outcomeWriter.Flush(); err != nil {
		return err
	}
	if serverWriter != nil {
		if err := serverWriter.Flush(); err != nil {
			return err
		}
	}
	if count == 0 {
		return fmt.Errorf("no usable steps")
	}
	if rewardWriter != nil {
		if err := rewardWriter.Flush(); err != nil {
			return err
		}
	}
	start := struct {
		Version        string                  `json:"version"`
		Observation    *policy.Observation     `json:"first_usable_observation"`
		ResetConfirmed bool                    `json:"fixture_reset_confirmed"`
		Scope          string                  `json:"scope"`
		ResetProof     *learningenv.ResetProof `json:"observed_reset_proof"`
	}{"observed_episode_start_v1", initial, false, "First usable client observation after harness overrides, with optional observed-field fixture proof. Full server state/RNG reset is unconfirmed.", resetProof}
	data, err := json.MarshalIndent(start, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*out, "episode_start.json"), data, 0644); err != nil {
		return err
	}
	summary := struct {
		Version                string                    `json:"version"`
		Steps                  int                       `json:"steps"`
		Terminals              int                       `json:"terminals"`
		Truncated              int                       `json:"truncated"`
		Scope                  string                    `json:"scope"`
		ServerWindows          int                       `json:"server_windows"`
		ServerEvents           int                       `json:"server_events"`
		JoinedEvents           int                       `json:"joined_server_events"`
		UnassignedEvents       int                       `json:"unassigned_server_events"`
		CommandProof           harness.CommandProof      `json:"command_proof"`
		ExclusiveWindows       int                       `json:"exclusive_execution_windows"`
		ObservedResetConfirmed bool                      `json:"observed_reset_confirmed"`
		NativeSteps            int                       `json:"native_steps"`
		SynchronousConfirmed   bool                      `json:"synchronous_confirmed"`
		RewardConfig           *learningenv.RewardConfig `json:"reward_config,omitempty"`
		RewardSteps            int                       `json:"reward_steps,omitempty"`
		RewardSum              *float64                  `json:"reward_sum,omitempty"`
		RewardMasks            map[string]int            `json:"reward_masks,omitempty"`
	}{Version: learningenv.StepVersion, Steps: count, Terminals: terminals, Truncated: truncated,
		Scope: "Observed transitions with optional exact native dispatch proof; server damage windows describe effects, not shot accuracy or delayed causal credit. No victory or scalar reward inferred.", ServerWindows: serverWindows, CommandProof: proof, ExclusiveWindows: exclusiveWindows}
	if resetProof != nil {
		summary.ObservedResetConfirmed = resetProof.ObservedFieldsConfirmed
	}
	if rewardConfig != nil {
		summary.RewardConfig = rewardConfig
		summary.RewardSteps = rewardSteps
		summary.RewardSum = &rewardSum
		summary.RewardMasks = rewardMasks
		summary.Scope = "Observed transitions, exact native execution and experimental first-life reward from phased server effects. No victory, shot accuracy or demonstration quality inferred."
	}
	if native != nil {
		summary.NativeSteps = len(native.Steps)
		summary.SynchronousConfirmed = true
	}
	if joiner != nil {
		summary.ServerEvents = len(joiner.Events)
		summary.JoinedEvents = len(joiner.Used)
		summary.UnassignedEvents = summary.ServerEvents - summary.JoinedEvents
	}
	data, err = json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*out, "report.json"), data, 0644)
}
