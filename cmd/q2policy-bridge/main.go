// q2policy-bridge is a diagnostic peer for the learner protocol. It replays a
// probe; it does not train, choose tactics, or read server ground truth.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"sync"
	"time"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	address := flag.String("listen", "127.0.0.1:33000", "Loopback TCP listener")
	probePath := flag.String("probe", "", "Diagnostic probe JSON")
	logPath := flag.String("log", "", "Fresh request/response JSONL")
	version := flag.String("policy-version", "diagnostic_probe_bridge_v1", "Explicit diagnostic version expected by client")
	delay := flag.Int("delay-first-ms", 0, "Diagnostic delay for first response on each connection, 0..1000ms")
	flag.Parse()
	if err := policy.ValidateBridgeAddress(*address); err != nil {
		return err
	}
	if *probePath == "" || *logPath == "" || *version == "" || len(*version) > 128 || *delay < 0 || *delay > 1000 {
		return fmt.Errorf("probe, fresh log, valid version and delay required")
	}
	p, err := policy.LoadProbe(*probePath)
	if err != nil {
		return err
	}
	log, err := os.OpenFile(*logPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer log.Close()
	listener, err := net.Listen("tcp4", *address)
	if err != nil {
		return err
	}
	defer listener.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() { <-ctx.Done(); listener.Close() }()
	var mutex sync.Mutex
	encoder := json.NewEncoder(log)
	writeLog := func(v any) error { mutex.Lock(); defer mutex.Unlock(); return encoder.Encode(v) }
	if err := writeLog(map[string]any{"version": policy.BridgeVersion, "kind": "diagnostic_peer", "probe_version": p.Version(), "policy_version": *version, "delay_first_ms": *delay}); err != nil {
		return err
	}
	fmt.Printf("bridge_ready address=%s policy_version=%s\n", *address, *version)
	for {
		conn, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		go func() {
			defer conn.Close()
			probe := p.NewSession()
			reader := bufio.NewReader(conn)
			var session, episode string
			var seed int
			var previous uint64
			feedbackStarted, feedbackEnded := false, false
			lastStep := 0
			for {
				conn.SetDeadline(time.Now().Add(10 * time.Second))
				var req policy.BridgeRequest
				if err := policy.ReadBridgeMessage(reader, &req); err != nil {
					return
				}
				if req.Version != policy.BridgeVersion || req.PolicyVersion != *version || req.Session == "" || req.Episode == "" || req.Seed < 0 || req.Request <= previous {
					return
				}
				if previous > 0 && (req.Session != session || req.Episode != episode || req.Seed != seed) {
					return
				}
				session, episode, seed = req.Session, req.Episode, req.Seed
				received := time.Now().UnixNano()
				if previous == 0 && *delay > 0 {
					time.Sleep(time.Duration(*delay) * time.Millisecond)
				}
				previous = req.Request
				reply := policy.BridgeResponse{Version: policy.BridgeVersion, Session: req.Session, Request: req.Request, PolicyVersion: *version}
				if req.Kind == "feedback" {
					var event learningenv.FeedbackEvent
					if err := policy.ReadBridgeMessage(bufio.NewReader(bytes.NewReader(append(req.Feedback, '\n'))), &event); err != nil || event.Version != learningenv.FeedbackVersion || feedbackEnded {
						return
					}
					switch event.Kind {
					case "reset":
						if feedbackStarted || event.Initial == nil || event.Reset == nil || !event.Reset.ObservedFieldsConfirmed || event.Reset.NativeBarrier == nil || event.Reset.NativeBarrier.Seed != req.Seed {
							return
						}
						feedbackStarted = true
					case "step":
						if !feedbackStarted || event.Step == nil || event.Reward == nil || event.Effects == nil || event.Step.Observation.Identity.Life != 1 || event.Step.Index != lastStep+1 {
							return
						}
						lastStep = event.Step.Index
						feedbackEnded = event.Step.Terminal || event.Step.Truncated
					default:
						return
					}
					reply.FeedbackAck = true
				} else {
					if req.Kind != "" || len(req.Feedback) != 0 || req.Observation.Version != policy.ObservationVersion || req.Observation.Health <= 0 {
						return
					}
					action, err := probe.Decide(req.Observation)
					if err != nil {
						return
					}
					reply.Action = action
				}
				if err := policy.WriteBridgeMessage(conn, reply); err != nil {
					return
				}
				if err := writeLog(map[string]any{"request": req, "response": reply, "received_unix_ns": received, "sent_unix_ns": time.Now().UnixNano()}); err != nil {
					return
				}
			}
		}()
	}
}
