package policy

import (
	"bufio"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRemoteRejectsCrossRequestReplies(t *testing.T) {
	for _, name := range []string{"valid", "frame", "session", "request", "version", "policy", "timeout"} {
		t.Run(name, func(t *testing.T) {
			listener, err := net.Listen("tcp4", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			done := make(chan struct{})
			go func() {
				defer close(done)
				conn, err := listener.Accept()
				if err != nil {
					return
				}
				defer conn.Close()
				var req BridgeRequest
				if ReadBridgeMessage(bufio.NewReader(conn), &req) != nil {
					return
				}
				reply := BridgeResponse{Version: BridgeVersion, Session: req.Session, Request: req.Request, PolicyVersion: req.PolicyVersion, Action: Action{Version: ActionVersion, Identity: req.Observation.Identity, Forward: .3, Vertical: "release"}}
				switch name {
				case "frame":
					reply.Action.Identity.Frame++
				case "session":
					reply.Session = "old"
				case "request":
					reply.Request++
				case "version":
					reply.Version = "unknown"
				case "policy":
					reply.PolicyVersion = "other"
				case "timeout":
					time.Sleep(120 * time.Millisecond)
				}
				WriteBridgeMessage(conn, reply)
			}()
			config := RemoteConfig{Kind: RemoteKind, Address: listener.Addr().String(), TimeoutMS: 60, PolicyVersion: "test", Episode: "seed-9", Seed: 9}
			data, _ := json.MarshalIndent(config, "", "  ")
			path := filepath.Join(t.TempDir(), "provider.json")
			os.WriteFile(path, data, 0600)
			provider, err := LoadProvider(path, true)
			if err != nil {
				t.Fatal(err)
			}
			remote := provider.(*Remote)
			defer remote.Close()
			o := Observation{Version: ObservationVersion, Identity: Identity{Life: 1, Map: "base1", Connection: 1, Spawncount: 42, Actor: 1, Frame: 10}, Health: 100}
			start := time.Now()
			a, err := remote.Decide(o)
			if name == "valid" {
				if err != nil || a.Identity != o.Identity || a.Forward != .3 {
					t.Fatal(a, err)
				}
			} else {
				if err == nil || remote.conn != nil {
					t.Fatal("accepted bad reply or retained poisoned connection", a, err)
				}
			}
			if name == "timeout" && time.Since(start) > 300*time.Millisecond {
				t.Fatal("timeout not bounded")
			}
			<-done
		})
	}
}

func TestBridgeMessageBoundsAndStrictSchema(t *testing.T) {
	for _, text := range []string{"{\"version\":\"v\",\"secret\":true}\n", "{} {}\n", strings.Repeat("x", MaxBridgeMessage+1) + "\n", "{}"} {
		var reply BridgeResponse
		if ReadBridgeMessage(bufio.NewReader(strings.NewReader(text)), &reply) == nil {
			t.Fatal("accepted invalid message")
		}
	}
}

func TestRemoteRequiresLoopbackAndSynchronousBarrier(t *testing.T) {
	c := RemoteConfig{Kind: RemoteKind, Address: "127.0.0.1:33000", TimeoutMS: 200, PolicyVersion: "test", Episode: "seed-9", Seed: 9}
	for _, address := range []string{"localhost:33000", "192.168.1.1:33000", "0.0.0.0:33000", "127.0.0.1:1"} {
		bad := c
		bad.Address = address
		if bad.Validate() == nil {
			t.Fatal(address)
		}
	}
	data, _ := json.Marshal(c)
	path := filepath.Join(t.TempDir(), "provider.json")
	os.WriteFile(path, data, 0600)
	if _, err := LoadProvider(path, false); err == nil {
		t.Fatal("remote permitted in free-running server")
	}
}

func TestRemoteReconnectCannotConsumeTimedOutReply(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	var handlers sync.WaitGroup
	done := make(chan struct{})
	go func() {
		defer close(done)
		for n := 0; n < 2; n++ {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			handlers.Add(1)
			go func(n int, conn net.Conn) {
				defer handlers.Done()
				defer conn.Close()
				var req BridgeRequest
				if ReadBridgeMessage(bufio.NewReader(conn), &req) != nil {
					return
				}
				if n == 0 {
					time.Sleep(200 * time.Millisecond)
				}
				reply := BridgeResponse{Version: BridgeVersion, Session: req.Session, Request: req.Request, PolicyVersion: req.PolicyVersion, Action: Action{Version: ActionVersion, Identity: req.Observation.Identity, Side: float64(req.Request) / 10, Vertical: "release"}}
				WriteBridgeMessage(conn, reply)
			}(n, conn)
		}
		handlers.Wait()
	}()
	config := RemoteConfig{Kind: RemoteKind, Address: listener.Addr().String(), TimeoutMS: 80, PolicyVersion: "test", Episode: "seed-9", Seed: 9}
	data, _ := json.MarshalIndent(config, "", "  ")
	path := filepath.Join(t.TempDir(), "remote.json")
	os.WriteFile(path, data, 0600)
	provider, err := LoadProvider(path, true)
	if err != nil {
		t.Fatal(err)
	}
	remote := provider.(*Remote)
	defer remote.Close()
	o := Observation{Version: ObservationVersion, Identity: Identity{Life: 1, Map: "base1", Connection: 1, Spawncount: 42, Actor: 1, Frame: 10}, Health: 100}
	if _, err := remote.Decide(o); err == nil {
		t.Fatal("first request did not time out")
	}
	o.Identity.Frame++
	a, err := remote.Decide(o)
	if err != nil || a.Identity != o.Identity || a.Side != .2 {
		t.Fatal("reconnected request reused old response", a, err)
	}
	<-done
}

func TestFeedbackRequiresMatchingAcknowledgement(t *testing.T) {
	for _, name := range []string{"valid", "no_ack", "wrong_request"} {
		t.Run(name, func(t *testing.T) {
			listener, err := net.Listen("tcp4", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			done := make(chan struct{})
			go func() {
				defer close(done)
				conn, err := listener.Accept()
				if err != nil {
					return
				}
				defer conn.Close()
				var req BridgeRequest
				if ReadBridgeMessage(bufio.NewReader(conn), &req) != nil || req.Kind != "feedback" || len(req.Feedback) == 0 {
					return
				}
				reply := BridgeResponse{Version: BridgeVersion, Session: req.Session, Request: req.Request, PolicyVersion: req.PolicyVersion, FeedbackAck: true}
				if name == "no_ack" {
					reply.FeedbackAck = false
				}
				if name == "wrong_request" {
					reply.Request++
				}
				WriteBridgeMessage(conn, reply)
			}()
			config := RemoteConfig{Kind: RemoteKind, Address: listener.Addr().String(), TimeoutMS: 200, PolicyVersion: "test", Episode: "seed-9", Seed: 9}
			data, _ := json.Marshal(config)
			path := filepath.Join(t.TempDir(), "remote.json")
			os.WriteFile(path, data, 0600)
			provider, err := LoadProvider(path, true)
			if err != nil {
				t.Fatal(err)
			}
			remote := provider.(*Remote)
			defer remote.Close()
			err = remote.Notify(map[string]any{"version": "combat_feedback_v1", "kind": "test"})
			if name == "valid" && err != nil || name != "valid" && (err == nil || remote.conn != nil) {
				t.Fatal(name, err)
			}
			<-done
		})
	}
}
