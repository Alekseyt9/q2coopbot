package netfault

import (
	"bytes"
	"context"
	"net"
	"testing"
	"time"
)

func TestScheduleBoundaries(t *testing.T) {
	c := Config{AfterMS: 20, DurationMS: 50}
	for ms, want := range map[int]string{19: "before", 20: "blackout", 69: "blackout", 70: "after"} {
		if got := stage(time.Duration(ms)*time.Millisecond, c); got != want {
			t.Fatalf("%d: %s", ms, got)
		}
	}
	for _, c := range []Config{{Listen: "0.0.0.0:1234", Server: "127.0.0.1:2345", DurationMS: 1}, {Listen: "127.0.0.1:1234", Server: "8.8.8.8:1234", DurationMS: 1}, {Listen: "127.0.0.1:1234", Server: "127.0.0.1:1234", DurationMS: 1}} {
		if c.Validate() == nil {
			t.Fatal("accepted invalid config", c)
		}
	}
}

func socket(t *testing.T) *net.UDPConn {
	t.Helper()
	c, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func TestRealUDPBlackoutAndRecovery(t *testing.T) {
	server := socket(t)
	reservation := socket(t)
	address := reservation.LocalAddr().String()
	reservation.Close()
	client := socket(t)
	relayAddr, _ := net.ResolveUDPAddr("udp4", address)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	done := make(chan error, 1)
	ready := make(chan struct{})
	var events []Event
	go func() {
		done <- Serve(ctx, Config{Listen: address, Server: server.LocalAddr().String(), AfterMS: 30, DurationMS: 80}, func(e Event) error { events = append(events, e); return nil }, func(string) { close(ready) })
	}()
	select {
	case <-ready:
	case err := <-done:
		t.Fatal(err)
	case <-ctx.Done():
		t.Fatal("no readiness")
	}
	payload := []byte{1, 0, 0, 0, 0xff, 0, 13, 10, 99}
	if _, err := client.WriteToUDP(payload, relayAddr); err != nil {
		t.Fatal(err)
	}
	server.SetReadDeadline(time.Now().Add(time.Second))
	buf := make([]byte, 100)
	n, peer, err := server.ReadFromUDP(buf)
	if err != nil || !bytes.Equal(buf[:n], payload) {
		t.Fatalf("forward: %x %v", buf[:n], err)
	}
	// Server packets continue even when all client packets are dropped.
	until := time.Now().Add(220 * time.Millisecond)
	for time.Now().Before(until) {
		server.WriteToUDP(payload, peer)
		client.WriteToUDP(payload, relayAddr)
		time.Sleep(5 * time.Millisecond)
	}
	client.SetReadDeadline(time.Now().Add(time.Second))
	n, _, err = client.ReadFromUDP(buf)
	if err != nil || !bytes.Equal(buf[:n], payload) {
		t.Fatalf("reply: %x %v", buf[:n], err)
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, e := range events {
		counts[e.Direction+"/"+e.Stage+"/"+e.Action]++
		if e.Bytes != len(payload) {
			t.Fatal(e)
		}
	}
	for _, direction := range []string{"client_to_server", "server_to_client"} {
		for _, suffix := range []string{"before/forward", "blackout/drop", "after/forward"} {
			if counts[direction+"/"+suffix] == 0 {
				t.Fatalf("missing %s/%s: %v", direction, suffix, counts)
			}
		}
	}
	if r := Analyze(Config{Listen: address, Server: server.LocalAddr().String(), AfterMS: 30, DurationMS: 80}, events, 500, true); !r.Accepted {
		t.Fatal(r)
	}
}
