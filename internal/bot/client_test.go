package bot

import (
	"bytes"
	"net"
	"testing"
	"time"

	"q2coopbot/internal/quake"
)

func TestClientSafetyStopWhenFramesStop(t *testing.T) {
	now := time.Now()
	c := &Client{begun: true, framePaced: true, frameReady: true, latestFrame: 7, lastMoveFrame: 7,
		previous: quake.UserCmd{Forward: 400, Buttons: 1}, planner: &Planner{World: World{Updated: now}}}
	if c.needsSafetyStop(now.Add(300 * time.Millisecond)) {
		t.Fatal("fresh observation triggered safety stop")
	}
	if !c.needsSafetyStop(now.Add(301 * time.Millisecond)) {
		t.Fatal("stalled frame did not trigger safety stop")
	}
	c.previous = quake.UserCmd{}
	if c.needsSafetyStop(now.Add(time.Second)) {
		t.Fatal("neutral command caused repeated safety stop")
	}
	c.previous = quake.UserCmd{Forward: 400}
	c.latestFrame++
	if c.needsSafetyStop(now.Add(time.Second)) {
		t.Fatal("new frame should use normal command path")
	}
}

func TestReconnectUsesFreshUDPChannel(t *testing.T) {
	server, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	old, err := net.ListenUDP("udp4", nil)
	if err != nil {
		t.Fatal(err)
	}
	c := &Client{conn: old, address: server.LocalAddr().(*net.UDPAddr), qport: 42, seq: 20, serverSeq: 10, connected: true, begun: true, decoder: quake.NewDecoder(), planner: &Planner{}}
	if err := c.reconnect(); err != nil {
		t.Fatal(err)
	}
	defer c.conn.Close()
	if c.conn == old || c.seq != 1 || c.serverSeq != 0 || c.connected || c.begun {
		t.Fatalf("reconnect retained stale channel state: seq=%d serverSeq=%d connected=%t begun=%t", c.seq, c.serverSeq, c.connected, c.begun)
	}
	if err := server.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 256)
	for i := 0; i < 2; i++ {
		n, from, err := server.ReadFromUDP(buf)
		if err != nil {
			t.Fatal(err)
		}
		if n >= 4 && bytes.Equal(buf[:4], []byte{255, 255, 255, 255}) {
			if string(buf[4:n]) != "getchallenge\n" || from.Port != c.conn.LocalAddr().(*net.UDPAddr).Port {
				t.Fatalf("reconnect challenge came from wrong channel: %q, %v", buf[4:n], from)
			}
			return
		}
	}
	t.Fatal("reconnect did not request a new challenge")
}
