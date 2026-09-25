package bot

import (
	"bytes"
	"net"
	"testing"
	"time"

	"q2coopbot/internal/quake"
)

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
