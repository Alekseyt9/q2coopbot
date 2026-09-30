package bot

import (
	"encoding/binary"
	"net"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

// The server's reliable signon buffer must get feedback even when stufftext
// was seen already. Otherwise waiting for the next chunk can deadlock signon.
func TestHandshakeAcknowledgesRepeatedRequest(t *testing.T) {
	server, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	conn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	c := &Client{conn: conn, address: server.LocalAddr().(*net.UDPAddr), seq: 1, qport: 42, connected: true, decoder: quake.NewDecoder(), seenCommands: map[string]bool{"cmd configstrings 123 96": true}}
	packet := func(seq uint32, reliable bool) []byte {
		if reliable {
			seq |= 1 << 31
		}
		p := binary.LittleEndian.AppendUint32(nil, seq)
		p = binary.LittleEndian.AppendUint32(p, 0)
		return append(append(p, 11), append([]byte("cmd configstrings 123 96\n"), 0)...)
	}
	for _, tc := range []struct {
		seq      uint32
		reliable bool
		ack      uint32
	}{{6, false, 6}, {7, true, 7 | 1<<31}, {8, false, 8 | 1<<31}, {9, true, 9}} {
		c.handle(packet(tc.seq, tc.reliable))
		if err := server.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
			t.Fatal(err)
		}
		buf := make([]byte, 128)
		n, _, err := server.ReadFromUDP(buf)
		if err != nil {
			t.Fatal("signon did not acknowledge ignored request", err)
		}
		if n != 10 || binary.LittleEndian.Uint32(buf[:4])&(1<<31) != 0 || binary.LittleEndian.Uint32(buf[4:8]) != tc.ack || binary.LittleEndian.Uint16(buf[8:10]) != 42 {
			t.Fatalf("invalid bare acknowledgement: %x", buf[:n])
		}
	}
	// Old/duplicate packets must not toggle the reliable acknowledgement.
	previous := c.seq
	c.handle(packet(9, true))
	c.handle(packet(8, true))
	if c.seq != previous || c.serverReliable != 0 {
		t.Fatal("duplicate packets produced an acknowledgement or toggled reliable state")
	}
	// During gameplay acknowledgements travel with the usual movement packets.
	c.begun = true
	c.handle(packet(10, false))
	if c.seq != previous {
		t.Fatal("gameplay added a signon acknowledgement")
	}
}
