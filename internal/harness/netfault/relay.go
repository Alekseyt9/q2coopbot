// Package netfault injects packet loss outside the bot's decision loop.
package netfault

import (
	"context"
	"encoding/binary"
	"fmt"
	"net"
	"net/netip"
	"time"
)

type Config struct {
	Listen     string `json:"listen"`
	Server     string `json:"server"`
	AfterMS    int    `json:"after_ms"`
	DurationMS int    `json:"duration_ms"`
}

type Event struct {
	ElapsedMS int64   `json:"elapsed_ms"`
	Direction string  `json:"direction"`
	Action    string  `json:"action"`
	Bytes     int     `json:"bytes"`
	Sequence  *uint32 `json:"sequence,omitempty"`
	Stage     string  `json:"stage"`
}

// AfterMS starts at the first server datagram, not process startup or a game
// frame. Duration is wall time and does not scale with server timescale.
func (c Config) Validate() error {
	var endpoints []netip.AddrPort
	for _, address := range []string{c.Listen, c.Server} {
		a, err := netip.ParseAddrPort(address)
		if err != nil || !a.Addr().Is4() || !a.Addr().IsLoopback() || a.Port() < 1 {
			return fmt.Errorf("expected explicit IPv4 loopback endpoint: %q", address)
		}
		endpoints = append(endpoints, a)
	}
	if c.AfterMS < 0 || c.AfterMS > 120000 || c.DurationMS < 1 || c.DurationMS > 120000 {
		return fmt.Errorf("invalid blackout interval")
	}
	if endpoints[0] == endpoints[1] {
		return fmt.Errorf("relay cannot target itself")
	}
	return nil
}

func stage(elapsed time.Duration, c Config) string {
	if elapsed < time.Duration(c.AfterMS)*time.Millisecond {
		return "before"
	}
	if elapsed < time.Duration(c.AfterMS+c.DurationMS)*time.Millisecond {
		return "blackout"
	}
	return "after"
}

// Serve owns both sockets and supports one pinned client endpoint. A different
// endpoint is rejected explicitly; reconnect migration requires a separate
// fixture contract. No payloads (including RCON strings) are written to events.
func Serve(ctx context.Context, c Config, emit func(Event) error, ready func(string)) error {
	if err := c.Validate(); err != nil {
		return err
	}
	listen, _ := net.ResolveUDPAddr("udp4", c.Listen)
	server, _ := net.ResolveUDPAddr("udp4", c.Server)
	front, err := net.ListenUDP("udp4", listen)
	if err != nil {
		return err
	}
	defer front.Close()
	back, err := net.DialUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)}, server)
	if err != nil {
		return err
	}
	defer back.Close()
	type packet struct {
		data      []byte
		from      *net.UDPAddr
		direction string
		err       error
	}
	packets := make(chan packet, 128)
	readerCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	read := func(conn *net.UDPConn, direction string) {
		for {
			buf := make([]byte, 65535)
			n, from, err := conn.ReadFromUDP(buf)
			select {
			case packets <- packet{buf[:n], from, direction, err}:
			case <-readerCtx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}
	go read(front, "client_to_server")
	go read(back, "server_to_client")
	if ready != nil {
		ready(front.LocalAddr().String())
	}
	var peer *net.UDPAddr
	var started time.Time
	for {
		select {
		case <-ctx.Done():
			return nil
		case p := <-packets:
			if p.err != nil {
				return p.err
			}
			if p.direction == "client_to_server" {
				if peer == nil {
					peer = p.from
				}
				if peer.String() != p.from.String() {
					return fmt.Errorf("relay client endpoint changed")
				}
			} else if started.IsZero() {
				started = time.Now()
			}
			e := Event{Direction: p.direction, Bytes: len(p.data), Action: "forward", Stage: "unarmed"}
			if !started.IsZero() {
				elapsed := time.Since(started)
				e.ElapsedMS = elapsed.Milliseconds()
				e.Stage = stage(elapsed, c)
			}
			if len(p.data) >= 4 {
				v := binary.LittleEndian.Uint32(p.data[:4])
				if v != 0xffffffff {
					v &= 0x7fffffff
					e.Sequence = &v
				}
			}
			if e.Stage == "blackout" {
				e.Action = "drop"
			} else if p.direction == "client_to_server" {
				_, err = back.Write(p.data)
			} else if peer != nil {
				_, err = front.WriteToUDP(p.data, peer)
			} else {
				return fmt.Errorf("server packet without client")
			}
			if err != nil {
				return err
			}
			if err = emit(e); err != nil {
				return err
			}
		}
	}
}
