package bot

import (
	"bytes"
	"net"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestDeathFallbackAndMeetingPlayer(t *testing.T) {
	p := &Planner{}
	a := quake.Snapshot{Map: "base2", Frame: 10, Health: 100, Self: quake.Vec3{700, 20, 24}}
	d := a
	d.Frame++
	d.Health = 0
	p.observeRespawnRegroup(a, d)
	s := d
	s.Frame++
	s.Health = 100
	s.Self = quake.Vec3{}
	p.observeRespawnRegroup(d, s)
	if p.respawnRegroup == nil || p.respawnRegroup.target != a.Self {
		t.Fatal("death point lost")
	}
	mate := quake.Vec3{100, 0, 24}
	s.Teammate = &mate
	s.Frame++
	p.observeRespawnRegroup(d, s)
	if p.respawnRegroup != nil {
		t.Fatal("meeting player did not cancel return")
	}
}
func TestTravelMemoryRestartAndSessionIsolation(t *testing.T) {
	file := filepath.Join(t.TempDir(), "memory.json")
	newClient := func() *Client {
		return &Client{address: &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 29110}, memoryFile: file, memorySession: "server-a", spawncount: 123, planner: &Planner{}}
	}
	s := quake.Snapshot{Map: "base2", Frame: 100, Health: 100}
	death := quake.Vec3{700, 20, 24}
	c := newClient()
	c.prepareTravelMemory(s)
	c.planner.deathPoint = &death
	c.saveTravelMemory(s, false)
	r := newClient()
	s.Frame++
	r.prepareTravelMemory(s)
	if r.planner.respawnRegroup == nil || r.planner.respawnRegroup.target != death {
		t.Fatal("death fallback not restored")
	}
	player := quake.Vec3{900, 20, 24}
	s.Teammate = &player
	c.prepareTravelMemory(s)
	c.saveTravelMemory(s, false)
	s.Frame += 20
	c.saveTravelMemory(s, false)
	s.Teammate = nil
	r = newClient()
	r.prepareTravelMemory(s)
	if r.planner.respawnRegroup == nil || r.planner.respawnRegroup.target != player {
		t.Fatal("player priority not restored")
	}
	for _, kind := range []string{"server", "generation", "map", "rewind"} {
		r = newClient()
		v := s
		switch kind {
		case "server":
			r.memorySession = "server-b"
		case "generation":
			r.spawncount++
		case "map":
			v.Map = "base3"
		case "rewind":
			v.Frame = 1
		}
		r.prepareTravelMemory(v)
		if r.planner.respawnRegroup != nil {
			t.Fatalf("stale %s restored", kind)
		}
	}
}

func TestNewDeathReopensCompletedTravelMemory(t *testing.T) {
	file := filepath.Join(t.TempDir(), "memory.json")
	death := quake.Vec3{700, 20, 24}
	c := &Client{address: &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 29110}, memoryFile: file, memorySession: "server-a", spawncount: 123, planner: &Planner{}}
	s := quake.Snapshot{Map: "base2", Frame: 100, Health: 100}
	c.prepareTravelMemory(s)
	c.planner.deathPoint = &death
	c.saveTravelMemory(s, false)
	s.Frame++
	c.saveTravelMemory(s, true)
	if !c.travelMemory.Completed {
		t.Fatal("arrival not completed")
	}
	// A second death can happen at exactly the same coordinates.
	again := death
	c.planner.deathPoint = &again
	s.Frame++
	s.Health = 0
	c.saveTravelMemory(s, false)
	if c.travelMemory.Completed {
		t.Fatal("new death kept obsolete completed flag")
	}
	r := &Client{address: c.address, memoryFile: file, memorySession: c.memorySession, spawncount: 123, planner: &Planner{}}
	s.Frame++
	s.Health = 100
	r.prepareTravelMemory(s)
	if r.planner.respawnRegroup == nil || r.planner.respawnRegroup.target != death {
		t.Fatal("new death was not restored after restart")
	}
}

func TestReconnectReloadsTravelMemoryWithoutResumingStaleOrCompletedReturn(t *testing.T) {
	for _, kind := range []string{"active", "completed", "generation", "server", "rewind"} {
		t.Run(kind, func(t *testing.T) {
			server, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.ParseIP("127.0.0.1")})
			if err != nil {
				t.Fatal(err)
			}
			defer server.Close()
			conn, err := net.ListenUDP("udp4", nil)
			if err != nil {
				t.Fatal(err)
			}
			c := &Client{conn: conn, address: server.LocalAddr().(*net.UDPAddr), memoryFile: filepath.Join(t.TempDir(), "memory.json"), memorySession: "session-a", spawncount: 123, planner: &Planner{}}
			t.Cleanup(func() { c.conn.Close() })
			s := quake.Snapshot{Map: "base2", Frame: 100, Health: 100}
			c.prepareTravelMemory(s)
			death := quake.Vec3{194, 1940, -167.875}
			c.planner.World.Map = s.Map
			c.planner.deathPoint = &death
			if kind != "completed" {
				c.planner.respawnRegroup = &respawnRegroup{target: death}
			}
			c.saveTravelMemory(s, kind == "completed")
			if err := c.reconnect(); err != nil {
				t.Fatal(err)
			}
			before, err := os.ReadFile(c.memoryFile)
			if err != nil {
				t.Fatal(err)
			}
			// The harness briefly spawns both clients together before placement.
			// This must neither consume the reload nor replace the saved target.
			c.planner.testSetupHold = true
			placement := s
			placement.Frame += 100
			placement.Teammate = &quake.Vec3{876, 2232, -214}
			c.prepareTravelMemory(placement)
			c.saveTravelMemory(placement, false)
			after, err := os.ReadFile(c.memoryFile)
			if err != nil || !bytes.Equal(before, after) || c.memoryLoaded {
				t.Fatal("placement consumed or overwrote travel memory", err)
			}
			c.planner.testSetupHold = false
			s.Frame += 2
			switch kind {
			case "generation":
				c.spawncount++
			case "server":
				c.memorySession = "session-b"
			case "rewind":
				s.Frame = 1
			}
			c.prepareTravelMemory(s)
			active := c.planner.respawnRegroup != nil
			if active != (kind == "active") {
				t.Fatalf("wrong restored return for %s: %+v", kind, c.planner.respawnRegroup)
			}
			if kind == "active" && c.planner.respawnRegroup.target != death {
				t.Fatal("wrong restored death point")
			}
			c.saveTravelMemory(s, active)
			if kind == "active" || kind == "completed" {
				if c.travelMemory.Death == nil || *c.travelMemory.Death != death || c.travelMemory.Completed != (kind == "completed") {
					t.Fatal("memory erased or completed state changed")
				}
			} else if c.travelMemory.Death != nil {
				t.Fatal("stale death point survived invalid boundary")
			}
		})
	}
}
