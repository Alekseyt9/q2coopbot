package bot

import (
	"encoding/json"
	"log"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
)

// Network spawncount identifies the current map generation. Server endpoint,
// map and monotonic frame are additional guards against stale rendezvous points.
type travelMemory struct {
	Version    int         `json:"version"`
	Server     string      `json:"server"`
	Map        string      `json:"map"`
	Generation int         `json:"generation"`
	Frame      int         `json:"frame"`
	Player     *quake.Vec3 `json:"player,omitempty"`
	Death      *quake.Vec3 `json:"death,omitempty"`
	Completed  bool        `json:"completed"`
}

func (m *travelMemory) matches(server string, generation int, s quake.Snapshot) bool {
	if m == nil || m.Version != 1 || m.Server != server || m.Map != s.Map || m.Generation != generation || m.Frame > s.Frame {
		return false
	}
	for _, point := range []*quake.Vec3{m.Player, m.Death} {
		if point != nil {
			for _, v := range *point {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 65536 {
					return false
				}
			}
		}
	}
	return true
}

func (c *Client) prepareTravelMemory(s quake.Snapshot) {
	if c.memoryFile == "" {
		return
	}
	server := c.address.String() + "|" + c.memorySession
	if !c.memoryLoaded {
		c.memoryLoaded = true
		var m travelMemory
		if data, err := os.ReadFile(c.memoryFile); err == nil && json.Unmarshal(data, &m) == nil && m.matches(server, c.spawncount, s) {
			c.travelMemory = &m
			c.planner.setMap(s.Map, c.root)
			c.planner.deathPoint = m.Death
			point := m.Player
			if point == nil {
				point = m.Death
			}
			if point != nil && !m.Completed && s.Teammate == nil {
				c.planner.respawnRegroup = &respawnRegroup{target: *point}
				log.Printf("restored rendezvous map=%s target=%v", s.Map, *point)
			}
		}
	}
	if !c.travelMemory.matches(server, c.spawncount, s) {
		c.travelMemory = &travelMemory{Version: 1, Server: server, Map: s.Map, Generation: c.spawncount}
		c.memoryWritten = 0
	}
}

func (c *Client) saveTravelMemory(s quake.Snapshot, returning bool) {
	if c.memoryFile == "" || c.travelMemory == nil {
		return
	}
	m := c.travelMemory
	completedBefore := m.Completed
	deathChanged := m.Death != c.planner.deathPoint
	m.Death = c.planner.deathPoint
	if deathChanged && m.Death != nil {
		m.Completed = false
	}
	if s.Teammate != nil {
		point := *s.Teammate
		m.Player = &point
		m.Completed = false
	}
	if returning && c.planner.respawnRegroup == nil && s.Teammate == nil && s.Health > 0 {
		m.Completed = true
	}
	if !deathChanged && completedBefore == m.Completed && c.memoryWritten > 0 && s.Frame-c.memoryWritten < 10 {
		return
	}
	m.Frame = s.Frame
	data, err := json.MarshalIndent(m, "", "  ")
	if err == nil {
		err = os.MkdirAll(filepath.Dir(c.memoryFile), 0755)
	}
	if err == nil {
		err = os.WriteFile(c.memoryFile+".tmp", data, 0600)
	}
	if err == nil {
		err = os.Rename(c.memoryFile+".tmp", c.memoryFile)
	}
	c.memoryWritten = s.Frame
	if err != nil {
		log.Printf("travel memory save: %v", err)
	}
}
