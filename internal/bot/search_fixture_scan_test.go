package bot

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

// Opt-in authoring aid: geometric candidates are not live visibility proof.
func TestSearchFixtureScan(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("set Q2_SEARCH_SCAN_ROOT to a baseq2 asset directory")
	}
	name := os.Getenv("Q2_SEARCH_SCAN_MAP")
	if name == "" {
		name = "base1"
	}
	m, err := quake.LoadMap(root, name)
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(filepath.Join(root, "maps", name+".aas"))
	if err != nil {
		t.Fatal(err)
	}
	p := &Planner{Nav: nav, World: World{Geometry: &m}}
	if trace := os.Getenv("Q2_SEARCH_SCAN_TRACE"); trace != "" {
		scanRecordedWalk(t, p, trace, name)
		return
	}
	count := 0
	for i, area := range nav.Areas {
		last := area.Center
		last[2] = area.Min[2] + 1
		if area.Flags&1 == 0 || !m.PlayerMoveClear(last, last) {
			continue
		}
		if _, ok := m.GroundDrop(last, 24); !ok {
			continue
		}
		for _, offset := range []quake.Vec3{{63, 0, 0}, {-63, 0, 0}, {0, 63, 0}, {0, -63, 0}, {44, 44, 0}, {-44, 44, 0}, {44, -44, 0}, {-44, -44, 0}} {
			self := last
			self[0] += offset[0]
			self[1] += offset[1]
			if !m.PlayerMoveClear(self, last) {
				continue
			}
			if _, ok := m.GroundDrop(self, 24); !ok {
				continue
			}
			s := quake.Snapshot{Self: self, LastTeammate: &last}
			view, visibility, ok := p.selectSearchViewpoint(s)
			if !ok {
				continue
			}
			for _, targetArea := range nav.Areas {
				hidden := targetArea.Center
				hidden[2] = targetArea.Min[2] + 1
				if targetArea.Flags&1 == 0 || quake.Horizontal(last, hidden) > 512 || math.Abs(last[2]-hidden[2]) > 48 {
					continue
				}
				visible, known := m.PointPVS(searchEye(self), searchEye(hidden))
				viewVisible, viewKnown := m.PointPVS(searchEye(view), searchEye(hidden))
				if !known || visible || !viewKnown || !viewVisible {
					continue
				}
				route, routeOK := nav.SearchRoute(last, hidden)
				if !routeOK {
					continue
				}
				if _, safe := safeSearchRoute(route, last, hidden, 640); !safe {
					continue
				}
				t.Logf("area=%d last=%v hidden=%v view=%v gain=%+v self=%v", i, last, hidden, view, visibility, self)
				count++
				break
			}
			if count >= 10 {
				break
			}
		}
		if count >= 10 {
			break
		}
	}
	t.Logf("candidates=%d (requires UDP verification)", count)
}

// Replay only geometry for candidate selection. Physics and server fat PVS
// still have to be verified by the paired UDP fixture.
func scanRecordedWalk(t *testing.T, p *Planner, path, name string) {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var walk []quake.Vec3
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		var row struct {
			Map         string     `json:"map"`
			Self        quake.Vec3 `json:"self"`
			Arbitration struct {
				MoveSource string `json:"move_source"`
			} `json:"arbitration"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		if row.Map == name && row.Arbitration.MoveSource == "test_walk" {
			walk = append(walk, row.Self)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if len(walk) < 2 {
		t.Fatal("trace has no walking path")
	}
	m := p.World.Geometry
	target := walk[len(walk)-1]
	count := 0
	for _, area := range p.Nav.Areas[1:] {
		origin := area.Center
		origin[2] = area.Min[2] + 1
		if area.Flags&1 == 0 || quake.Horizontal(origin, walk[0]) > 400 || math.Abs(origin[2]-target[2]) > 48 || !m.PlayerMoveClear(origin, origin) {
			continue
		}
		if _, ok := m.GroundDrop(origin, 24); !ok {
			continue
		}
		visible, known := m.PointPVS(searchEye(origin), searchEye(walk[0]))
		if !known || !visible {
			continue
		}
		visible, known = m.PointPVS(searchEye(origin), searchEye(target))
		if !known || visible {
			continue
		}
		last := walk[0]
		for _, point := range walk {
			if visible, known := m.PointPVS(searchEye(origin), searchEye(point)); known && visible {
				last = point
			}
		}
		if quake.Horizontal(origin, last) <= 64 {
			continue
		}
		route, ok := p.Nav.SearchRoute(origin, last)
		if !ok {
			continue
		}
		if _, safe := safeSearchRoute(route, origin, last, 640); !safe {
			continue
		}
		at := origin
		next := 0
		reacquired := false
		for step := 0; step < 30 && quake.Horizontal(at, last) > 64; step++ {
			for next < len(route) && quake.Horizontal(at, route[next].Position) <= 24 {
				next++
			}
			goal := last
			if next < len(route) {
				goal = route[next].Position
			}
			d := quake.Horizontal(at, goal)
			if d < 0.01 {
				break
			}
			fraction := math.Min(30, d) / d
			for axis := range at {
				at[axis] += (goal[axis] - at[axis]) * fraction
			}
			if visible, known := m.PointPVS(searchEye(at), searchEye(target)); !known || visible {
				reacquired = true
				break
			}
		}
		if reacquired || quake.Horizontal(at, last) > 64 {
			continue
		}
		p.lastSeenSelf, p.lastSeenSelfKnown = origin, true
		view, gain, ok := p.selectSearchViewpoint(quake.Snapshot{Self: at, LastTeammate: &last})
		if !ok {
			continue
		}
		if visible, known := m.PointPVS(searchEye(view), searchEye(target)); !known || !visible {
			continue
		}
		t.Logf("recorded candidate bot=%v last=%v approach_end=%v view=%v gain=%+v", origin, last, at, view, gain)
		count++
		if count >= 10 {
			break
		}
	}
	t.Logf("recorded candidates=%d; geometric approximation only", count)
}
