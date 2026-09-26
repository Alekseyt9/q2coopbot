package bot

import (
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
		for _, offset := range []quake.Vec3{{48, 0, 0}, {-48, 0, 0}, {0, 48, 0}, {0, -48, 0}} {
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
			if !ok || visibility.NewlyVisible == 0 {
				continue
			}
			for _, hidden := range p.searchVisibilitySamples(s) {
				visible, known := m.PointPVS(searchEye(self), searchEye(hidden))
				viewVisible, viewKnown := m.PointPVS(searchEye(view), searchEye(hidden))
				if !known || visible || !viewKnown || !viewVisible {
					continue
				}
				if !m.ClearShot(searchEye(view), searchEye(hidden)) || !m.PlayerMoveClear(last, hidden) {
					continue
				}
				t.Logf("area=%d last=%v hidden=%v view=%v gain=%+v self=%v", i, last, hidden, view, visibility, self)
				count++
				break
			}
		}
		if count >= 10 {
			break
		}
	}
	t.Logf("candidates=%d (requires UDP verification)", count)
}
