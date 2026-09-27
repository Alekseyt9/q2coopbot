package quake

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBase2ElevatorFloorFixtures(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP/AAS")
	}
	g, err := LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	n, err := LoadAAS(filepath.Join(root, "maps/base2.aas"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"near", "far"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile("../../scripts/scenarios/base2-elevator-floor-" + name + ".json")
			if err != nil {
				t.Fatal(err)
			}
			var c struct {
				Bot   Vec3 `json:"bot_origin"`
				Actor Vec3 `json:"actor_origin"`
			}
			if err = json.Unmarshal(data, &c); err != nil {
				t.Fatal(err)
			}
			if c.Bot[0]-16 < -16 || !g.PlayerMoveClear(c.Bot, c.Bot) || n.ExactAreaFor(c.Bot) < 0 {
				t.Fatal("invalid exterior floor origin", c.Bot)
			}
			for _, offset := range []Vec3{{}, {-16, -16, 0}, {-16, 16, 0}, {16, -16, 0}, {16, 16, 0}} {
				p := c.Bot
				for i := range p {
					p[i] += offset[i]
				}
				d, ok := g.GroundDrop(p, 1)
				if !ok || d > 0.3 {
					t.Fatalf("missing static support at %v: %v %v", p, d, ok)
				}
			}
			route, ok := n.Route(c.Bot, c.Actor)
			found := false
			for _, w := range route {
				if w.Kind == 11 && w.Model == 50 {
					found = true
				}
			}
			if !ok || !found {
				t.Fatal("fixture route does not use platform50")
			}
		})
	}
}
