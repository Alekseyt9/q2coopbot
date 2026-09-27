package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

// Opt-in scene authoring aid; geometry candidates still require UDP validation.
func TestProjectileFixtureScan(t *testing.T) {
	if os.Getenv("Q2_PROJECTILE_SCAN") == "" {
		t.Skip("authoring scan")
	}
	g, err := quake.LoadMap(os.Getenv("Q2_SEARCH_SCAN_ROOT"), "base1")
	if err != nil {
		t.Fatal(err)
	}
	target := quake.Vec3{1088, 328, -31.875}
	count := 0
	for x := 576.0; x <= 1632; x += 32 {
		for y := -64.0; y <= 640; y += 32 {
			p := quake.Vec3{x, y, -31.875}
			d := quake.Distance(p, target)
			if d < 350 || d > 640 || !g.PlayerMoveClear(p, p) {
				continue
			}
			drop, ok := g.GroundDrop(p, 2)
			if !ok || drop > 2 {
				continue
			}
			from, to := p, target
			from[2] += 22
			to[2] += 22
			if !g.ClearShot(from, to) {
				continue
			}
			t.Logf("candidate=%v distance=%.1f", p, d)
			count++
		}
	}
	t.Logf("candidates=%d", count)
}
