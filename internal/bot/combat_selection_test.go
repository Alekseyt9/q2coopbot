package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestCoveredNearestEnemyDoesNotHideVisibleTarget(t *testing.T) {
	clear, blocked := true, false
	near := quake.Object{ID: 1, Origin: quake.Vec3{60, 0, 0}, ClearShot: &blocked}
	far := quake.Object{ID: 2, Origin: quake.Vec3{0, 200, 0}, ClearShot: &clear}
	for _, enemies := range [][]quake.Object{{near, far}, {far, near}} {
		p := &Planner{World: World{Map: "test", Updated: time.Now(), Goal: "cover_teammate",
			Snapshot: quake.Snapshot{Frame: 1, Health: 100, Weapon: "Blaster", Enemies: enemies}}}
		cmd := p.commandAt(quake.UserCmd{}, p.World.Updated)
		if cmd.Buttons&1 == 0 || cmd.Yaw != 16384 {
			t.Fatalf("visible target ignored: %+v", cmd)
		}
		p.World.Snapshot.Enemies = []quake.Object{near}
		if cmd := p.commandAt(cmd, p.World.Updated); cmd.Buttons != 0 {
			t.Fatal("attack persisted after visible target disappeared")
		}
	}
}
