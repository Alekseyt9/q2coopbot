package bot

import (
	"context"
	"q2coopbot/internal/quake"
	"testing"
)

func TestIsolatedCombatRetainsThreatWithoutExitRoute(t *testing.T) {
	clear := true
	p := &Planner{Campaign: true, testCombatOnly: true, World: World{Map: "base2", Navigation: "unreachable"}}
	s := quake.Snapshot{Map: "base2", Frame: 20, Health: 100, Weapon: "Blaster", OnGround: true, Enemies: []quake.Object{{ID: 7, Class: "monster_infantry", Origin: quake.Vec3{100, 0, 24}, ClearShot: &clear}}, Pickups: []quake.Object{{ID: 8, Class: "item_health", Origin: quake.Vec3{0, 40, 24}}}}
	p.update(s, "")
	if len(p.World.Snapshot.Enemies) != 1 || len(p.World.Snapshot.Pickups) != 1 || len(p.World.Route) != 0 || p.hasGoal {
		t.Fatal("combat reset lost observations or started campaign travel")
	}
	if err := Run(context.Background(), Config{TestCombatOnly: true}); err == nil {
		t.Fatal("combat fixture allowed in normal session")
	}
}
