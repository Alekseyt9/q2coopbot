package bot

import (
	"context"
	"os"
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

func TestIsolatedCombatKeepsShortRetreatWithoutExitRoute(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	clear := true
	p := &Planner{Campaign: true, testCombatOnly: true}
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 100, Weapon: "Blaster", OnGround: true,
		Self: quake.Vec3{32, -224, 24}, Enemies: []quake.Object{{ID: 7, Class: "monster_parasite", Origin: quake.Vec3{200, -224, 24}, Solid: 7266, ClearShot: &clear}}}
	p.update(s, root)
	cmd := p.command(quake.UserCmd{})
	if p.World.Goal != "combat_fixture" || p.hasGoal || len(p.World.Route) != 0 {
		t.Fatal("isolated fight acquired exit route")
	}
	if cmd.Forward == 0 && cmd.Side == 0 || p.World.Command.MoveSource != "combat_retreat" {
		t.Fatalf("isolated objective disabled observed short retreat: command=%+v decision=%+v", cmd, p.World.Command)
	}
}
