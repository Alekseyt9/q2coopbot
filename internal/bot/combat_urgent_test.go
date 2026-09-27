package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestUrgentRetreatChangesSpeedOnlyForLatchedThreat(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	friend := quake.Vec3{32, -352, 24}
	s := quake.Snapshot{Frame: 40, OnGround: true, Self: quake.Vec3{122, -224, 24.125}, Teammate: &friend, Weapon: "Blaster", Enemies: []quake.Object{{ID: 57, Class: "monster_parasite", Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}}}
	p := &Planner{World: World{Snapshot: s, Geometry: &g}, urgentRetreat: urgentRetreat{57, 70}}
	for _, target := range []int{57, 58, 0} {
		p.urgentRetreat.target = target
		p.World.Command = CommandDecision{AimSource: "enemy"}
		cmd := p.combatRetreat(quake.UserCmd{Buttons: 1}, combatSpacing(s))
		want := int16(-80)
		if target == 57 {
			want = -160
		}
		if cmd.Forward != want || cmd.Side != 0 || cmd.Buttons != 1 {
			t.Fatalf("target=%d command=%+v", target, cmd)
		}
	}
	p.urgentRetreat.target = 57
	p.World.Geometry = nil
	if cmd := p.combatRetreat(quake.UserCmd{}, combatSpacing(s)); cmd.Forward != 0 || cmd.Side != 0 {
		t.Fatal("urgent escape bypassed geometry")
	}
}

func TestUrgentRetreatRequiresObservedFastApproach(t *testing.T) {
	clear := true
	old := quake.Snapshot{Map: "base1", Frame: 10, Health: 100, OnGround: true, Self: quake.Vec3{100, 0, 24}}
	now := old
	now.Frame++
	now.Self[0] = 130
	now.Enemies = []quake.Object{{ID: 57, Class: "monster_parasite", Origin: quake.Vec3{200, 0, 24}, ClearShot: &clear}}
	p := &Planner{World: World{Snapshot: old}}
	p.observeUrgentRetreat(now)
	if p.urgentRetreat.target != 57 {
		t.Fatal("fast approach did not latch escape")
	}
	for _, kind := range []string{"still", "away", "slow", "teleport", "gap", "map", "dead", "air", "hidden", "far"} {
		s := now
		s.Enemies = append([]quake.Object(nil), now.Enemies...)
		switch kind {
		case "still":
			s.Self = old.Self
		case "away":
			s.Self[0] = 70
		case "slow":
			s.Self[0] = 110
		case "teleport":
			s.Self[0] = 160
		case "gap":
			s.Frame++
		case "map":
			s.Map = "base2"
		case "dead":
			s.Health = 0
		case "air":
			s.OnGround = false
		case "hidden":
			s.Enemies[0].ClearShot = nil
		case "far":
			s.Enemies[0].Origin[0] = 400
		}
		p.urgentRetreat = urgentRetreat{}
		p.observeUrgentRetreat(s)
		if p.urgentRetreat.target != 0 {
			t.Fatalf("unexpected urgency: %s", kind)
		}
	}
}

func TestUrgentRetreatExpiresOrClearsAfterEscape(t *testing.T) {
	clear := true
	old := quake.Snapshot{Map: "base1", Frame: 20, Health: 100, OnGround: true, Self: quake.Vec3{100, 0, 24}}
	now := old
	now.Frame++
	now.Self[0] = 84
	now.Enemies = []quake.Object{{ID: 57, Class: "monster_parasite", Origin: quake.Vec3{200, 0, 24}, ClearShot: &clear}}
	p := &Planner{World: World{Snapshot: old}, urgentRetreat: urgentRetreat{57, 50}}
	p.observeUrgentRetreat(now)
	if p.urgentRetreat.target != 57 {
		t.Fatal("escape stopped immediately on retreat")
	}
	for _, kind := range []string{"expired", "escaped", "lost", "gap", "dead"} {
		s := now
		p.urgentRetreat = urgentRetreat{57, 50}
		switch kind {
		case "expired":
			p.urgentRetreat.until = 20
		case "escaped":
			s.Self[0] = -100
		case "lost":
			s.Enemies = nil
		case "gap":
			s.Frame++
		case "dead":
			s.Health = 0
		}
		p.observeUrgentRetreat(s)
		if p.urgentRetreat.target != 0 {
			t.Fatalf("stale urgency: %s", kind)
		}
	}
}
