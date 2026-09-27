package bot

import (
	"encoding/json"
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestParasiteGroupFixtureStartsInOpenGeometry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../../scripts/scenarios/base1-parasite-group.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Bot    quake.Vec3 `json:"bot_origin"`
		Actor  quake.Vec3 `json:"actor_origin"`
		Target quake.Vec3 `json:"target_origin"`
		Second quake.Vec3 `json:"second_target_origin"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, point := range []quake.Vec3{fixture.Bot, fixture.Actor, fixture.Target, fixture.Second} {
		end := point
		end[0] += 0.125
		if !g.PlayerMoveClear(point, end) {
			t.Fatalf("fixture inside BSP: %v", point)
		}
		if _, ok := g.GroundDrop(point, 24); !ok {
			t.Fatalf("fixture lacks ground: %v", point)
		}
	}
	s := quake.Snapshot{Self: fixture.Bot}
	for _, point := range []quake.Vec3{fixture.Target, fixture.Second} {
		e := quake.Object{Origin: point, Solid: 7266}
		if !g.ClearShot(s.EyePoint(), e.AimPoint()) {
			t.Fatalf("fixture threat hidden by BSP: %v", point)
		}
	}
	if quake.Horizontal(fixture.Target, fixture.Second) < 48 {
		t.Fatal("overlapping targets")
	}
}

func TestCombatSpacingUsesThreatAndWeapon(t *testing.T) {
	clear := true
	s := quake.Snapshot{Weapon: "models/weapons/v_shotg/tris.md2", Enemies: []quake.Object{{ID: 1, Class: "monster_soldier", Origin: quake.Vec3{90, 0, 0}, ClearShot: &clear}, {ID: 2, Class: "monster_parasite", Origin: quake.Vec3{200, 0, 0}, ClearShot: &clear}}}
	p := combatSpacing(s)
	if p.Target != 2 || p.Minimum <= 256 || !p.NeedSpace || p.VisibleThreats != 2 {
		t.Fatalf("closer weak target masked melee threat: %+v", p)
	}
	s.Enemies = s.Enemies[:1]
	p = combatSpacing(s)
	if p.NeedSpace || p.PreferredMax != 192 {
		t.Fatal("shotgun range profile lost")
	}
	s.Weapon = "models/weapons/v_rail/tris.md2"
	p = combatSpacing(s)
	if !p.NeedSpace || p.PreferredMax != 900 {
		t.Fatal("rail spacing ignored")
	}
	s.Enemies[0].ClearShot = nil
	if combatSpacing(s) != nil {
		t.Fatal("unconfirmed line counted as actionable threat")
	}
}

func TestCombatRetreatPreservesAimAndRejectsGroupTrap(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	friend := quake.Vec3{32, -100, 24}
	s := quake.Snapshot{Self: quake.Vec3{32, -224, 24}, OnGround: true, Teammate: &friend, Weapon: "Blaster", Enemies: []quake.Object{{ID: 1, Class: "monster_soldier", Origin: quake.Vec3{96, -200, 24}, ClearShot: &clear}}}
	p := &Planner{World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimSource: "enemy"}}}
	cmd := quake.UserCmd{Buttons: 1, Pitch: 123, Yaw: 0}
	got := p.combatRetreat(cmd, combatSpacing(s))
	if got.Forward >= 0 || got.Buttons != 1 || got.Pitch != 123 || got.Yaw != 0 || p.World.Command.MoveSource != "combat_retreat" {
		t.Fatalf("safe retreat or aim missing: %+v %+v", got, p.World.Command)
	}
	s.Enemies = append(s.Enemies, quake.Object{ID: 2, Class: "monster_soldier", Origin: quake.Vec3{-32, -248, 24}, ClearShot: &clear},
		quake.Object{ID: 3, Class: "monster_soldier", Origin: quake.Vec3{32, -300, 24}, ClearShot: &clear},
		quake.Object{ID: 4, Class: "monster_soldier", Origin: quake.Vec3{32, -148, 24}, ClearShot: &clear})
	p.World.Snapshot = s
	p.World.Command = CommandDecision{AimSource: "enemy"}
	got = p.combatRetreat(cmd, combatSpacing(s))
	if got.Forward != 0 || got.Side != 0 || p.World.Command.MoveLimitReason != "combat_retreat_blocked" {
		t.Fatal("retreat approached another visible enemy")
	}
	s.Enemies = s.Enemies[:1]
	s.OnGround = false
	p.World.Snapshot = s
	if got := p.combatRetreat(cmd, combatSpacing(s)); got.Forward != 0 || got.Side != 0 {
		t.Fatal("airborne retreat accepted")
	}
}

func TestSystem1OffersRetreatOnlyForCurrentCloseThreat(t *testing.T) {
	clear := true
	friend := quake.Vec3{0, 100, 0}
	w := World{Goal: "cover_teammate", Snapshot: quake.Snapshot{OnGround: true, Weapon: "Blaster", Teammate: &friend, Enemies: []quake.Object{{ID: 1, Class: "monster_parasite", Origin: quake.Vec3{200, 0, 0}, ClearShot: &clear}}}}
	tactician := NewTactician("")
	has := func() bool {
		for _, o := range tactician.options(w) {
			if o == "retreat" {
				return true
			}
		}
		return false
	}
	if !has() {
		t.Fatal("System1 cannot choose spacing")
	}
	friend[1] = 300
	if !has() {
		t.Fatal("spacing leash stops retreat within attack reach")
	}
	friend[1] = 400
	if has() {
		t.Fatal("spacing leash is unbounded")
	}
	friend[1] = 100
	w.Snapshot.Enemies[0].Origin[0] = 500
	if has() {
		t.Fatal("stale retreat remains offered")
	}
}

func TestCombatLeashUsesBoundedProfileDistance(t *testing.T) {
	for _, tc := range []struct{ minimum, want float64 }{{96, 220}, {320, 352}, {384, 384}, {900, 384}} {
		if got := combatLeash(&CombatSpacing{Minimum: tc.minimum}); got != tc.want {
			t.Fatalf("minimum=%v leash=%v", tc.minimum, got)
		}
	}
}

func TestCombatSpacingHoldDoesNotResumeFollowIntoEnemy(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	friend := quake.Vec3{32, -100, 24}
	now := time.Now()
	s := quake.Snapshot{Map: "base1", Frame: 10, Health: 100, Self: quake.Vec3{32, -224, 24}, OnGround: true, Teammate: &friend, Weapon: "Blaster", Enemies: []quake.Object{{ID: 1, Class: "monster_soldier", Origin: quake.Vec3{132, -224, 24}, ClearShot: &clear}}}
	p := &Planner{World: World{Map: "base1", Snapshot: s, Geometry: &g, GeometryStatus: "ready", Goal: "follow_teammate", Updated: now}}
	cmd := p.commandAt(quake.UserCmd{}, now)
	if cmd.Forward != 0 || cmd.Side != 0 || cmd.Buttons != 1 || p.World.Command.MoveLimitReason != "combat_spacing_hold" {
		t.Fatalf("spacing hold lost: %+v %+v", cmd, p.World.Command)
	}
}

func TestCombatRetreatSlidesAlongWallWithoutUnknownGeometry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	friend := quake.Vec3{32, -352, 24.125}
	s := quake.Snapshot{Self: quake.Vec3{-39.375, -224, 24.125}, OnGround: true, Teammate: &friend, Weapon: "Blaster", Enemies: []quake.Object{{ID: 1, Class: "monster_parasite", Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}}}
	p := &Planner{World: World{Geometry: &g, Snapshot: s, Command: CommandDecision{AimSource: "enemy"}}}
	cmd := quake.UserCmd{Buttons: 1}
	got := p.combatRetreat(cmd, combatSpacing(s))
	if got.Forward != 0 || got.Side <= 0 || got.Buttons != 1 {
		t.Fatalf("did not slide along wall: %+v", got)
	}
	p.World.Geometry = nil
	if got = p.combatRetreat(cmd, combatSpacing(s)); got.Forward != 0 || got.Side != 0 {
		t.Fatal("retreat without verified geometry")
	}
}

func TestParasiteFiringPositionRevalidatesObservedThreats(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	friend := quake.Vec3{32, -352, 24.125}
	s := quake.Snapshot{Self: quake.Vec3{-39.125, -428.375, 24.125}, OnGround: true, Teammate: &friend, Weapon: "Blaster", Enemies: []quake.Object{{ID: 57, Class: "monster_parasite", Origin: quake.Vec3{200, -224, 24}, Solid: 7266, ClearShot: &clear}}}
	p := &Planner{World: World{Geometry: &g, Snapshot: s}}
	reset := func() { p.World.Command = CommandDecision{AimEntity: 57, LimitReason: "friendly_line_of_fire"} }
	reset()
	got := p.combatRetreat(quake.UserCmd{}, combatSpacing(s))
	if p.World.Command.MoveSource != "combat_firing_position" || got.Forward == 0 && got.Side == 0 || got.Buttons != 0 {
		t.Fatalf("no protected reposition: %+v %+v", got, p.World.Command)
	}
	// A nearby threat on the destination side invalidates the local detour,
	// even when its line of sight is unknown.
	s.Enemies = append(s.Enemies, quake.Object{ID: 58, Origin: quake.Vec3{0, -428, 24}})
	p.World.Snapshot = s
	reset()
	p.combatRetreat(quake.UserCmd{}, combatSpacing(s))
	if p.World.Command.MoveSource == "combat_firing_position" {
		t.Fatal("approached another threat")
	}
	s.Enemies = s.Enemies[:1]
	s.Enemies[0].Origin = quake.Vec3{180, -280, 24}
	p.World.Snapshot = s
	reset()
	p.combatRetreat(quake.UserCmd{}, combatSpacing(s))
	if p.World.Command.MoveSource == "combat_firing_position" {
		t.Fatal("traded attack reach for firing lane")
	}
	p.World.Geometry = nil
	reset()
	if got = p.combatRetreat(quake.UserCmd{}, combatSpacing(s)); got.Forward != 0 || got.Side != 0 {
		t.Fatal("moved without geometry")
	}
}
