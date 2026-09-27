package bot

import (
	"context"
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestProjectileComparisonControls(t *testing.T) {
	for _, light := range []int{-1, 128, 256} {
		if err := Run(context.Background(), Config{TestLight: &light}); err == nil {
			t.Fatal("test lighting allowed outside prepared combat fixture")
		}
	}
	if err := Run(context.Background(), Config{TestCombatBarrier: true}); err == nil {
		t.Fatal("combat barrier allowed outside prepared frame-paced scene")
	}
	for _, cfg := range []Config{{TestDisableProjectileLead: true}, {TestProjectileComparison: true}, {TestProjectileComparison: true, FramePaced: true, TestWeaponSwitchFixture: "rail_precision"}} {
		if err := Run(context.Background(), cfg); err == nil {
			t.Fatal("comparison allowed outside projectile fixture")
		}
	}
	if projectileFixtureWeaponReady("projectile_hyper", "Blaster") || !projectileFixtureWeaponReady("projectile_hyper", "models/weapons/v_hyperb/tris.md2") || !projectileFixtureWeaponReady("projectile_blaster", "Blaster") {
		t.Fatal("fixture weapon confirmation failed")
	}
}

func TestInterceptTime(t *testing.T) {
	for _, tc := range []struct {
		name string
		v    quake.Vec3
		ok   bool
	}{
		{"still", quake.Vec3{}, true}, {"crossing", quake.Vec3{0, 200, 0}, true},
		{"approaching", quake.Vec3{-200, 0, 0}, true}, {"escaping", quake.Vec3{1200, 0, 0}, false},
		{"same_speed_away", quake.Vec3{1000, 0, 0}, false}, {"same_speed_toward", quake.Vec3{-1000, 0, 0}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			at, ok := interceptTime(quake.Vec3{}, quake.Vec3{500, 0, 0}, tc.v, 1000)
			if ok != tc.ok {
				t.Fatalf("t=%v ok=%v", at, ok)
			}
			if ok {
				pos := quake.Vec3{500 + tc.v[0]*at, tc.v[1] * at, tc.v[2] * at}
				if math.Abs(quake.Distance(quake.Vec3{}, pos)-at*1000) > 1e-6 {
					t.Fatal("intercept misses target")
				}
			}
		})
	}
}

func TestProjectileConfidenceDependsOnFlightTime(t *testing.T) {
	if !projectileHistoryReliable(1, 0.15) || projectileHistoryReliable(2, 0.5) || !projectileHistoryReliable(6, 0.5) {
		t.Fatal("forecast horizon must grow only with observed steady history")
	}
	if !projectileMotionReliable(35, 0.1) || projectileMotionReliable(35, 0.5) || !projectileMotionReliable(0, 0.5) {
		t.Fatal("distant accelerating target trusted or steady motion rejected")
	}
	p := &Planner{}
	s := quake.Snapshot{Map: "base1", Health: 100, Enemies: []quake.Object{{ID: 7, Class: "monster_soldier"}}}
	for i, x := range []float64{0, 2, 4, 2} {
		s.Frame = i + 1
		s.Enemies[0].Origin[0] = x
		p.observeEnemyMotion(s)
		p.World.Snapshot = s
	}
	if p.enemyMotion[7].stable {
		t.Fatal("small reversal accepted as stable motion")
	}
}

func TestMotionRequiresStableContinuousObservations(t *testing.T) {
	p := &Planner{}
	s := quake.Snapshot{Map: "base1", Health: 100, Enemies: []quake.Object{{ID: 7, Class: "monster_infantry"}}}
	step := func(frame int, x float64, want bool) {
		t.Helper()
		s.Frame = frame
		s.Enemies[0].Origin = quake.Vec3{x, 0, 0}
		p.observeEnemyMotion(s)
		p.World.Snapshot = s
		if p.enemyMotion[7].stable != want {
			t.Fatalf("frame%d: %+v", frame, p.enemyMotion[7])
		}
	}
	step(1, 0, false)
	step(2, 5, false)
	step(3, 10, true)
	if p.enemyMotion[7].steadyFrames != 2 {
		t.Fatal("steady duration must include both observed intervals")
	}
	step(4, 5, false)
	if p.enemyMotion[7].steadyFrames != 1 {
		t.Fatal("turn retained the previous course's duration")
	}
	step(5, 0, true) // reversal needs a second matching velocity
	step(7, -10, false)
	if p.enemyMotion[7].steadyFrames != 0 {
		t.Fatal("observation gap retained steady history")
	}
	step(8, -15, false)
	step(9, -20, true)
	step(10, 500, false)
	step(11, 505, false)
	step(12, 510, true)
	s.Map = "base2"
	step(13, 515, false)
	s.Health = 0
	step(14, 520, false)
	s.Health = 100
	step(15, 525, false)
	s.Enemies = nil
	s.Frame = 16
	p.observeEnemyMotion(s)
	if len(p.enemyMotion) != 0 {
		t.Fatal("unobserved target retained")
	}
}

func TestProjectileAimUsesCurrentWeaponAndGeometry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	e := quake.Object{ID: 7, Class: "monster_infantry", Origin: quake.Vec3{96, -200, 24}, Solid: 8290}
	s := quake.Snapshot{Frame: 3, Self: quake.Vec3{32, -224, 24}, Weapon: "Blaster"}
	p := &Planner{World: World{Geometry: &g}, enemyMotion: map[int]enemyMotion{7: {class: e.Class, frame: 3, stable: true, steadyFrames: 8, velocity: quake.Vec3{0, 100, 0}}}}
	for _, weapon := range []string{"Blaster", "models/weapons/v_hyperb/tris.md2"} {
		s.Weapon = weapon
		aim, flight := p.projectileAim(s, e)
		if flight <= 0 || aim[1] <= e.AimPoint()[1] || aim[2] != e.AimPoint()[2] {
			t.Fatalf("%s: %v %v", weapon, aim, flight)
		}
	}
	p.TestDisableProjectileLead = true
	if aim, flight := p.projectileAim(s, e); flight != 0 || aim != e.AimPoint() {
		t.Fatal("baseline did not use current observed aim point")
	}
	p.TestDisableProjectileLead = false
	s.Weapon = "models/weapons/v_rail/tris.md2"
	if aim, flight := p.projectileAim(s, e); flight != 0 || aim != e.AimPoint() {
		t.Fatal("hitscan given projectile lead")
	}
	s.Weapon = "Blaster"
	s.Frame = 4
	if _, flight := p.projectileAim(s, e); flight != 0 {
		t.Fatal("stale velocity accepted")
	}
	s.Frame = 3
	// The eventual fire decision must still protect the teammate on the
	// predicted shot segment, rather than bypassing arbitration for a lead.
	clear := true
	e.ClearShot = &clear
	s.Health = 100
	s.Map = "base1"
	s.Enemies = []quake.Object{e}
	friend := quake.Vec3{64, -210, 24}
	s.Teammate = &friend
	now := time.Now()
	p.World = World{Map: "base1", Snapshot: s, Geometry: &g, GeometryStatus: "ready", Goal: "cover_teammate", Updated: now}
	cmd := p.commandAt(quake.UserCmd{}, now)
	if cmd.Buttons&1 != 0 || p.World.Command.LimitReason != "friendly_line_of_fire" || p.World.Command.LeadSeconds <= 0 {
		t.Fatalf("unsafe lead: %+v %+v", cmd, p.World.Command)
	}
	p.World.Geometry = nil
	if _, flight := p.projectileAim(s, e); flight != 0 {
		t.Fatal("unverified trajectory accepted")
	}
}
