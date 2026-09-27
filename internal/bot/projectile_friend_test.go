package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestProjectileFriendUsesCollisionTime(t *testing.T) {
	friend := quake.Vec3{200, 60, 24}
	s := quake.Snapshot{Frame: 10, Teammate: &friend, TeammateEntity: 1, Weapon: "Blaster"}
	from, to := quake.Vec3{0, 0, 46}, quake.Vec3{500, 0, 46}
	p := &Planner{shotTeammateMotion: shotTeammateMotion{velocity: quake.Vec3{0, -300, 0}, frame: 10, entity: 1, known: true}}
	if teammateBlocksShot(from, to, friend) {
		t.Fatal("test must be clear now")
	}
	if !p.teammateEntersProjectile(s, from, to) {
		t.Fatal("missed approaching teammate")
	}
	p.shotTeammateMotion.velocity[1] = 300
	if p.teammateEntersProjectile(s, from, to) {
		t.Fatal("blocked departing teammate")
	}
	p.shotTeammateMotion.velocityChange = 180
	if !p.teammateEntersProjectile(s, from, to) {
		t.Fatal("ignored uncertain departing velocity")
	}
	friend[1] = 40
	p.shotTeammateMotion.velocityChange = 62.5
	if !p.teammateEntersProjectile(s, from, to) {
		t.Fatal("released guard during continued braking")
	}
	friend[1] = 60
	p.shotTeammateMotion.velocityChange = 0
	p.shotTeammateMotion.velocity[1] = -100
	if p.teammateEntersProjectile(s, from, to) {
		t.Fatal("blocked crossing after bolt passes")
	}
	p.shotTeammateMotion.velocity[1] = -300
	s.Weapon = "models/weapons/v_rail/tris.md2"
	if p.teammateEntersProjectile(s, from, to) {
		t.Fatal("applied bolt speed to hitscan")
	}
	s.Weapon = "Blaster"
	s.Frame++
	if p.teammateEntersProjectile(s, from, to) {
		t.Fatal("used stale motion")
	}
}

func TestProjectileFriendDiscardsDiscontinuousObservations(t *testing.T) {
	a, b := quake.Vec3{200, 90, 24}, quake.Vec3{200, 60, 24}
	old := quake.Snapshot{Map: "base1", Frame: 9, Health: 100, Teammate: &a, TeammateEntity: 1}
	now := quake.Snapshot{Map: "base1", Frame: 10, Health: 100, Teammate: &b, TeammateEntity: 1}
	p := &Planner{World: World{Snapshot: old}}
	p.observeShotTeammateMotion(now)
	if !p.shotTeammateMotion.known || p.shotTeammateMotion.velocity[1] != -300 {
		t.Fatal("missing visible velocity")
	}
	for _, kind := range []string{"gap", "map", "entity", "hidden", "dead", "teleport"} {
		s := now
		switch kind {
		case "gap":
			s.Frame++
		case "map":
			s.Map = "base2"
		case "entity":
			s.TeammateEntity = 2
		case "hidden":
			s.Teammate = nil
		case "dead":
			s.Health = 0
		case "teleport":
			v := quake.Vec3{300, 0, 24}
			s.Teammate = &v
		}
		p.observeShotTeammateMotion(s)
		if p.shotTeammateMotion.known {
			t.Fatalf("retained motion after %s", kind)
		}
	}
}
