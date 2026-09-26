package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

// Replays the entity state from ware1-continuous-fire-20260926-184307
// through perception and command generation, including release of held attack.
func TestGunnerDeathReleasesAttack(t *testing.T) {
	d := quake.NewDecoder()
	d.Map = "ware1"
	d.Config[34] = "models/monsters/gunner/tris.md2"
	d.Config[35] = "models/weapons/v_blast/tris.md2"
	f := quake.Frame{Number: 1, Gun: 3, Origin: quake.Vec3{-2609.75, -719.625, 24.125},
		Entities: map[int]quake.Entity{181: {Number: 181, Model: 2, Origin: quake.Vec3{-2199.875, -461.625, 24}}}}
	f.Stats[1] = 74
	p := &Planner{}
	previous := quake.UserCmd{}
	check := func(animation int, wantAttack bool) {
		t.Helper()
		e := f.Entities[181]
		e.Frame = animation
		f.Entities[181] = e
		s := d.Snapshot(f)
		clear := true
		for i := range s.Enemies {
			s.Enemies[i].ClearShot = &clear
		}
		p.World = World{Map: "ware1", Snapshot: s, Goal: "cover_teammate", Updated: time.Now(), GeometryStatus: "ready"}
		cmd := p.commandAt(previous, p.World.Updated)
		if got := cmd.Buttons&1 != 0; got != wantAttack {
			t.Fatalf("animation %d server frame %d: attack=%v, want %v", animation, f.Number, got, wantAttack)
		}
		previous = cmd
		f.Number++
	}
	check(189, true) // Live pain animation immediately before death range.
	for animation := 190; animation <= 200; animation++ {
		check(animation, false)
	}
	for i := 0; i < 501; i++ {
		check(200, false) // Settled corpse remains network-visible indefinitely.
	}
	check(201, true) // A live ducking gunner must remain a target.
}
