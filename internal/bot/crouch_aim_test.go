package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestObservedCrouchChangesShotPitch(t *testing.T) {
	d := quake.NewDecoder()
	d.Config[34] = "models/monsters/gunner/tris.md2"
	d.Config[35] = "models/weapons/v_blast/tris.md2"
	f := quake.Frame{Number: 1, Gun: 3, Entities: map[int]quake.Entity{}}
	f.Stats[1] = 100
	previous := quake.UserCmd{}
	for i, top := range []int{32, 0, 32} {
		solid := uint16(2 | 3<<5 | ((top+32)/8)<<10)
		f.Number = i + 1
		f.Entities[7] = quake.Entity{Number: 7, Model: 2, Origin: quake.Vec3{150, 0, 0}, Solid: solid, Frame: 201 + i}
		s := d.Snapshot(f)
		if len(s.Enemies) != 1 || s.Enemies[0].Solid != solid {
			t.Fatal("lost observed bounds")
		}
		clear := true
		s.Enemies[0].ClearShot = &clear
		p := &Planner{World: World{Map: "base1", Snapshot: s, Goal: "cover_teammate", Updated: time.Now(), GeometryStatus: "ready"}}
		cmd := p.commandAt(previous, p.World.Updated)
		z := 22.0
		if top == 0 {
			z = -8
		}
		want := quake.PitchTo(quake.Vec3{0, 0, 22}, quake.Vec3{150, 0, z}, 0)
		if cmd.Buttons&1 == 0 || cmd.Pitch != want {
			t.Fatalf("step%d: cmd=%+v want pitch=%v", i, cmd, want)
		}
		previous = cmd
	}
}
