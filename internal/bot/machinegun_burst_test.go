package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestMachinegunShortBursts(t *testing.T) {
	p := &Planner{}
	for f := 1; f <= 30; f++ {
		s := quake.Snapshot{Map: "base2", Frame: f, Health: 100, Weapon: "models/weapons/v_machn/tris.md2"}
		for repeat := 0; repeat < 2; repeat++ {
			cmd := p.limitMachinegunBurst(s, quake.UserCmd{Buttons: 3, Forward: 200, Pitch: 123})
			want := (f-1)%5 < 3
			if (cmd.Buttons&1 != 0) != want || cmd.Buttons&2 == 0 || cmd.Forward != 200 || cmd.Pitch != 123 {
				t.Fatalf("frame%d repeat%d cmd%+v", f, repeat, cmd)
			}
		}
	}
}
func TestMachinegunPauseAndReset(t *testing.T) {
	p := &Planner{}
	s := quake.Snapshot{Map: "base2", Frame: 10, Health: 100, Weapon: "Machinegun"}
	p.limitMachinegunBurst(s, quake.UserCmd{Buttons: 1})
	s.Frame = 11
	p.limitMachinegunBurst(s, quake.UserCmd{})
	s.Frame = 12
	p.limitMachinegunBurst(s, quake.UserCmd{})
	s.Frame = 13
	if p.limitMachinegunBurst(s, quake.UserCmd{Buttons: 1}).Buttons != 1 {
		t.Fatal("natural release did not reset burst")
	}
	s.Health = -1
	s.Frame++
	if p.limitMachinegunBurst(s, quake.UserCmd{Buttons: 1}).Buttons != 1 {
		t.Fatal("respawn command suppressed")
	}
	s.Health = 100
	s.Weapon = "Chaingun"
	for i := 0; i < 15; i++ {
		s.Frame++
		if p.limitMachinegunBurst(s, quake.UserCmd{Buttons: 1}).Buttons != 1 {
			t.Fatal("other weapon gated")
		}
	}
	s.Weapon = "Machinegun"
	s.Frame = 1
	if p.limitMachinegunBurst(s, quake.UserCmd{Buttons: 1}).Buttons != 1 {
		t.Fatal("rewind retained state")
	}
}
