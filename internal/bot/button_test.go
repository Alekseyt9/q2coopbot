package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestButtonOwnerLossCancelsBeforeSearchEarlyReturn(t *testing.T) {
	for _, kind := range []string{"hidden", "changed", "dead"} {
		t.Run(kind, func(t *testing.T) {
			mate := quake.Vec3{200, 0, 24}
			p := &Planner{World: World{Map: "test", Geometry: &quake.MapInfo{}}, button: &buttonTask{teammateEntity: 1, started: 1}}
			s := quake.Snapshot{Map: "test", Frame: 10, Health: 100, Teammate: &mate, TeammateEntity: 1}
			switch kind {
			case "hidden":
				s.Teammate = nil
			case "changed":
				s.TeammateEntity = 2
			case "dead":
				s.Health = 0
			}
			p.update(s, "")
			if p.button != nil || p.buttonCooldown != 40 {
				t.Fatalf("old interaction survived: %+v", p.button)
			}
		})
	}
}

func TestButtonDoesNotOverrideNewParentGoal(t *testing.T) {
	for _, recover := range []bool{false, true} {
		mate := quake.Vec3{200, 0, 24}
		hp := int16(100)
		want := "cover_teammate"
		s := quake.Snapshot{Map: "test", Frame: 10, Self: quake.Vec3{150, 0, 24}, Teammate: &mate, TeammateEntity: 1, Health: hp}
		if recover {
			s.Health = 20
			s.Pickups = []quake.Object{{Class: "item_health", Origin: quake.Vec3{180, 0, 24}}}
			want = "recover_health"
		}
		p := &Planner{World: World{Map: "test", Geometry: &quake.MapInfo{}}, button: &buttonTask{teammateEntity: 1, started: 1}}
		p.update(s, "")
		if p.button != nil || p.World.Goal != want {
			t.Fatalf("recover=%t task=%+v goal=%s", recover, p.button, p.World.Goal)
		}
	}
}
