package quake

import "testing"

func TestButtonRelayChainAndInvalidLinks(t *testing.T) {
	base := []MapEntity{{Class: "func_door", Model: 1, TargetName: "gate"}, {Class: "func_button", Model: 2, Target: "relay"}, {Class: "trigger_relay", TargetName: "relay", Target: "gate"}}
	for _, kind := range []string{"touch", "shoot", "remote", "missing", "cycle", "unsupported", "depth"} {
		t.Run(kind, func(t *testing.T) {
			m := &MapInfo{Models: make([]BSPModel, 3), Entities: append([]MapEntity(nil), base...)}
			want := kind == "touch" || kind == "shoot"
			switch kind {
			case "shoot":
				m.Entities[1].Health = 20
			case "remote":
				m.Entities[1].TargetName = "remote"
			case "missing":
				m.Entities = m.Entities[:2]
			case "cycle":
				m.Entities[2].Target = "relay"
			case "unsupported":
				m.Entities[2].Class = "target_crosslevel_trigger"
			case "depth":
				m.Entities[2].Target = "r1"
				for i := 1; i <= 8; i++ {
					name := string(rune('a' + i))
					next := string(rune('a' + i + 1))
					if i == 1 {
						name = "r1"
					}
					if i == 8 {
						next = "gate"
					}
					m.Entities = append(m.Entities, MapEntity{Class: "trigger_relay", TargetName: name, Target: next})
				}
			}
			button, action, ok := m.ButtonForDoor(1)
			if ok != want {
				t.Fatal("invalid physical candidate", ok)
			}
			if want {
				chain := m.ButtonDoorChain(button.Model, 1)
				if len(chain) != 2 || chain[1].Class != "trigger_relay" {
					t.Fatal(chain)
				}
				expected := "touch"
				if kind == "shoot" {
					expected = "shoot"
				}
				if action != expected {
					t.Fatal(action)
				}
			}
		})
	}
}
