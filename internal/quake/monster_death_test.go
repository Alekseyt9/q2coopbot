package quake

import "testing"

func TestMonsterDeathBoundariesAndModelNames(t *testing.T) {
	for _, tc := range []struct {
		model       string
		first, last int
	}{
		{"berserk", 223, 243}, {"brain", 123, 145}, {"bitch", 48, 82},
		{"gladiatr", 61, 82}, {"medic", 147, 176}, {"mutant", 15, 33},
		{"float", 104, 116}, {"hover", 162, 172},
	} {
		t.Run(tc.model, func(t *testing.T) {
			d := Decoder{Config: map[int]string{33: "models/monsters/" + tc.model + "/tris.md2"}}
			// A return to a living animation must clear the classification:
			// medics can revive corpses and entity numbers can be reused.
			for _, frame := range []int{tc.first - 1, tc.first, tc.last, tc.last + 1, tc.first - 1} {
				s := d.Snapshot(Frame{Number: 1, Entities: map[int]Entity{7: {Number: 7, Model: 1, Frame: frame, Solid: 8290}}})
				dead := frame >= tc.first && frame <= tc.last
				if (len(s.Defeated) == 1) != dead || (len(s.Enemies) == 0) != dead || len(s.Obstacles) != 1 {
					t.Fatalf("frame%d: %+v", frame, s)
				}
			}
		})
	}
	for _, model := range []string{"custom", "chick", "gladiator", "berserker"} {
		if monsterDeathAnimation(model, 70) {
			t.Fatalf("unverified model %s suppressed", model)
		}
	}
}
