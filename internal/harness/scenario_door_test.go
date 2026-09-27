package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestDoorPassProbeScenario(t *testing.T) {
	s, err := Load("../../scripts/scenarios/base2-closing-door-approach.json")
	if err != nil {
		t.Fatal(err)
	}
	if !s.BotDoorPassProbe {
		t.Fatal("fixture did not enable door probe")
	}
	for _, x := range []float64{-96, -64, -48} {
		for _, speed := range []int{0, 80, 160, 240, 300} {
			variant := s
			variant.BotOrigin[0] = x
			variant.BotDoorPassSpeed = speed
			if err := variant.Validate(); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, tc := range []struct {
		name   string
		change func(*Scenario)
	}{
		{"wrong_map", func(s *Scenario) { s.Map = "base1" }},
		{"wrong_origin", func(s *Scenario) { s.BotOrigin[0]-- }},
		{"no_hold", func(s *Scenario) { s.BotReleaseFrame = 0 }},
		{"low_speed", func(s *Scenario) { s.BotDoorPassSpeed = 79 }},
		{"high_speed", func(s *Scenario) { s.BotDoorPassSpeed = 301 }},
		{"speed_without_probe", func(s *Scenario) { s.BotDoorPassProbe = false; s.BotDoorPassSpeed = 160 }},
		{"relocation", func(s *Scenario) { s.BotReleaseOrigin = &quake.Vec3{0, 0, 24} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			invalid := s
			tc.change(&invalid)
			if invalid.Validate() == nil {
				t.Fatal("unsupported door setup accepted")
			}
		})
	}
}
