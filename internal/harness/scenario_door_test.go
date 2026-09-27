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
	for _, tc := range []struct {
		name   string
		change func(*Scenario)
	}{
		{"wrong_map", func(s *Scenario) { s.Map = "base1" }},
		{"wrong_origin", func(s *Scenario) { s.BotOrigin[0]-- }},
		{"no_hold", func(s *Scenario) { s.BotReleaseFrame = 0 }},
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
