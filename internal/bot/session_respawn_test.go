package bot

import (
	"q2coopbot/internal/harness"
	"testing"
)

func TestObserverFixtureKillsOnceAndStopsAfterObservedRespawn(t *testing.T) {
	f := &harness.ObserverRespawn{AfterFrames: 5}
	var s observerRespawnState
	if kill, _, err := s.tick(f, 40, 44, 100); kill || err != nil {
		t.Fatal(kill, err)
	}
	if kill, _, err := s.tick(f, 40, 45, 100); !kill || err != nil {
		t.Fatal(kill, err)
	}
	if kill, _, err := s.tick(f, 40, 45, 100); kill || err != nil {
		t.Fatal("duplicate kill", err)
	}
	if kill, pulse, err := s.tick(f, 40, 46, 0); kill || !pulse || err != nil {
		t.Fatal(kill, pulse, err)
	}
	if _, pulse, _ := s.tick(f, 40, 47, 0); pulse {
		t.Fatal("missing pulse release")
	}
	if _, pulse, err := s.tick(f, 40, 48, 100); pulse || !s.done || err != nil {
		t.Fatal(s, err)
	}
	s = observerRespawnState{}
	if _, _, err := s.tick(f, 40, 46, 100); err == nil {
		t.Fatal("late injection accepted")
	}
}
