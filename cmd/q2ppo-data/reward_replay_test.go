package main

import (
	"math"
	"q2coopbot/internal/learningenv"
	"testing"
)

func TestRewardReplayOnlyAllowsLegacySumRoundoff(t *testing.T) {
	x := -0.06664932857607843
	a := learningenv.Reward{Version: learningenv.TargetSequenceRewardVersion, Available: true, Score: &x,
		Components: map[string]float64{"turn_away": -.005}}
	y := math.Nextafter(x, math.Inf(-1))
	b := a
	b.Score = &y
	if !equivalentRewardReplay(a, b) {
		t.Fatal("legacy roundoff rejected")
	}
	y = x + 1e-9
	if equivalentRewardReplay(a, b) {
		t.Fatal("material score difference accepted")
	}
	y = x
	b.Components = map[string]float64{"turn_away": -.005000000000000001}
	if equivalentRewardReplay(a, b) {
		t.Fatal("component difference accepted")
	}
	b = a
	b.Episode = "another"
	if equivalentRewardReplay(a, b) {
		t.Fatal("identity difference accepted")
	}
	b = a
	b.TargetCycle = &learningenv.TargetCycleEvidence{Entity: 7}
	if equivalentRewardReplay(a, b) {
		t.Fatal("cycle evidence difference accepted")
	}
	b = a
	y = math.NaN()
	b.Score = &y
	if equivalentRewardReplay(a, b) {
		t.Fatal("nonfinite score accepted")
	}
}
