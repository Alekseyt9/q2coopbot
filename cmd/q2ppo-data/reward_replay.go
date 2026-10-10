package main

import (
	"fmt"
	"math"
	"q2coopbot/internal/learningenv"
	"reflect"
)

// Legacy v6/v7 exports summed three costs in Go map iteration order.
// Permit only floating-point summation roundoff in the total; every component
// and all identity, availability, terminal and target-cycle evidence stay exact.
func equivalentRewardReplay(a, b learningenv.Reward) bool {
	if a.Score == nil || b.Score == nil {
		return reflect.DeepEqual(a, b)
	}
	x, y := *a.Score, *b.Score
	a.Score, b.Score = nil, nil
	if !reflect.DeepEqual(a, b) || math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
		return false
	}
	if a.Version != learningenv.ActionQualityRewardVersion && a.Version != learningenv.TargetSequenceRewardVersion {
		return x == y
	}
	scale := 1.0
	for _, value := range a.Components {
		scale += math.Abs(value)
	}
	return math.Abs(x-y) <= 8*(math.Nextafter(1, 2)-1)*scale
}

func verifyRewardReplay(replay, captured string) error {
	a, err := rows[learningenv.Reward](replay)
	if err != nil {
		return err
	}
	b, err := rows[learningenv.Reward](captured)
	if err != nil {
		return err
	}
	if len(a) != len(b) {
		return fmt.Errorf("reward row count differs")
	}
	for i := range a {
		if !equivalentRewardReplay(a[i], b[i]) {
			return fmt.Errorf("reward row %d differs", i)
		}
	}
	return nil
}
