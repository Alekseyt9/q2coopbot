package policy

import (
	"fmt"
	"math"
	"sort"
)

const TargetHeadVersion = "combat_target_conditioned_aim_v1"
const TargetFeatureVersion = "combat_features_v7"
const TargetFeatureWidth = 854
const TargetLogitOffset = 20
const TargetAimOffset = 29
const TargetOutputWidth = 45

// Intent is the provider's previous visible choice, not a hidden enemy track.
type TargetIntent struct {
	Identity Identity `json:"identity"`
	Entity   int      `json:"entity"`
	Track    int      `json:"track"`
}

// Slot ordering matches the observation feature contract, including ties.
func TargetEnemies(o Observation) []Enemy {
	enemies := append([]Enemy(nil), o.Enemies...)
	sort.SliceStable(enemies, func(i, j int) bool { return enemies[i].Distance < enemies[j].Distance })
	if len(enemies) > 8 {
		enemies = enemies[:8]
	}
	return enemies
}

func TargetAvailability(o Observation) ([]bool, error) {
	mask := make([]bool, 9)
	mask[0] = true
	for i, e := range TargetEnemies(o) {
		_, known, err := ObservedAimDirection(o, e)
		if err != nil {
			return nil, err
		}
		mask[i+1] = known && e.ID > 0
	}
	return mask, nil
}

func targetHistoryFeatures(o Observation) []float64 {
	v := make([]float64, 9)
	v[0] = 1
	p := o.PreviousTarget
	if p == nil || !SameLife(p.Identity, o.Identity) || p.Identity.Frame+1 != o.Identity.Frame {
		return v
	}
	for i, e := range TargetEnemies(o) {
		if e.ID == p.Entity && e.Track != nil && *e.Track == p.Track && e.ClearShot != nil && *e.ClearShot {
			v[0] = 0
			v[i+1] = 1
			break
		}
	}
	return v
}

func targetLogProbabilities(o Observation, raw []float64) ([]float64, error) {
	if len(raw) != 9 {
		return nil, fmt.Errorf("invalid target logits")
	}
	mask, err := TargetAvailability(o)
	if err != nil {
		return nil, err
	}
	maximum := math.Inf(-1)
	for i, x := range raw {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("nonfinite target logits")
		}
		if mask[i] && x > maximum {
			maximum = x
		}
	}
	sum := 0.
	for i, x := range raw {
		if mask[i] {
			sum += math.Exp(x - maximum)
		}
	}
	lp := make([]float64, 9)
	for i, x := range raw {
		lp[i] = math.Inf(-1)
		if mask[i] {
			lp[i] = x - maximum - math.Log(sum)
		}
	}
	return lp, nil
}

func targetMeans(raw []float64, slot int) ([4]float64, error) {
	if len(raw) != TargetOutputWidth || slot < 0 || slot > 8 {
		return [4]float64{}, fmt.Errorf("invalid target action")
	}
	means := [4]float64{raw[0], raw[1], raw[2], raw[3]}
	if slot > 0 {
		means[2] = raw[TargetAimOffset+2*(slot-1)]
		means[3] = raw[TargetAimOffset+2*(slot-1)+1]
	}
	return means, nil
}
