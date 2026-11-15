package policy

import (
	"fmt"
	"math"
)

// Primitive plans always contain full primitive vectors. Each predictor receives
// exactly its trained action-window width, including nonconstant action blocks.
func hPredict(l HJEPALevel, z []float64, actions [][]float64) ([]float64, error) {
	if len(actions) != l.Stride {
		return nil, fmt.Errorf("H-JEPA prediction horizon")
	}
	in := append([]float64{}, z...)
	for _, a := range actions {
		if len(a) != 8 {
			return nil, fmt.Errorf("H-JEPA primitive width")
		}
		in = append(in, a...)
	}
	return hNet(l.Predictor, in)
}
func hCopyPlan(plan [][]float64) [][]float64 {
	out := make([][]float64, len(plan))
	for i, a := range plan {
		out[i] = append([]float64{}, a...)
	}
	return out
}

// Search both held actions and two-phase sequences, around the parent's plan.
// At lower levels changes affect the first block while retaining its suffix.
func hSequenceCandidates(parent [][]float64, candidates [][]float64, block int) [][][]float64 {
	out := [][][]float64{hCopyPlan(parent)}
	for _, a := range candidates {
		held := hCopyPlan(parent)
		prefix := hCopyPlan(parent)
		suffix := hCopyPlan(parent)
		for t := range parent {
			held[t] = append([]float64{}, a...)
			if t < block {
				prefix[t] = append([]float64{}, a...)
			} else {
				suffix[t] = append([]float64{}, a...)
			}
		}
		out = append(out, held, prefix, suffix)
	}
	return out
}
func hPriorCost(plan [][]float64, prior []float64) float64 {
	cost := 0.
	for _, a := range plan {
		for i, v := range a {
			cost += (v - prior[i]) * (v - prior[i]) / float64(len(plan))
		}
	}
	return cost
}
func hDistance(a, b []float64) float64 {
	cost := 0.
	for i, v := range a {
		cost += (v - b[i]) * (v - b[i]) / float64(len(a))
	}
	return cost
}

func (p *HJEPA) hRoll(level int, z []float64, plan [][]float64) ([][]float64, error) {
	stride := p.file.Levels[level].Stride
	if len(plan) == 0 || len(plan)%stride != 0 {
		return nil, fmt.Errorf("H-JEPA rollout horizon")
	}
	var states [][]float64
	for t := 0; t < len(plan); t += stride {
		next, e := hPredict(p.file.Levels[level], z, plan[t:t+stride])
		if e != nil {
			return nil, e
		}
		states = append(states, next)
		z = next
	}
	return states, nil
}

// Plan 16 frames at level 3; refine it via four level-2 transitions to the same
// terminal subgoal. Then refine the first 4-frame subgoal via four level-1
// transitions. Execute only the first primitive, and replan on the next frame.
func (p *HJEPA) sequenceAction(zs [][]float64, prior []float64) ([]float64, error) {
	parent := make([][]float64, 16)
	for t := range parent {
		parent[t] = append([]float64{}, prior...)
	}
	var goal []float64
	var middlePlan [][]float64
	for level := 2; level >= 0; level-- {
		block := len(parent) / 2
		if level == 1 {
			block = 4
		}
		if level == 0 {
			block = 1
		}
		best := math.Inf(1)
		var selected [][]float64
		var selectedStates [][]float64
		for _, plan := range hSequenceCandidates(parent, p.candidates(prior), block) {
			states, e := p.hRoll(level, zs[level], plan)
			if e != nil {
				return nil, e
			}
			terminal := states[len(states)-1]
			var score float64
			if goal == nil {
				u, e := hNet(p.file.Levels[level].Utility, terminal)
				if e != nil {
					return nil, e
				}
				score = -u[0]
			} else {
				projected, e := hNet(p.file.Levels[level+1].Encoder, terminal)
				if e != nil {
					return nil, e
				}
				score = hDistance(projected, goal)
			}
			score += p.file.PriorWeight * hPriorCost(plan, prior)
			if score < best {
				best = score
				selected = plan
				selectedStates = states
			}
		}
		if selected == nil {
			return nil, fmt.Errorf("H-JEPA no finite sequence")
		}
		if level == 0 {
			plan := hCopyPlan(middlePlan)
			for i := range selected {
				plan[i] = append([]float64{}, selected[i]...)
			}
			return p.admitPlan(zs, prior, plan)
		}
		// The middle level's first predicted state is the next four-frame subgoal.
		goal = selectedStates[0]
		parent = selected
		if level == 1 {
			middlePlan = hCopyPlan(selected)
			parent = hCopyPlan(selected[:4])
		}
	}
	return nil, fmt.Errorf("H-JEPA missing primitive")
}
