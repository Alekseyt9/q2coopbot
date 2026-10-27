package policy

import (
	"math"
	"testing"
)

// Schema/state checks only. No network construction, LoadPPO or neural forward.
func movementOptionsMetadata() PPOFile {
	head := &MovementOptionsFile{Version: MovementOptionsVersion, Sampler: MovementOptionsSampler, Duration: 5, Shifts: movementOptionShifts}
	for name, n := range map[string]int{"actor": 5, "value": 1} {
		width := 1635
		net := []DenseLayer{}
		for _, count := range []int{32, 16, n} {
			layer := DenseLayer{Bias: make([]float64, count), Weight: make([][]float64, count)}
			for i := range layer.Weight {
				layer.Weight[i] = make([]float64, width)
			}
			net = append(net, layer)
			width = count
		}
		if name == "actor" {
			head.Actor = net
		} else {
			head.Value = net
		}
	}
	return PPOFile{Features: MemoryRouteFeatureVersion, ActionDistribution: MovementOptionsDistribution, Attention: &AttentionFile{}, AimCandidateHead: &AimCandidateFile{}, ConditionalFire: &SpatialAimFile{}, MovementOptions: head}
}

func TestMovementOptionsMetadataAdmission(t *testing.T) {
	f := movementOptionsMetadata()
	if err := validateMovementOptions(f); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*PPOFile){
		func(f *PPOFile) { f.MovementOptions.Duration = 4 },
		func(f *PPOFile) { f.MovementOptions.Shifts[1][0] = -.5 },
		func(f *PPOFile) { f.ActionDistribution = JointAimCandidateVersion },
		func(f *PPOFile) { f.Features = EquippedWeaponFeatureVersion },
		func(f *PPOFile) { f.Attention = nil },
		func(f *PPOFile) { f.MovementOptions = nil },
		func(f *PPOFile) { f.Deterministic = true },
		func(f *PPOFile) { f.DiagnosticMovementProbe = &DiagnosticMovementProbeFile{} },
		func(f *PPOFile) { f.WideAim = &WideAimFile{} },
		func(f *PPOFile) { f.MovementOptions.Actor[0].Weight[0][0] = math.NaN() },
	} {
		bad := movementOptionsMetadata()
		mutate(&bad)
		if validateMovementOptions(bad) == nil {
			t.Fatal("invalid movement options admitted")
		}
	}
}

func TestMovementOptionsScheduleAndHeldReceipt(t *testing.T) {
	id := Identity{Map: "jail3", Life: 1, Frame: 98}
	state := movementOptionState{Identity: id, Started: 98, Choice: 1, Valid: true}
	for _, frame := range []int{99, 100, 101, 102} {
		id.Frame = frame
		if movementOptionStart(state, id) {
			t.Fatal("option ended early", frame)
		}
		state.Identity = id
	}
	id.Frame = 103
	if !movementOptionStart(state, id) {
		t.Fatal("option held more than5frames")
	}
	id.Frame = 104
	if !movementOptionStart(state, id) {
		t.Fatal("missing frame did not reset option")
	}
	id.Frame = 102
	id.Life = 2
	if !movementOptionStart(state, id) {
		t.Fatal("life reset ignored")
	}
	f := movementOptionsMetadata()
	f.MovementOptions.Enabled = true
	p := PPO{file: f}
	o := Observation{Identity: Identity{Frame: 100}}
	r := &MovementOptionSample{Version: MovementOptionsVersion, Started: 98, Choice: 1}
	shift, lp, err := p.reviewMovementOption(o, nil, r)
	if err != nil || shift != [2]float64{-1, 0} || lp != 0 {
		t.Fatal("held action changed", shift, lp, err)
	}
	r.LogProbability = .1
	if _, _, err = p.reviewMovementOption(o, nil, r); err == nil {
		t.Fatal("held decision density counted twice")
	}
	f.MovementOptions.Enabled = false
	p.file = f
	if _, _, err = p.reviewMovementOption(o, nil, r); err == nil {
		t.Fatal("disabled event accepted")
	}
}
