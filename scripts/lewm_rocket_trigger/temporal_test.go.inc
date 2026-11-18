package policy

import "testing"

func TestSlowOptionPeriodAndLifeReset(t *testing.T) {
	p := slowPolicy{
		Encoder: []DenseLayer{hTestLayer(871, 64)},
		Projection: hTestLayer(64, 2),
		Controller: hTestLayer(5, 3),
	}
	// Depend on previous state so a spurious update or a missed reset is visible.
	p.Controller.Bias[0] = .5
	p.Controller.Weight[0][2] = 1
	o := testObservation()
	step := func(frame int) float64 {
		o.Identity.Frame = frame
		raw := make([]float64, 8)
		if err := p.condition(o, raw); err != nil { t.Fatal(err) }
		return raw[0]
	}
	first := step(1)
	for frame := 2; frame <= 8; frame++ {
		if got := step(frame); got != first { t.Fatalf("early update at %d",frame) }
	}
	if got := step(9); got == first { t.Fatal("missed slow update") }
	o.Identity.Life++
	if got := step(10); got != first { t.Fatal("new life retained previous option") }
	for frame := 11; frame <= 18; frame++ { step(frame) }
	if got := step(20); got != first { t.Fatal("frame gap retained previous option") }
}

func TestLearnedTerminationCanHoldBeyondEightAndReset(t *testing.T) {
	head := hTestLayer(6,1)
	head.Weight[0][5], head.Bias[0] = 1,-1.5
	p := slowPolicy{Encoder: []DenseLayer{hTestLayer(871,64)},Projection:hTestLayer(64,2),
		Controller:hTestLayer(5,3),Termination:&head}
	o := testObservation()
	for frame := 1; frame <= 14; frame++ {
		o.Identity.Frame = frame
		if err := p.condition(o,make([]float64,8)); err != nil { t.Fatal(err) }
		expected := 1
		if frame >= 13 { expected = 13 }
		if p.start != expected { t.Fatalf("frame %d, option start %d",frame,p.start) }
	}
	o.Identity.Frame = 16
	if err := p.condition(o,make([]float64,8)); err != nil { t.Fatal(err) }
	if p.start != 16 { t.Fatal("gap should reset the option") }
}

func TestTerminationUsesHeldDecisionAtEqualAge(t *testing.T) {
	for _,bias := range []float64{-.5,.5} {
		head := hTestLayer(6,1)
		head.Weight[0][2] = 1
		p := slowPolicy{Encoder:[]DenseLayer{hTestLayer(871,64)},Projection:hTestLayer(64,2),
			Controller:hTestLayer(5,3),Termination:&head}
		p.Controller.Bias[0] = bias
		o := testObservation()
		for f:=1; f<=2; f++ {
			o.Identity.Frame=f
			if err:=p.condition(o,make([]float64,8)); err!=nil { t.Fatal(err) }
		}
		if (p.start==2) != (bias>0) { t.Fatalf("unexpected refresh with bias %v",bias) }
	}
}

func TestModeSchedulerKeepsChoiceUntilBoundary(t *testing.T) {
	// Switch the readout preference between frames. Scores are consulted
	// only at the next decision boundary or at the beginning of a new life.
	m := modeScheduler{Readout:hTestLayer(76,2),Heads:[]DenseLayer{hTestLayer(6,1),hTestLayer(6,1)},Frames:3}
	m.Readout.Bias[0]=1
	p:=slowPolicy{Encoder:[]DenseLayer{hTestLayer(871,64)},Projection:hTestLayer(64,2),
		Controller:hTestLayer(5,3),Scheduler:&m}
	o:=testObservation()
	step:=func(frame,index int){
		o.Identity.Frame=frame
		if err:=p.condition(o,make([]float64,8)); err!=nil { t.Fatal(err) }
		if m.index!=index { t.Fatalf("frame %d chose %d, want %d",frame,m.index,index) }
	}
	step(1,0)
	m.Readout.Bias[1]=2
	step(2,0)
	step(3,0)
	step(4,1)
	o.Identity.Life++
	m.Readout.Bias[0]=3
	step(5,0)
}
