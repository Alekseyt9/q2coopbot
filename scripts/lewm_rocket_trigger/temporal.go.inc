type modeScheduler struct {
	Readout DenseLayer
	Heads []DenseLayer
	Frames int
	SecondChoice *int
	ProbeDecision int
	SecondReadout *DenseLayer
	SecondSwitchMargin float64
	index, started int
	decisions int
}

func (m *modeScheduler) choose(z, equipped []float64) int {
	scores:=affine(m.Readout,append(z,equipped...))
	best:=0
	for i:=1; i<len(scores); i++ { if scores[i]>scores[best] { best=i } }
	return best
}

type slowPolicy struct {
	Encoder []DenseLayer
	Projection DenseLayer
	Controller DenseLayer
	Termination *DenseLayer
	Scheduler *modeScheduler
	last Identity
	start int
	z [3]float64
}

func loadSlowPolicy(path string) (*slowPolicy, error) {
	data, err := os.ReadFile(path)
	if err != nil { return nil, err }
	var policy slowPolicy
	if err := json.Unmarshal(data, &policy); err != nil { return nil, err }
	if err := validateHNet(policy.Encoder, 871, 64); err != nil { return nil, err }
	if err := validateHNet([]DenseLayer{policy.Projection}, 64, 2); err != nil { return nil, err }
	if err := validateHNet([]DenseLayer{policy.Controller}, 5, 3); err != nil { return nil, err }
	if policy.Termination != nil {
		if err := validateHNet([]DenseLayer{*policy.Termination}, 6, 1); err != nil { return nil, err }
	}
	if m:=policy.Scheduler; m!=nil {
		if policy.Termination!=nil || m.Frames<1 || m.Frames>128 || len(m.Heads)<2 || len(m.Heads)>8 {
			return nil,fmt.Errorf("invalid temporal mode configuration")
		}
		if err:=validateHNet([]DenseLayer{m.Readout},76,len(m.Heads)); err!=nil {return nil,err}
		if m.SecondChoice != nil && (*m.SecondChoice < 0 || *m.SecondChoice >= len(m.Heads)) {
			return nil,fmt.Errorf("invalid second temporal choice")
		}
		if m.ProbeDecision != 0 && (m.SecondChoice == nil || m.ProbeDecision < 2 || m.ProbeDecision > 6) {
			return nil,fmt.Errorf("invalid temporal probe decision")
		}
		if m.SecondReadout != nil {
			if m.SecondChoice != nil || m.SecondSwitchMargin < 0 {
				return nil,fmt.Errorf("invalid learned second temporal choice")
			}
			if err:=validateHNet([]DenseLayer{*m.SecondReadout},76,len(m.Heads)); err!=nil {return nil,err}
		}
		for _,h:=range m.Heads {
			if err:=validateHNet([]DenseLayer{h},6,1); err!=nil {return nil,err}
		}
	}
	return &policy, nil
}

func (p *slowPolicy) condition(o Observation, raw []float64) error {
	continuous := SameLife(p.last,o.Identity) && o.Identity.Frame == p.last.Frame+1
	if !continuous { p.z = [3]float64{} }
	age := o.Identity.Frame-p.start
	refresh := !continuous || o.Identity.Frame-p.start >= 8
	if refresh || p.Termination != nil || p.Scheduler != nil {
		x, err := HJEPAFeatures(o,HJEPAContextFeatureVersion)
		if err != nil { return err }
		z, err := hNet(p.Encoder,x)
		if err != nil { return err }
		head:=p.Termination
		switched:=false
		if m:=p.Scheduler; m!=nil {
			if !continuous || o.Identity.Frame-m.started>=m.Frames {
				choice:=m.choose(z,x[845:857])
				if !continuous { m.decisions = 0 }
				m.decisions++
				probe := m.ProbeDecision
				if probe == 0 { probe = 2 }
				if m.SecondChoice != nil && m.decisions == probe {
					choice = *m.SecondChoice
				} else if m.SecondReadout != nil && m.decisions == 2 {
						scores := affine(*m.SecondReadout,append(z,x[845:857]...))
						incumbent := scores[choice]
						for i, score := range scores {
							if score > incumbent + m.SecondSwitchMargin && score > scores[choice] {
								choice = i
							}
						}
				}
				switched=continuous && choice!=m.index
				m.index,m.started=choice,o.Identity.Frame
				if os.Getenv("Q2_LEWM_DURATION_AUDIT")=="1" {
					fmt.Fprintf(os.Stderr,"lewm_mode frame=%d index=%d\n",o.Identity.Frame,choice)
				}
			}
			head=&m.Heads[m.index]
		}
		ctx := affine(p.Projection,z)
		for i := range ctx { ctx[i] = math.Tanh(ctx[i]) }
		input := append(ctx,p.z[:]...)
		if continuous && head != nil {
			refresh = switched || affine(*head,append(input,float64(age)/8))[0] >= 0
		}
		if refresh {
			decision := affine(p.Controller,input)
			for i := range p.z { p.z[i] = math.Tanh(decision[i]) }
			p.start = o.Identity.Frame
			if os.Getenv("Q2_LEWM_DURATION_AUDIT") == "1" {
				fmt.Fprintf(os.Stderr,"lewm_refresh frame=%d elapsed=%d continuous=%t\n",o.Identity.Frame,age,continuous)
			}
		}
	}
	p.last = o.Identity
	for i, index := range []int{0,1,4} { raw[index] += p.z[i] }
	return nil
}
