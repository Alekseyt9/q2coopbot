package policy

// Cross-horizon agreement is a diagnostic, not a calibrated probability. All
// horizons compare the same proposed plan against the same learned motor plan.
type HJEPASearchAudit struct {
	ProposedChange   bool       `json:"proposed_change"`
	Accepted         bool       `json:"accepted"`
	AgreementChecked bool       `json:"agreement_checked"`
	UtilityGains     [3]float64 `json:"utility_gains_1_4_16"`
}

func (p *HJEPA) LastSearchAudit() HJEPASearchAudit { return p.lastSearch }
func (p *HJEPA) Proposal(o Observation) (Action, error) {
	q := *p
	q.file.Planner = "bc"
	return q.Decide(o)
}

func (p *HJEPA) planUtilities(zs [][]float64, plan [][]float64) ([3]float64, error) {
	var values [3]float64
	for level := 0; level < 3; level++ {
		states, err := p.hRoll(level, zs[level], plan)
		if err != nil {
			return values, err
		}
		z := states[len(states)-1]
		for upper := level + 1; upper < 3; upper++ {
			z, err = hNet(p.file.Levels[upper].Encoder, z)
			if err != nil {
				return values, err
			}
		}
		u, err := hNet(p.file.Levels[2].Utility, z)
		if err != nil {
			return values, err
		}
		values[level] = u[0]
	}
	return values, nil
}

func (p *HJEPA) admitPlan(zs [][]float64, prior []float64, plan [][]float64) ([]float64, error) {
	changed := false
	for i, v := range plan[0] {
		if v != prior[i] {
			changed = true
		}
	}
	p.lastSearch = HJEPASearchAudit{ProposedChange: changed, Accepted: changed}
	if p.file.ConsistencyMargin == nil || !changed {
		return append([]float64{}, plan[0]...), nil
	}
	baseline := make([][]float64, len(plan))
	for i := range baseline {
		baseline[i] = append([]float64{}, prior...)
	}
	proposed, err := p.planUtilities(zs, plan)
	if err != nil {
		return nil, err
	}
	reference, err := p.planUtilities(zs, baseline)
	if err != nil {
		return nil, err
	}
	p.lastSearch.AgreementChecked = true
	for i := range proposed {
		gain := proposed[i] - reference[i]
		p.lastSearch.UtilityGains[i] = gain
		if gain <= *p.file.ConsistencyMargin {
			p.lastSearch.Accepted = false
		}
	}
	if !p.lastSearch.Accepted {
		return append([]float64{}, prior...), nil
	}
	return append([]float64{}, plan[0]...), nil
}
