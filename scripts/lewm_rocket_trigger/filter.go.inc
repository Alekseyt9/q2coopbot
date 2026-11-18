// Compiled inside package policy only in the separate experimental client.
type rocketGate struct {
	Encoder, Step, Head []DenseLayer
	Probability, Clearance float64
}

func loadRocketGate(path string) (*rocketGate, error) {
	data, err := os.ReadFile(path)
	if err != nil { return nil, err }
	var g rocketGate
	if err = json.Unmarshal(data, &g); err != nil { return nil, err }
	if g.Probability <= 0 || g.Probability > 1 || g.Clearance < 0 || g.Clearance > 256 {
		return nil, fmt.Errorf("invalid rocket gate thresholds")
	}
	if err = validateHNet(g.Encoder, 871, 64); err != nil { return nil, err }
	if err = validateHNet(g.Step, 76, 64); err != nil { return nil, err }
	if err = validateHNet(g.Head, 140, 4); err != nil { return nil, err }
	return &g, nil
}

func (g *rocketGate) blastRisk(o Observation) (float64, error) {
	x, err := HJEPAFeatures(o, HJEPAContextFeatureVersion)
	if err != nil { return 0, err }
	start, err := hNet(g.Encoder, x)
	if err != nil { return 0, err }
	// The v2 readout distinguishes holding the equipped weapon (slot zero)
	// from a new switch request. This policy only acts on an equipped rocket.
	keep := make([]float64, 12)
	keep[0] = 1
	state := start
	for i := 0; i < 10; i++ {
		state, err = hNet(g.Step, append(state, keep...))
		if err != nil { return 0, err }
	}
	equipped := make([]float64, 12)
	equipped[7] = 1
	y, err := hNet(g.Head, append(append(start, state...), equipped...))
	if err != nil { return 0, err }
	return 1 / (1 + math.Exp(-y[3])), nil
}

func (g *rocketGate) deferRocket(o Observation, a Action) (bool, float64, error) {
	risk, err := g.blastRisk(o)
	if err != nil { return false, 0, err }
	if risk >= g.Probability { return true, risk, nil }
	if o.Geometry == nil || g.Clearance == 0 { return false, risk, nil }
	// Find the existing horizontal BSP probe nearest the proposed game yaw.
	diff, distance := math.Inf(1), math.Inf(1)
	for _, probe := range o.Geometry.Probes {
		d := math.Abs(math.Remainder(probe.YawOffset-a.YawDelta, 360))
		if d < diff { diff, distance = d, probe.RayDistance }
	}
	return distance < g.Clearance, risk, nil
}
