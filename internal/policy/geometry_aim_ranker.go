package policy

import (
	"fmt"
	"math"
	"reflect"
)

const GeometryAimRankerVersion = "combat_geometry_aim_ranker_v1"
const ExecutionGeometryAimRanker = "geometry_shot_rank_aim_v1"

type GeometryShotPredictor struct {
	Motor           *ActionEffectMotor `json:"motor,omitempty"`
	Kind            string             `json:"kind"`
	GeometryVersion string             `json:"geometry_version"`
	InputWidth      int                `json:"input_width"`
	CausalWidth     int                `json:"causal_width"`
	GeometryNames   []string           `json:"geometry_names"`
	Mean            []float64          `json:"mean"`
	Std             []float64          `json:"std"`
	Weight          []float64          `json:"weight"`
	Bias            float64            `json:"bias"`
	TrainedWeapon   string             `json:"trained_weapon"`
	DatasetSHA256   string             `json:"dataset_sha256"`
	Scope           string             `json:"scope"`
}
type GeometryAimRankerFile struct {
	Version   string                `json:"version"`
	Offsets   []float64             `json:"offset_degrees"`
	Predictor GeometryShotPredictor `json:"predictor"`
}
type GeometryAimRankSample struct {
	SourceLatent [2]float64 `json:"source_latent"`
	Candidate    int        `json:"candidate"`
	Candidates   int        `json:"candidates"`
	BeforeLogit  float64    `json:"before_logit"`
	AfterLogit   float64    `json:"after_logit"`
}

func validateGeometryAimRanker(f PPOFile) error {
	s := f.GeometryAimRanker
	if s == nil || s.Version != GeometryAimRankerVersion || !equippedPrefixFeatures(f.Features) ||
		!reflect.DeepEqual(s.Offsets, []float64{0, -5, -2.5, 2.5, 5}) {
		return fmt.Errorf("invalid geometry aim ranker contract")
	}
	p := s.Predictor
	if p.Kind != "combat_geometry_shot_predictor_v1" || (p.GeometryVersion != ActionGeometrySmoothVersion && p.GeometryVersion != ActionEffectGeometryVersion) || p.InputWidth != 214 || p.CausalWidth != 185 ||
		p.TrainedWeapon != "machinegun" || len(p.DatasetSHA256) != 64 || !reflect.DeepEqual(p.GeometryNames, ActionGeometryNamesForVersion(p.GeometryVersion)) ||
		len(p.Mean) != 214 || len(p.Std) != 214 || len(p.Weight) != 214 {
		return fmt.Errorf("invalid geometry predictor contract")
	}
	if p.GeometryVersion == ActionEffectGeometryVersion {
		if err := validateActionEffectMotor(p.Motor); err != nil {
			return err
		}
	} else if p.Motor != nil {
		return fmt.Errorf("motor outside predicted geometry version")
	}
	for _, values := range [][]float64{p.Mean, p.Std, p.Weight, {p.Bias}} {
		for _, v := range values {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
				return fmt.Errorf("invalid geometry predictor parameter")
			}
		}
	}
	for _, v := range p.Std {
		if v <= 0 {
			return fmt.Errorf("invalid geometry predictor normalization")
		}
	}
	return nil
}

// scoreGeometryAim evaluates learned parameters only during live Decide calls.
func scoreGeometryAim(p GeometryShotPredictor, causal []float64, timing [2]float64) (float64, error) {
	value, _, err := scoreGeometryAimWithGeometry(p, causal, timing)
	return value, err
}

// Same live-only forecast, with its causal geometry available to shared heads.
func scoreGeometryAimWithGeometry(p GeometryShotPredictor, causal []float64, timing [2]float64) (float64, []float64, error) {
	var g []float64
	var err error
	if p.Motor != nil {
		var forecast []float64
		forecast, err = forecastMotor(p.Motor, causal, timing)
		if err == nil {
			g, err = ProjectMotorGeometry(causal, forecast)
		}
	} else {
		g, err = ObservedActionGeometryForVersion(causal, p.GeometryVersion)
	}
	if err != nil {
		return 0, nil, err
	}
	x := append(append([]float64(nil), causal...), g...)
	value := p.Bias
	for i, w := range p.Weight {
		value += w * (x[i] - p.Mean[i]) / p.Std[i]
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, nil, fmt.Errorf("nonfinite geometry predictor score")
	}
	return value, g, nil
}

// Only the learned critic selects among nearby angles. No new random draws,
// target changes, weapon choice, movement controller or firing veto is added.
func (p *PPO) rankGeometryAim(o Observation, s *Sample) error {
	spec := p.file.GeometryAimRanker
	if spec == nil || equippedWeaponIndex(o.Weapon) != 4 || s.Target == 0 {
		return nil
	}
	features, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return err
	}
	base, err := GeometryActionInputs(features, *s)
	if err != nil {
		return err
	}
	observedVersion := spec.Predictor.GeometryVersion
	if spec.Predictor.Motor != nil {
		observedVersion = ActionGeometryCenteredVersion
	}
	geometry, err := ObservedActionGeometryForVersion(base, observedVersion)
	if err != nil {
		return err
	}
	if geometry[0] == 0 {
		return nil
	}
	timing := [2]float64{float64(o.AgeMS) / 300, float64(o.PreviousCommand.Msec) / 100}
	before, err := scoreGeometryAim(spec.Predictor, base, timing)
	if err != nil {
		return err
	}
	receipt := &GeometryAimRankSample{SourceLatent: [2]float64{s.Latent[2], s.Latent[3]}, Candidates: 25, BeforeLogit: before, AfterLogit: before}
	scale := 180.
	if s.AimMode == 1 {
		scale = PrecisionFineDegrees
	}
	best := *s
	index := 0
	for _, yaw := range spec.Offsets {
		for _, pitch := range spec.Offsets {
			if index == 0 {
				index++
				continue
			}
			candidate := *s
			causal := append([]float64(nil), base...)
			for axis, offset := range []float64{yaw, pitch} {
				ratio := math.Max(-1+1e-6, math.Min(1-1e-6, (base[164+axis]*180+offset)/scale))
				candidate.Latent[2+axis] = math.Atanh(ratio)
				causal[164+axis] = math.Tanh(candidate.Latent[2+axis]) * scale / 180
			}
			value, err := scoreGeometryAim(spec.Predictor, causal, timing)
			if err != nil {
				return err
			}
			if value > receipt.AfterLogit+1e-9 {
				receipt.AfterLogit = value
				receipt.Candidate = index
				best = candidate
			}
			index++
		}
	}
	s.Latent[2], s.Latent[3] = best.Latent[2], best.Latent[3]
	s.AimRanking = receipt
	return nil
}
