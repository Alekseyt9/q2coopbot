package policy

import (
	"math"
	"reflect"
	"testing"
)

func TestHJEPAFireTemperatureChangesOnlyFireDistribution(t *testing.T) {
	cold, warm := hSamplingModel(), hSamplingModel()
	cold.PrimitiveSampling.Temperature, warm.PrimitiveSampling.Temperature = .25, .25
	cold.Actor[len(cold.Actor)-1].Bias[4], warm.Actor[len(warm.Actor)-1].Bias[4] = 2, 2
	temperature := 2.
	warm.PrimitiveSampling.FireTemperature = &temperature
	p, err := LoadHJEPA(hWrite(t, cold))
	if err != nil {
		t.Fatal(err)
	}
	q, err := LoadHJEPA(hWrite(t, warm))
	if err != nil {
		t.Fatal(err)
	}
	o, changes := testObservation(), 0
	for i := 0; i < 128; i++ {
		o.Identity.Frame++
		if _, err := p.Decide(o); err != nil {
			t.Fatal(err)
		}
		a := *p.LastSample()
		if _, err := q.Decide(o); err != nil {
			t.Fatal(err)
		}
		b := *q.LastSample()
		if a.Latent != b.Latent || a.Vertical != b.Vertical || a.SamplingSeed != b.SamplingSeed {
			t.Fatal("fire exploration changed movement, aim, pose or seed")
		}
		if a.Attack != b.Attack {
			changes++
		}
		lp, err := q.PrimitiveLogProbability(o, b)
		if err != nil || math.Abs(lp-b.LogProbability) > 1e-12 {
			t.Fatal("warm density replay mismatch", err)
		}
	}
	if changes == 0 {
		t.Fatal("fire temperature never changed a draw")
	}
	// An explicit legacy temperature must reproduce both draws AND densities.
	temperature = .25
	q, err = LoadHJEPA(hWrite(t, warm))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		o.Identity.Frame++
		_, _ = p.Decide(o)
		_, _ = q.Decide(o)
		if !reflect.DeepEqual(p.LastSample(), q.LastSample()) {
			t.Fatal("legacy default changed")
		}
	}
	for _, v := range []float64{0, -.1, 5.1, math.NaN(), math.Inf(1)} {
		warm.PrimitiveSampling.FireTemperature = &v
		if validateHJEPAActorSampling(warm) == nil {
			t.Fatal("invalid fire temperature admitted", v)
		}
	}
}
