package policy

import (
	"reflect"
	"testing"
)

func TestHJEPARootFireOverrideExpiresAndNestedSaveKeepsCommonFuture(t *testing.T) {
	path := hWrite(t, hSamplingModel())
	source, _ := LoadHJEPA(path)
	o := testObservation()
	o.Identity.Life, o.Identity.Frame = 1, 180
	state, err := source.CaptureSamplingCheckpoint(o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	native := o.Identity
	native.Frame, native.Spawncount = 1, native.Spawncount+7
	observed := o
	observed.Identity = native
	selected := map[bool]int64{}
	for stream := int64(1); stream <= 128 && len(selected) < 2; stream++ {
		p, _ := LoadHJEPA(path)
		branch, err := state.WithStream(stream, HJEPASamplingRootFireScope)
		if err != nil || p.RestoreSamplingCheckpoint(branch, native) != nil {
			t.Fatal(err)
		}
		a, err := p.Decide(observed)
		if err != nil {
			t.Fatal(err)
		}
		selected[a.Attack] = stream
	}
	if len(selected) != 2 {
		t.Fatal("root action enumeration not exercised")
	}
	var branches []*HJEPA
	for _, stream := range []int64{0, selected[false], selected[true]} {
		p, _ := LoadHJEPA(path)
		branch, _ := state.WithStream(stream, HJEPASamplingRootFireScope)
		if err := p.RestoreSamplingCheckpoint(branch, native); err != nil {
			t.Fatal(err)
		}
		branches = append(branches, p)
	}
	for step := 0; step < 64; step++ {
		original := o
		original.Identity.Frame += step
		want, err := source.Decide(original)
		if err != nil {
			t.Fatal(err)
		}
		obs := original
		obs.Identity = native
		obs.Identity.Frame += step
		for index, p := range branches {
			got, err := p.Decide(obs)
			if err != nil {
				t.Fatal(err)
			}
			want.Identity = obs.Identity
			if step > 0 || index == 0 {
				if got != want || !reflect.DeepEqual(p.LastSample(), source.LastSample()) {
					t.Fatal("root override changed common future or stream zero", step, index)
				}
			} else {
				withoutFire := got
				withoutFire.Attack = want.Attack
				if withoutFire != want || p.LastSample().Latent != source.LastSample().Latent {
					t.Fatal("root override changed non-fire proposal")
				}
			}
		}
	}
	next := native
	next.Frame = 65
	nested, err := branches[1].CaptureSamplingCheckpoint(next)
	if err != nil || nested.Version != HJEPASamplingCheckpointVersion || nested.Scope != "" || nested.Stream != 0 || nested.Source.Frame != 244 {
		t.Fatal("past root override would be reapplied after nested save", nested, err)
	}
	p, _ := LoadHJEPA(path)
	fresh := native
	fresh.Spawncount++
	if err := p.RestoreSamplingCheckpoint(nested, fresh); err != nil {
		t.Fatal(err)
	}
	original, loaded := o, o
	original.Identity.Frame = 244
	loaded.Identity = fresh
	want, err := source.Decide(original)
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Decide(loaded)
	want.Identity = loaded.Identity
	if err != nil || got != want || !reflect.DeepEqual(p.LastSample(), source.LastSample()) {
		t.Fatal("nested future draw changed", err)
	}
}
