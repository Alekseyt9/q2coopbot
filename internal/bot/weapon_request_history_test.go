package bot

import (
	"q2coopbot/internal/policy"
	"reflect"
	"testing"
)

func weaponRequestObservation(frame int) policy.Observation {
	return policy.Observation{Health: 100, Identity: policy.Identity{Map: "jail1", Spawncount: 1, Connection: 1, Actor: 1, Life: 1, Frame: frame}}
}

func TestWeaponRequestHistoryNextFrameAndImmutableObservations(t *testing.T) {
	h := weaponRequestHistory{}
	first := weaponRequestObservation(100)
	h.record(first.Identity, "Machinegun")
	h.enrich(&first)
	if first.PreviousWeaponRequest != nil {
		t.Fatal("same-frame intent leaked into observation")
	}
	next := weaponRequestObservation(101)
	h.enrich(&next)
	if next.PreviousWeaponRequest == nil || *next.PreviousWeaponRequest != "Machinegun" || *next.PreviousWeaponRequestAgeFrames != 1 {
		t.Fatal("missing own request history")
	}
	h.record(next.Identity, "Shotgun")
	if *next.PreviousWeaponRequest != "Machinegun" || *next.PreviousWeaponRequestAgeFrames != 1 {
		t.Fatal("cached observation mutated")
	}
	without := next
	without.PreviousWeaponRequest, without.PreviousWeaponRequestAgeFrames = nil, nil
	before, err := policy.FeaturesForVersion(without, policy.ThreatFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	after, err := policy.FeaturesForVersion(next, policy.ThreatFeatureVersion)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("own-request metadata changed legacy v9 features", err)
	}
	later := weaponRequestObservation(104)
	h.enrich(&later)
	if *later.PreviousWeaponRequest != "Shotgun" || *later.PreviousWeaponRequestAgeFrames != 3 {
		t.Fatal("wrong request age")
	}
}

func TestWeaponRequestHistoryResetAndUnknownCommands(t *testing.T) {
	for _, change := range []func(*policy.Observation){
		func(o *policy.Observation) { o.Identity.Map = "city1" },
		func(o *policy.Observation) { o.Identity.Spawncount++ },
		func(o *policy.Observation) { o.Identity.Connection++ },
		func(o *policy.Observation) { o.Identity.Actor++ },
		func(o *policy.Observation) { o.Identity.Life++ },
		func(o *policy.Observation) { o.Identity.Frame = 90 },
		func(o *policy.Observation) { o.Health = 0 },
	} {
		h := weaponRequestHistory{}
		initial := weaponRequestObservation(100)
		h.record(initial.Identity, "Machinegun")
		later := weaponRequestObservation(101)
		change(&later)
		h.enrich(&later)
		if h.name != "" || later.PreviousWeaponRequest != nil {
			t.Fatal("stale history survived identity/death reset")
		}
	}
	h := weaponRequestHistory{}
	o := weaponRequestObservation(100)
	for _, name := range []string{"", "weapnext", "modded", "use Shotgun"} {
		h.record(o.Identity, name)
	}
	o.Identity.Frame++
	h.enrich(&o)
	if o.PreviousWeaponRequest != nil {
		t.Fatal("invented requested weapon")
	}
}
