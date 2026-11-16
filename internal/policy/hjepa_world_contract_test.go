package policy

import "testing"

func TestHJEPAWorldInputContractPreservesLegacyAndRejectsUnknown(t *testing.T) {
	f := hTestModel()
	legacy, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	f.WorldActionSpace = "proposed_primitive_v1"
	explicit, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	a, err := legacy.Decide(o)
	if err != nil {
		t.Fatal(err)
	}
	b, err := explicit.Decide(o)
	if err != nil || a != b {
		t.Fatal("declaration changed unchanged networks", err)
	}
	f.WorldActionSpace = "unknown"
	if _, err = LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("unknown predictor input contract admitted")
	}
}
