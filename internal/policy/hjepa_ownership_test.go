package policy

import "testing"

func TestHJEPAOwnershipIsExplicitAndModelBound(t *testing.T) {
	f := hTestModel()
	legacy, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	f.OwnershipScope = HJEPAFullEncounterScope
	full, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	if legacy.OwnsFullEncounter() || !full.OwnsFullEncounter() || legacy.Version() == full.Version() {
		t.Fatal("ownership must be an explicit, hash-bound model choice")
	}
	f.OwnershipScope = "always"
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("unknown ownership accepted")
	}
}
