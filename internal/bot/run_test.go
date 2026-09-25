package bot

import "testing"

func TestParseTestTeleport(t *testing.T) {
	position, err := parseTestTeleport("-8,1408,-53")
	if err != nil || position != [3]float64{-8, 1408, -53} {
		t.Fatalf("position=%v err=%v", position, err)
	}
	for _, input := range []string{"0,1", "NaN,0,0", "40000,0,0"} {
		if _, err := parseTestTeleport(input); err == nil {
			t.Fatalf("accepted %q", input)
		}
	}
}

func TestTransitionMapArgumentUsesPreviousMapEntry(t *testing.T) {
	got, err := transitionMapArgument("base2", "base1")
	if err != nil || got != "base2 base1" {
		t.Fatalf("map argument = %q, %v", got, err)
	}
	if _, err := transitionMapArgument("base2", "base1;quit"); err == nil {
		t.Fatal("unsafe previous map name accepted")
	}
}
