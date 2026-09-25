package bot

import "testing"

func TestTransitionMapArgumentUsesPreviousMapEntry(t *testing.T) {
	got, err := transitionMapArgument("base2", "base1")
	if err != nil || got != "base2 base1" {
		t.Fatalf("map argument = %q, %v", got, err)
	}
	if _, err := transitionMapArgument("base2", "base1;quit"); err == nil {
		t.Fatal("unsafe previous map name accepted")
	}
}
