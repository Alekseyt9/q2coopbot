package bot

import (
	"context"
	"strings"
	"testing"
)

func TestCombatCaptureCannotMislabelLLMOrUnpacedActions(t *testing.T) {
	for _, cfg := range []Config{
		{CombatCapture: true},
		{CombatCapture: true, FramePaced: true},
		{CombatCapture: true, FramePaced: true, TracePath: "unused", System1Model: "teacher-llm"},
		{CombatCapture: true, FramePaced: true, TracePath: "unused", System2Model: "planner-llm"},
	} {
		err := Run(context.Background(), cfg)
		if err == nil || !strings.Contains(err.Error(), "rules combat capture requires") {
			t.Fatalf("invalid rules capture reached session startup: %v", err)
		}
	}
}
