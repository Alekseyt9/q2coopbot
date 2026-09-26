package bot

import (
	"path/filepath"
	"q2coopbot/internal/harness"
	"testing"
)

func TestTransitionWaitsForBothAcknowledgedStreams(t *testing.T) {
	s := botSessionFixture()
	actor := Client{sessionDefinition: &s, session: &harness.SessionRunner{Status: harness.SessionStatus{State: "waiting_map"}}, sessionMap: "base1", spawncount: 2, frameReady: true, lastMoveFrame: 50, latestFrame: 50, sessionPendingMap: "base2", scenarioResultPath: filepath.Join(t.TempDir(), "end.json")}
	observer := Client{sessionDefinition: &s, sessionMap: "base1", spawncount: 2, frameReady: true, lastMoveFrame: 49, latestFrame: 50, scenarioResultPath: actor.scenarioResultPath}
	if pause, ready, err := actor.sessionTransitionBarrier(); !pause || ready || err != nil {
		t.Fatal(pause, ready, err)
	}
	if pause, _, err := observer.sessionTransitionBarrier(); pause || err != nil {
		t.Fatal("observer must record terminal frame", pause, err)
	}
	observer.lastMoveFrame = 50
	if pause, _, err := observer.sessionTransitionBarrier(); !pause || err != nil || observer.sessionTransitionAcked {
		t.Fatal("unacknowledged observer released", pause, err)
	}
	observer.latestFrame = 51
	if _, _, err := observer.sessionTransitionBarrier(); err != nil || !observer.sessionTransitionAcked {
		t.Fatal(err)
	}
	if _, ready, err := actor.sessionTransitionBarrier(); ready || err != nil {
		t.Fatal("unacknowledged actor released", ready, err)
	}
	actor.latestFrame = 51
	if pause, ready, err := actor.sessionTransitionBarrier(); !pause || !ready || err != nil {
		t.Fatal(pause, ready, err)
	}
	actor.sessionPendingMap = ""
	if pause, ready, err := actor.sessionTransitionBarrier(); !pause || ready || err != nil {
		t.Fatal("stream resumed before new map", pause, ready, err)
	}
}
