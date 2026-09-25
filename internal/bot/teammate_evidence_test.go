package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestTeammateEvidenceKeepsSoundsSeparateFromReacquiredPosition(t *testing.T) {
	p := &Planner{World: World{Map: "test"}}
	last, age0 := quake.Vec3{0, 0, 24}, 0
	s := quake.Snapshot{Map: "test", Frame: 10, Teammate: &last, TeammateEntity: 2,
		LastTeammate: &last, LastTeammateEntity: 2, TeammateAgeFrames: &age0}
	p.update(s, "")
	s.Frame, s.Teammate = 11, nil
	age1 := 1
	s.TeammateAgeFrames = &age1
	s.Sounds = []quake.SoundEvent{
		{Entity: 3, Name: "other.wav", Attenuation: 1},
		{Entity: 2, Name: "global.wav"},
		{Entity: 2, Name: "jump.wav", Attenuation: 1},
	}
	p.update(s, "")
	if got := p.World.TeammateEvidence; got == nil || got.ActivitySounds != 1 || got.ActivityFrames != 1 ||
		got.LocationStatus != "unknown" || got.SourceOutsideNominal || p.World.Goal != "wait_for_teammate" {
		t.Fatalf("entity-only sound implied a location: %+v", got)
	}
	position := quake.Vec3{300, 0, 24}
	s.Frame = 14
	age4 := 4
	s.TeammateAgeFrames = &age4
	s.Sounds = []quake.SoundEvent{{Entity: 2, Name: "remote.wav", Attenuation: 1, Position: &position}}
	p.update(s, "")
	p.update(s, "") // a repeated update of the same frame cannot add evidence
	if got := p.World.TeammateEvidence; got == nil || got.ActivitySounds != 2 || got.ActivityFrames != 2 ||
		!got.SourceOutsideNominal || got.SourceConflictFrame != 14 || got.LocationStatus != "unknown" {
		t.Fatalf("repeated sound accounting or source comparison: %+v", got)
	}
	s.Frame, s.Teammate = 15, &position
	age0After := 0
	s.TeammateAgeFrames = &age0After
	s.Sounds = nil
	p.update(s, "")
	if got := p.World.TeammateEvidence; got == nil || !got.Reacquired || !got.VisualOutsideNominal ||
		got.LocationStatus != "observed" || got.ActivitySounds != 2 || got.ActivityFrames != 2 ||
		got.ObservedDistance != 300 || got.NominalRadius != 232 {
		t.Fatalf("reacquired position did not flag nominal movement mismatch: %+v", got)
	}
	s.Frame = 16
	p.update(s, "")
	if p.World.TeammateEvidence != nil {
		t.Fatal("old evidence survived a fresh visible frame")
	}
}
