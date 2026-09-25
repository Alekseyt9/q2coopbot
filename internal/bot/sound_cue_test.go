package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestTeammateSoundCueIsActivityOnlyAndExpires(t *testing.T) {
	p := &Planner{World: World{Map: "test"}}
	last, age := quake.Vec3{100, 0, 24}, 1
	s := quake.Snapshot{Map: "test", Frame: 20, Self: quake.Vec3{0, 0, 24},
		LastTeammate: &last, LastTeammateEntity: 2, TeammateAgeFrames: &age,
		Sounds: []quake.SoundEvent{
			{Entity: 3, Name: "*jump1.wav", Attenuation: 1},
			{Entity: 2, Name: "global.wav", Attenuation: 0},
			{Entity: 2, Name: "*jump1.wav", Attenuation: 1},
		}}
	p.update(s, "")
	if cue := p.World.TeammateSound; cue == nil || cue.Entity != 2 || cue.Name != "*jump1.wav" || cue.Frame != 20 || cue.AgeFrames != 0 {
		t.Fatalf("confirmed sound not recorded: %+v", cue)
	}
	if p.World.Goal != "wait_for_teammate" || p.hasGoal || p.World.Snapshot.Teammate != nil {
		t.Fatalf("sound was treated as a visible location: %+v", p.World)
	}
	s.Sounds = nil
	s.Frame, age = 30, 11
	p.update(s, "")
	if p.World.TeammateSound == nil || p.World.TeammateSound.AgeFrames != 10 {
		t.Fatalf("cue should remain for 10 frames: %+v", p.World.TeammateSound)
	}
	s.Frame, age = 31, 12
	p.update(s, "")
	if p.World.TeammateSound != nil {
		t.Fatalf("expired cue survived: %+v", p.World.TeammateSound)
	}
	s.Frame, age = 32, 13
	s.Sounds = []quake.SoundEvent{{Entity: 2, Name: "*pain100_1.wav", Attenuation: 1}}
	p.update(s, "")
	if p.World.TeammateSound == nil {
		t.Fatal("second sound did not refresh cue")
	}
	s.LastTeammateEntity = 0 // slot changed without a new visual confirmation
	s.LastTeammate = nil
	s.Frame = 33
	p.update(s, "")
	if p.World.TeammateSound != nil {
		t.Fatal("unconfirmed slot kept sound cue")
	}
}

func TestSoundAreasRequireExplicitPositionOfConfirmedPlayer(t *testing.T) {
	p := &Planner{World: World{Map: "test"}, Nav: &quake.Navigator{Areas: []quake.Area{
		{},
		{Min: quake.Vec3{80, -20, 0}, Max: quake.Vec3{120, 20, 48}, Center: quake.Vec3{100, 0, 24}, Flags: 1},
		{Min: quake.Vec3{100, -20, 0}, Max: quake.Vec3{140, 20, 48}, Center: quake.Vec3{120, 0, 24}, Flags: 1},
		{Min: quake.Vec3{300, -20, 0}, Max: quake.Vec3{340, 20, 48}, Center: quake.Vec3{320, 0, 24}, Flags: 1},
		{Min: quake.Vec3{80, -20, 0}, Max: quake.Vec3{120, 20, 48}, Center: quake.Vec3{100, 0, 24}},
		{Min: quake.Vec3{80, -20, 100}, Max: quake.Vec3{120, 20, 148}, Center: quake.Vec3{100, 0, 124}, Flags: 1},
	}}}
	last, age := quake.Vec3{0, 0, 24}, 3
	position := quake.Vec3{104, 0, 24}
	s := quake.Snapshot{Map: "test", Frame: 20, LastTeammate: &last,
		LastTeammateEntity: 2, TeammateAgeFrames: &age,
		Sounds: []quake.SoundEvent{
			{Entity: 3, Name: "other.wav", Attenuation: 1, Position: &position},
			{Entity: 2, Name: "global.wav", Position: &position},
			{Entity: 2, Name: "jump.wav", Attenuation: 1, Position: &position},
		}}
	p.update(s, "")
	if cue := p.World.TeammateSound; cue == nil || len(cue.SourceAreas) != 2 ||
		cue.SourceAreas[0] != 1 || cue.SourceAreas[1] != 2 {
		t.Fatalf("explicit sound area candidates = %+v", cue)
	}
	s.Frame++
	s.Sounds = []quake.SoundEvent{{Entity: 2, Name: "jump.wav", Attenuation: 1}}
	p.update(s, "")
	if cue := p.World.TeammateSound; cue == nil || len(cue.SourceAreas) != 0 {
		t.Fatalf("entity-only sound retained an area hypothesis: %+v", cue)
	}
}
