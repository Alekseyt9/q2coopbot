package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestTeammateMotionIsConditionalAndClearsOnNewObservation(t *testing.T) {
	nav := &quake.Navigator{Areas: []quake.Area{
		{},
		{Min: quake.Vec3{0, -10, 0}, Max: quake.Vec3{20, 10, 48}, Flags: 1},
		{Min: quake.Vec3{80, -10, 0}, Max: quake.Vec3{100, 10, 48}, Flags: 1},
		{Min: quake.Vec3{250, -10, 0}, Max: quake.Vec3{270, 10, 48}, Flags: 1},
		{Min: quake.Vec3{0, -10, 0}, Max: quake.Vec3{20, 10, 48}}, // not grounded
	}}
	p := &Planner{Nav: nav, World: World{Map: "test"}}
	last, age := quake.Vec3{10, 0, 24}, 1
	s := quake.Snapshot{Map: "test", Frame: 11, LastTeammate: &last,
		LastTeammateEntity: 2, TeammateAgeFrames: &age}
	p.update(s, "")
	if got := p.World.TeammateMotion; got == nil || got.Entity != 2 || got.Radius != 72 ||
		got.NearbyGroundAreas != 2 || got.TotalGroundAreas != 3 || len(got.AreaIDs) != 2 ||
		got.AreaIDs[0] != 1 || got.AreaIDs[1] != 2 || got.Method != "conditional_horizontal_radius" {
		t.Fatalf("initial motion footprint: %+v", got)
	}
	s.Frame, age = 16, 6
	s.Sounds = []quake.SoundEvent{{Entity: 2, Name: "jump.wav", Attenuation: 1}}
	p.update(s, "")
	if got := p.World.TeammateMotion; got == nil || got.Radius != 272 || got.NearbyGroundAreas != 3 {
		t.Fatalf("sound changed the movement footprint: %+v", got)
	}
	s.Teammate = &last
	s.Frame, age = 17, 0
	p.update(s, "")
	if p.World.TeammateMotion != nil {
		t.Fatal("fresh sighting kept stale motion footprint")
	}
	s.Teammate = nil
	s.LastTeammateEntity = 0
	s.Frame, age = 18, 1
	p.update(s, "")
	if p.World.TeammateMotion != nil {
		t.Fatal("invalidated player slot kept motion footprint")
	}
	s.LastTeammateEntity = 2
	s.Frame, age = 60, 41
	p.update(s, "")
	if p.World.TeammateMotion != nil {
		t.Fatal("expired observation kept motion footprint")
	}
}

func TestTeammateMotionOmitsTruncatedAreaList(t *testing.T) {
	areas := make([]quake.Area, 70)
	for id := 1; id < len(areas); id++ {
		areas[id] = quake.Area{Min: quake.Vec3{-10, -10, 0}, Max: quake.Vec3{10, 10, 48}, Flags: 1}
	}
	p := &Planner{Nav: &quake.Navigator{Areas: areas}}
	last, age := quake.Vec3{0, 0, 24}, 1
	p.updateTeammateMotion(quake.Snapshot{LastTeammate: &last, LastTeammateEntity: 2,
		TeammateAgeFrames: &age})
	if got := p.World.TeammateMotion; got == nil || got.NearbyGroundAreas != 69 || len(got.AreaIDs) != 0 {
		t.Fatalf("broad envelope exposed a truncated list: %+v", got)
	}
}
