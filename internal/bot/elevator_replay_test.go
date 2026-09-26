package bot

import (
	"bufio"
	"encoding/json"
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestRouteReplacementDropsOldElevator(t *testing.T) {
	goal := quake.Vec3{200, 0, 0}
	nav := &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 10}},
		{Min: quake.Vec3{190, -10, -10}, Max: quake.Vec3{210, 10, 10}},
	}, Edges: [][]quake.Edge{{}, {{To: 2, Start: quake.Vec3{20, 0, 0}, End: goal, Kind: 2, Cost: 10}}, nil}}
	p := &Planner{GameClock: true, Nav: nav, World: World{Map: "test"},
		elevator:   &elevatorRide{model: 37, toArea: 2, stage: "approach"},
		routeKnown: false, lastProgress: time.Unix(1, 0)}
	p.update(quake.Snapshot{Map: "test", Frame: 10, Health: 100, Teammate: &goal}, "")
	if p.elevator != nil || len(p.World.Route) != 2 || p.World.Route[0].Kind != 2 {
		t.Fatalf("replacement inherited elevator: ride=%+v route=%+v", p.elevator, p.World.Route)
	}
}

func TestElevatorLiveTraceAudit(t *testing.T) {
	path := os.Getenv("Q2_ELEVATOR_TRACE")
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if path == "" || root == "" {
		t.Skip("requires local trace and baseq2 assets")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p := &Planner{GameClock: true}
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 4096), 4<<20)
	for scan.Scan() {
		var s quake.Snapshot
		if err := json.Unmarshal(scan.Bytes(), &s); err != nil {
			t.Fatal(err)
		}
		p.update(s, root)
		p.command(quake.UserCmd{})
		if p.elevator != nil && (p.routeIndex >= len(p.route) || p.route[p.routeIndex].ElevatorPhase != "board" || p.route[p.routeIndex].Model != p.elevator.model || p.route[p.routeIndex].ToArea != p.elevator.toArea) {
			t.Fatalf("orphan elevator at frame %d: ride=%+v routeIndex=%d", s.Frame, p.elevator, p.routeIndex)
		}
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
}
