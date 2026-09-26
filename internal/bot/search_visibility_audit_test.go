package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

// Real-map counterexamples prove geometric misses, not server reacquisition.
func TestSearchVisibilityAudit(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local baseq2 assets")
	}
	m, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(filepath.Join(root, "maps", "base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ self, last, view, witness quake.Vec3 }{
		{quake.Vec3{-172.2761993408203, 619.2762451171875, -95}, quake.Vec3{-235.2761993408203, 619.2762451171875, -95}, quake.Vec3{-96.52063751220703, 605.8857421875, -103}, quake.Vec3{-398.1352233886719, 890.0330200195312, -103}},
		{quake.Vec3{-8.60000228881836, 1035.2000732421875, -199}, quake.Vec3{54.39999771118164, 1035.2000732421875, -199}, quake.Vec3{-163.49090576171875, 1010.3624877929688, -199}, quake.Vec3{207.875, 1111.2181396484375, -231}},
		{quake.Vec3{-941.3428344726562, 1656.80419921875, -23}, quake.Vec3{-878.3428344726562, 1656.80419921875, -23}, quake.Vec3{-907.7332761875, 1572.8001708984375, -23}, quake.Vec3{-1175.0589599609375, 1709.7042236328125, -23}},
	} {
		if m.ClearShot(searchEye(tc.self), searchEye(tc.witness)) || !m.ClearShot(searchEye(tc.view), searchEye(tc.witness)) {
			t.Fatal("counterexample geometry changed")
		}
		p := &Planner{Nav: nav, World: World{Geometry: &m, GeometryStatus: "ready"}, searchApproachStarted: true}
		age := 1
		s := quake.Snapshot{Self: tc.self, LastTeammate: &tc.last, LastTeammateEntity: 2, Frame: 10, TeammateAgeFrames: &age, Health: 100, OnGround: true}
		_, kind, ok := p.hiddenTeammateGoal(s)
		if !ok || kind != "probe_last_seen" || p.searchAttempt == nil {
			t.Fatal("useful geometric viewpoint rejected")
		}
		v := p.searchAttempt.Visibility
		if v.MaxNewlyVisible != 0 || v.AuditMaxGain <= 0 || v.noNewCoverage() {
			t.Fatalf("miss not detected: %+v", v)
		}
		t.Logf("self=%v audit=%+v", tc.self, v)
	}
}
