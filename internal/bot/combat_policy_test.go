package bot

import (
	"os"
	"testing"
	"time"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type probeProvider struct {
	panicNow   bool
	wrongFrame bool
}

func (p probeProvider) Version() string { return "test_probe" }
func (p probeProvider) Decide(o policy.Observation) (policy.Action, error) {
	if p.panicNow {
		panic("diagnostic failure")
	}
	a := policy.Action{Version: policy.ActionVersion, Identity: o.Identity, Forward: .2, Side: .1, YawDelta: 30, Attack: false, Vertical: "release"}
	if p.wrongFrame {
		a.Identity.Frame--
	}
	return a, nil
}

func policyClient(t *testing.T, mode string) *Client {
	t.Helper()
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 20, Health: 100, Weapon: "Blaster", OnGround: true, Self: quake.Vec3{32, -224, 24.125}, Enemies: []quake.Object{{ID: 300, Origin: quake.Vec3{200, -224, 24.125}, ClearShot: &clear}}}
	p := &Planner{World: World{Map: s.Map, Geometry: &g, GeometryStatus: "ready", Snapshot: s, Updated: time.Now()}, Nav: &quake.Navigator{}}
	return &Client{decoder: &quake.Decoder{PlayerNumber: 1}, planner: p, latestFrame: s.Frame, combatControl: combatControl{mode: mode, provider: probeProvider{}}}
}

func TestDirectProviderControlsAnglesWithoutTacticalAssistance(t *testing.T) {
	c := policyClient(t, "learned")
	o := c.combatObservation(time.Now())
	cmd, proposed, sel, direct := c.combatCommand(o, time.Now())
	if !direct || sel.Owner != "provider" || proposed.Forward != 80 || proposed.Side != 40 || proposed.Buttons != 0 || cmd.Yaw != proposed.Yaw || cmd.Pitch != proposed.Pitch || c.planner.World.Command.AimSource != "policy" || c.planner.World.Command.AimEntity != 0 {
		t.Fatal("rules retargeted or replaced provider", cmd, proposed, sel, c.planner.World.Command)
	}
}

func TestShadowAndFailuresKeepRulesOwner(t *testing.T) {
	for _, tc := range []struct {
		mode        string
		provider    probeProvider
		wantFailure bool
	}{{"learned-shadow", probeProvider{}, false}, {"learned", probeProvider{wrongFrame: true}, true}, {"learned", probeProvider{panicNow: true}, true}} {
		c := policyClient(t, tc.mode)
		c.combatControl.provider = tc.provider
		o := c.combatObservation(time.Now())
		cmd, _, sel, direct := c.combatCommand(o, time.Now())
		if direct || sel.Owner != "rules" || tc.wantFailure && sel.Fallback == "" || !tc.wantFailure && sel.CandidateCommand == nil {
			t.Fatal(sel, direct)
		}
		if tc.mode == "learned-shadow" {
			baseline := policyClient(t, "rules")
			obs := baseline.combatObservation(time.Now())
			want, _, _, _ := baseline.combatCommand(obs, time.Now())
			if cmd != want {
				t.Fatal("shadow changed rules command", cmd, want)
			}
		}
	}
}

func TestPolicyRayNeverReceivesPerfectAim(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	cmd := quake.UserCmd{Yaw: 16384, Buttons: 1, Msec: 100}
	got, changes := c.planner.guardDirectCombat(s, cmd)
	if got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch || got.Buttons != cmd.Buttons {
		t.Fatal("guard corrected policy aim or timing of a harmless miss", got, changes)
	}
	partner := quake.Vec3{32, -160, 24.125}
	s.Teammate = &partner
	got, changes = c.planner.guardDirectCombat(s, cmd)
	if got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch || got.Buttons != 0 || len(changes) == 0 {
		t.Fatal("partner ray not suppressed independently of aim", got, changes)
	}
}
