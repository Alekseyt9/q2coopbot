package bot

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

// Enumerate native recoil extrema and spread corners independently of the cone
// calculation. Place the partner on each possible ray, including after water.
func TestHitscanFriendNativePathCoverage(t *testing.T) {
	basis := func(yaw, pitch float64) (quake.Vec3, quake.Vec3, quake.Vec3) {
		return quake.Vec3{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)},
			quake.Vec3{math.Sin(yaw), -math.Cos(yaw), 0},
			quake.Vec3{math.Sin(pitch) * math.Cos(yaw), math.Sin(pitch) * math.Sin(yaw), math.Cos(pitch)}
	}
	shot := func(f, right, up quake.Vec3, h, v float64) quake.Vec3 {
		d := quake.Vec3{}
		for i := range d {
			d[i] = 8192*f[i] + h*right[i] + v*up[i]
		}
		norm := math.Sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
		for i := range d {
			d[i] /= norm
		}
		return d
	}
	cases := 0
	for _, weapon := range []string{"Machinegun", "Shotgun", "Super Shotgun"} {
		for _, cmd := range []quake.UserCmd{{Msec: 100}, {Msec: 100, Yaw: 16384}, {Msec: 100, Pitch: -5461}, {Msec: 100, Pitch: 10922}} {
			yaw := float64(cmd.Yaw) * 2 * math.Pi / 65536
			pitch := float64(cmd.Pitch) * 2 * math.Pi / 65536
			for kick := 0; kick <= 9; kick++ {
				for _, yawKick := range []float64{-.7, 0, .7} {
					h := 500.0
					rp, ry := pitch, yaw
					if weapon == "Machinegun" {
						h = 300
						rp -= float64(kick) * 1.5 * math.Pi / 180
						ry += yawKick * math.Pi / 180
					}
					if weapon == "Super Shotgun" {
						h = 1000
						ry += yawKick / .7 * 5 * math.Pi / 180
					}
					f, right, up := basis(ry, rp)
					for _, a := range []float64{-1, 0, 1} {
						for _, b := range []float64{-1, 0, 1} {
							d := shot(f, right, up, a*h, b*500)
							wf, wr, wu := basis(math.Atan2(d[1], d[0]), math.Atan2(-d[2], math.Hypot(d[0], d[1])))
							water := shot(wf, wr, wu, a*h*2, b*1000)
							for _, wet := range []bool{false, true} {
								origin := quake.Vec3{8 * right[0], 8 * right[1], 14}
								dir := d
								distance := 4000.0
								if wet {
									for i := range origin {
										origin[i] += 4000 * d[i]
									}
									dir = water
									distance = 7000
								}
								mate := origin
								for i := range mate {
									mate[i] += distance * dir[i]
								}
								s := quake.Snapshot{Weapon: weapon, Teammate: &mate}
								if !hitscanTeammateRisk(s, cmd) {
									t.Fatalf("unprotected native path %s kick%d yaw%v a%v b%v wet%v mate%v", weapon, kick, yawKick, a, b, wet, mate)
								}
								cases++
							}
						}
					}
				}
			}
		}
	}
	t.Logf("%d native path samples protected", cases)
}

func TestHitscanFriendRecordedCommands(t *testing.T) {
	path := os.Getenv("Q2_LIVE_GUARD_TRACE")
	if path == "" {
		t.Skip("optional real companion trace")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 65536), 4*1024*1024)
	proposed, released, retained, idle := 0, 0, 0, 0
	last := ""
	for scanner.Scan() {
		var row struct {
			quake.Snapshot
			Goal       string `json:"goal"`
			Spawncount int    `json:"spawncount"`
			Connection int    `json:"connection"`
			Combat     struct {
				Selection struct {
					Owner            string        `json:"owner"`
					CandidateCommand quake.UserCmd `json:"candidate_command"`
					Interventions    []struct {
						Reason string `json:"reason"`
					} `json:"interventions"`
				} `json:"selection"`
			} `json:"combat_policy"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		key := fmt.Sprintf("%d/%d/%s/%d", row.Connection, row.Spawncount, row.Map, row.Frame)
		if key == last {
			continue
		}
		last = key
		if row.Goal == "wait_for_teammate" && row.Teammate == nil && len(row.Enemies) == 0 {
			idle++
		} else {
			idle = 0
		}
		if idle >= 200 {
			break
		}
		if row.Combat.Selection.Owner != "provider" {
			continue
		}
		blocked := false
		for _, change := range row.Combat.Selection.Interventions {
			if change.Reason == "machinegun_partner_guard" {
				blocked = true
			}
		}
		if !blocked {
			continue
		}
		proposed++
		if hitscanTeammateRisk(row.Snapshot, row.Combat.Selection.CandidateCommand) {
			retained++
		} else {
			released++
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if proposed == 0 || released == 0 || retained == 0 {
		t.Fatalf("insufficient trace coverage total%d released%d retained%d", proposed, released, retained)
	}
	t.Logf("offline counterfactual only: old blanket blocks%d, new cone retained%d released%d; no native safety claim", proposed, retained, released)
}

func TestHitscanFriendGeometry(t *testing.T) {
	for _, weapon := range []string{"Machinegun", "Shotgun"} {
		for _, tc := range []struct {
			name string
			mate quake.Vec3
			want bool
		}{
			{"behind", quake.Vec3{-400, 0, 0}, false},
			{"offside", quake.Vec3{400, 1000, 0}, false},
			{"near_muzzle", quake.Vec3{0, 20, 0}, true},
			{"central_beyond_old_1024", quake.Vec3{4000, 0, 0}, true},
			{"spread_off_central_ray", quake.Vec3{4000, 180, 0}, true},
			{"water_extended_range", quake.Vec3{12000, 0, 0}, true},
			{"above_cone", quake.Vec3{300, 0, 2000}, false},
		} {
			t.Run(weapon+"/"+tc.name, func(t *testing.T) {
				s := quake.Snapshot{Weapon: weapon, Teammate: &tc.mate}
				if got := hitscanTeammateRisk(s, quake.UserCmd{Msec: 100}); got != tc.want {
					t.Fatalf("risk=%v want %v", got, tc.want)
				}
			})
		}
	}
}

func TestHitscanFriendRecentPositionAndView(t *testing.T) {
	point := quake.Vec3{-400, 0, 0}
	age := 0
	s := quake.Snapshot{Weapon: "Machinegun", LastTeammate: &point, TeammateAgeFrames: &age}
	cmd := quake.UserCmd{Msec: 100}
	if hitscanTeammateRisk(s, cmd) {
		t.Fatal("fresh player behind incorrectly blocks")
	}
	age = 10
	if !hitscanTeammateRisk(s, cmd) {
		t.Fatal("recent position uncertainty ignored")
	}
	// Current observation supersedes the stale position.
	current := quake.Vec3{-1000, 0, 0}
	s.Teammate = &current
	if hitscanTeammateRisk(s, cmd) {
		t.Fatal("stale player used despite current observation")
	}
	point = quake.Vec3{0, 4000, 0}
	s.Teammate = &point
	s.DeltaAngles[1] = 16384
	if !hitscanTeammateRisk(s, cmd) {
		t.Fatal("delta yaw ignored")
	}
	s.DeltaAngles[1] = 0
	cmd.Yaw = 16384
	if !hitscanTeammateRisk(s, cmd) {
		t.Fatal("command yaw ignored")
	}
}

func TestHitscanFriendGuardIntegration(t *testing.T) {
	c := policyClient(t, "learned")
	for _, weapon := range []string{"Machinegun", "Shotgun"} {
		for _, tc := range []struct {
			mate    quake.Vec3
			blocked bool
		}{
			{quake.Vec3{-400, -224, 24.125}, false},
			{quake.Vec3{2000, -224, 24.125}, true},
			{quake.Vec3{32, 1000, 24.125}, false},
		} {
			s := c.planner.World.Snapshot
			s.Weapon = weapon
			s.Teammate = &tc.mate
			cmd := quake.UserCmd{Buttons: 1, Msec: 100}
			got, _ := c.planner.guardDirectCombat(s, cmd)
			if (got.Buttons&1 == 0) != tc.blocked {
				t.Fatalf("%s mate%v cmd%+v", weapon, tc.mate, got)
			}
			if got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch || got.Forward != cmd.Forward || got.Side != cmd.Side {
				t.Fatal("fire guard changed another action")
			}
		}
	}
}
