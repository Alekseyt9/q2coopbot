package aimquery

import (
	"math"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// Compute the native planar wish vector independently of the transformation.
func planarCommand(c quake.UserCmd) [2]float64 {
	y, p := float64(c.Yaw)*2*math.Pi/65536, float64(c.Pitch)*2*math.Pi/65536/3
	f, s := float64(c.Forward)*math.Cos(p), float64(c.Side)
	return [2]float64{f*math.Cos(y) + s*math.Sin(y), f*math.Sin(y) - s*math.Cos(y)}
}

func TestCoordinatedInputAcrossViewsAndBounds(t *testing.T) {
	for _, yaw := range []float64{-179, -90, 0, 45, 179} {
		for _, pitch := range []float64{-80, 0, 80} {
			for _, movement := range [][2]float64{{0, 0}, {.4, 0}, {0, .5}, {.3, -.4}, {1, 1}, {-1, 1}} {
				o := observation()
				o.OnGround = true
				o.ViewAngles[0] = int16(math.Round(pitch * 65536 / 360))
				o.ViewAngles[1] = int16(math.Round(yaw * 65536 / 360))
				a := policy.Action{Version: policy.ActionVersion, Identity: o.Identity, Forward: movement[0], Side: movement[1], YawDelta: 13, PitchDelta: -5, Vertical: "release", Attack: true}
				before := a
				q, err := QueryCoordinated(o, a)
				if err != nil || q == nil {
					t.Fatal(q, err)
				}
				old, _ := policy.Command(o, a, [3]int16{})
				newCmd, err := policy.Command(o, q.Action, [3]int16{})
				if err != nil {
					t.Fatal(err)
				}
				v, w := planarCommand(old), planarCommand(newCmd)
				for i := range v {
					if math.Abs(w[i]-v[i]*q.InputScale) > 1.01 {
						t.Fatalf("yaw=%v pitch=%v movement=%v old=%v new=%v scale=%v", yaw, pitch, movement, v, w, q.InputScale)
					}
				}
				if a != before || q.Action.Attack || q.Action.Vertical != "release" || q.Action.Weapon != "" || q.Version != CoordinatedVersion || q.InputScale <= 0 || q.InputScale > 1 {
					t.Fatal("invalid annotation", q)
				}
			}
		}
	}
}

func TestCoordinatedUnsupportedStates(t *testing.T) {
	o := observation()
	o.OnGround = true
	a := policy.Action{Version: policy.ActionVersion, Identity: o.Identity, Vertical: "release"}
	for _, kind := range []string{"air", "duck", "roll", "jump", "crouch"} {
		x, b := o, a
		switch kind {
		case "air":
			x.OnGround = false
		case "duck":
			x.Ducked = true
		case "roll":
			x.ViewAngles[2] = 1
		default:
			b.Vertical = kind
		}
		q, err := QueryCoordinated(x, b)
		if err != nil || q != nil {
			t.Fatal(kind, q, err)
		}
	}
	a.Forward = math.NaN()
	if _, err := QueryCoordinated(o, a); err == nil {
		t.Fatal("nonfinite input accepted")
	}
}
