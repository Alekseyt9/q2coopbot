// Diagnostic only: observed-pose MG ray versus static BSP and observed bbox.
// This command packs geometry; it never loads or evaluates neural weights.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type rayGeometry struct {
	Known      bool       `json:"known"`
	Muzzle     quake.Vec3 `json:"muzzle"`
	Direction  quake.Vec3 `json:"direction"`
	Intersects bool       `json:"intersects"`
	Near       float64    `json:"near"`
	Far        float64    `json:"far"`
}

// slabInterval uses a forward ray and exact parallel-axis rejection. Distances
// are world units because the direction is normalized by construction.
func slabInterval(origin, direction, lower, upper quake.Vec3) (float64, float64, bool) {
	near, far := 0., math.Inf(1)
	for i, v := range direction {
		if math.Abs(v) < 1e-12 {
			if origin[i] < lower[i] || origin[i] > upper[i] {
				return 0, 0, false
			}
			continue
		}
		a, b := (lower[i]-origin[i])/v, (upper[i]-origin[i])/v
		near, far = math.Max(near, math.Min(a, b)), math.Min(far, math.Max(a, b))
		if far < near {
			return 0, 0, false
		}
	}
	if math.IsInf(far, 0) {
		return 0, 0, false
	}
	return near, far, true
}

func observedRay(x []float64, position quake.Vec3) (rayGeometry, error) {
	return observedRayForHand(x, position, 0)
}

func observedRayForHand(x []float64, position quake.Vec3, hand int) (rayGeometry, error) {
	if hand < 0 || hand > 2 {
		return rayGeometry{}, fmt.Errorf("invalid native hand")
	}
	// Reuse the versioned input/eligibility validation, without any learned score.
	g, err := policy.ObservedActionGeometryForVersion(x, policy.ActionGeometrySmoothVersion)
	if err != nil {
		return rayGeometry{}, err
	}
	if g[0] == 0 {
		return rayGeometry{}, nil
	}
	yaw, pitch := math.Atan2(x[6], x[7]), math.Atan2(x[8], x[9])
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	e, t := x[99:111], x[111:151]
	target := quake.Vec3{position[0] + 512*(cy*e[2]-sy*e[3]), position[1] + 512*(sy*e[2]+cy*e[3]), position[2] + 512*e[4]}
	eye := 22.
	if x[3] > 0 {
		eye = -2
	}
	var kick quake.Vec3
	if x[181] > 0 {
		for i := range kick {
			kick[i] = x[182+i] * 32 * math.Pi / 180
		}
	}
	ay, ap := yaw+x[164]*math.Pi+kick[1], pitch+x[165]*math.Pi+kick[0]
	right := quake.Vec3{-math.Sin(kick[2])*math.Sin(ap)*math.Cos(ay) + math.Cos(kick[2])*math.Sin(ay), -math.Sin(kick[2])*math.Sin(ap)*math.Sin(ay) - math.Cos(kick[2])*math.Cos(ay), -math.Sin(kick[2]) * math.Cos(ap)}
	r := rayGeometry{Known: true, Direction: quake.Vec3{math.Cos(ap) * math.Cos(ay), math.Cos(ap) * math.Sin(ay), -math.Sin(ap)}}
	lateral := 8.
	if hand == 1 {
		lateral = -8
	}
	if hand == 2 {
		lateral = 0
	}
	for i := range r.Muzzle {
		r.Muzzle[i] = position[i] + lateral*right[i]
	}
	r.Muzzle[2] += eye - 8
	hw := math.Max(1, t[37]*64)
	lower, upper := quake.Vec3{target[0] - hw, target[1] - hw, target[2] + t[38]*64}, quake.Vec3{target[0] + hw, target[1] + hw, target[2] + t[39]*64}
	r.Near, r.Far, r.Intersects = slabInterval(r.Muzzle, r.Direction, lower, upper)
	return r, nil
}

type result struct {
	Hand             int               `json:"hand"`
	Geometry         rayGeometry       `json:"geometry"`
	BSPSHA256        string            `json:"bsp_sha256"`
	StaticTrace      *quake.PointTrace `json:"static_trace,omitempty"`
	StaticBeforeBBox bool              `json:"static_before_bbox"`
	MuzzleTrace      quake.PointTrace  `json:"muzzle_trace"`
}

func main() {
	in := flag.String("input", "", "observed inputs JSONL")
	out := flag.String("out", "", "fresh output JSONL")
	assets := flag.String("assets", "", "existing baseq2 asset directory")
	hand := flag.Int("hand", 2, "native handedness: 0 right,1 left,2 centered (ConnectRequest)")
	flag.Parse()
	if err := run(*in, *out, *assets, *hand); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(input, output, assets string, hand int) error {
	if input == "" || output == "" || assets == "" {
		return fmt.Errorf("input, out and assets required")
	}
	f, err := os.Open(input)
	if err != nil {
		return err
	}
	defer f.Close()
	w, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer w.Close()
	b := bufio.NewWriter(w)
	enc := json.NewEncoder(b)
	maps := map[string]*quake.MapInfo{}
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 8*1024*1024)
	for s.Scan() {
		var row struct {
			Observation policy.Observation `json:"observation"`
			Sample      policy.Sample      `json:"sample"`
		}
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return err
		}
		name := row.Observation.Identity.Map
		m := maps[name]
		if m == nil {
			loaded, err := quake.LoadMap(assets, name)
			if err != nil {
				return err
			}
			if !loaded.HasCollision() {
				return fmt.Errorf("map has no collision")
			}
			m = &loaded
			maps[name] = m
		}
		features, err := policy.FeaturesForVersion(row.Observation, policy.EquippedWeaponFeatureVersion)
		if err != nil {
			return err
		}
		x, err := policy.GeometryActionInputs(features, row.Sample)
		if err != nil {
			return err
		}
		g, err := observedRayForHand(x, row.Observation.Position, hand)
		if err != nil {
			return err
		}
		r := result{Hand: hand, Geometry: g, BSPSHA256: m.BSPSHA256}
		if g.Known {
			r.MuzzleTrace = m.TraceProjectile(row.Observation.Position, g.Muzzle)
			if g.Intersects {
				end := g.Muzzle
				for i := range end {
					end[i] += g.Direction[i] * g.Near
				}
				tr := m.TraceProjectile(g.Muzzle, end)
				r.StaticTrace = &tr
				// BSP trace epsilon is 1/32 unit. Ignore a contact at the bbox
				// boundary within 1/8 unit, matching protocol position precision.
				r.StaticBeforeBBox = tr.StartSolid || (tr.Valid && (1-tr.Fraction)*g.Near > .125)
			}
		}
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	if err := s.Err(); err != nil {
		return err
	}
	return b.Flush()
}
