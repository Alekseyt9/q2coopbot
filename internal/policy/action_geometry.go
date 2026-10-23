package policy

import (
	"fmt"
	"math"
)

const ActionGeometryVersion = "combat_observed_action_geometry_v1"
const ActionGeometrySmoothVersion = "combat_observed_action_geometry_v2"
const ActionGeometryCenteredVersion = "combat_observed_action_geometry_v3"
const ActionGeometryWidth = 29

func ActionGeometryNames() []string {
	names := []string{"known", "inverse_distance", "relative_transverse_angular_speed"}
	for _, origin := range []string{"eye", "muzzle"} {
		for _, name := range []string{"yaw_sin", "yaw_cos", "pitch_sin", "pitch_cos", "yaw_error_pi", "pitch_error_pi", "yaw_error_over_halfwidth", "pitch_error_over_halfheight", "yaw_outside_pi", "pitch_outside_pi", "closest_bbox_distance", "ray_slab_margin", "ray_bbox_intersection"} {
			names = append(names, origin+"_"+name)
		}
	}
	return names
}

func ActionGeometryNamesForVersion(version string) []string {
	names := ActionGeometryNames()
	if version == ActionGeometrySmoothVersion || version == ActionGeometryCenteredVersion || version == ActionEffectGeometryVersion {
		names[15] = "eye_ray_bbox_soft_intersection"
		names[28] = "muzzle_ray_bbox_soft_intersection"
	}
	return names
}

// GeometryActionInputs packs observations and the chosen action without NN inference.
func GeometryActionInputs(features []float64, sample Sample) ([]float64, error) {
	x, err := conditionalFireInputs(features, sample)
	if err != nil {
		return nil, err
	}
	return append(x, features[810:814]...), nil
}

func geometryNorm(v [3]float64) float64   { return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) }
func geometryDot(a, b [3]float64) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// ObservedActionGeometry mirrors combat_action_geometry.features. Geometry only:
// no predictor, trace queries, future state, decisions or native outcome inputs.
func ObservedActionGeometry(x []float64) ([]float64, error) {
	return ObservedActionGeometryForVersion(x, ActionGeometryVersion)
}

func ObservedActionGeometryForVersion(x []float64, version string) ([]float64, error) {
	if version != ActionGeometryVersion && version != ActionGeometrySmoothVersion && version != ActionGeometryCenteredVersion {
		return nil, fmt.Errorf("unknown action geometry version")
	}
	if len(x) != 185 {
		return nil, fmt.Errorf("geometry requires input185")
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite geometry input")
		}
	}
	if x[151] != 0 || x[99] <= 0 || x[147] <= 0 {
		return make([]float64, ActionGeometryWidth), nil
	}
	e, t := x[99:111], x[111:151]
	yaw, pitch := math.Atan2(x[6], x[7]), math.Atan2(x[8], x[9])
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	position := [3]float64{512 * (cy*e[2] - sy*e[3]), 512 * (sy*e[2] + cy*e[3]), 512 * e[4]}
	hw, bottom, top := math.Max(1, t[37]*64), t[38]*64, t[39]*64
	height := math.Min(math.Max(22, bottom+8), top-8)
	if top-bottom < 16 {
		height = (top + bottom) / 2
	}
	eye := 22.
	if x[3] > 0 {
		eye = -2
	}
	var kick [3]float64
	if x[181] > 0 {
		for i := range kick {
			kick[i] = x[182+i] * 32 * math.Pi / 180
		}
	}
	ay, ap := yaw+x[164]*math.Pi+kick[1], pitch+x[165]*math.Pi+kick[0]
	right := [3]float64{-math.Sin(kick[2])*math.Sin(ap)*math.Cos(ay) + math.Cos(kick[2])*math.Sin(ay), -math.Sin(kick[2])*math.Sin(ap)*math.Sin(ay) - math.Cos(kick[2])*math.Cos(ay), -math.Sin(kick[2]) * math.Cos(ap)}
	origins := [2][3]float64{{0, 0, eye}, {8 * right[0], 8 * right[1], eye - 8 + 8*right[2]}}
	if version == ActionGeometryCenteredVersion {
		// ConnectRequest uses hand=2 (CENTER_HANDED), so native
		// P_ProjectSource zeros the lateral muzzle offset.
		origins[1] = [3]float64{0, 0, eye - 8}
	}
	direction := [3]float64{math.Cos(ap) * math.Cos(ay), math.Cos(ap) * math.Sin(ay), -math.Sin(ap)}
	var velocity [3]float64
	if e[5] > 0 {
		velocity = [3]float64{400 * (cy*e[6] - sy*e[7] - x[10]), 400 * (sy*e[6] + cy*e[7] - x[11]), 400 * (e[8] - x[12])}
	}
	cross := [3]float64{direction[1]*velocity[2] - direction[2]*velocity[1], direction[2]*velocity[0] - direction[0]*velocity[2], direction[0]*velocity[1] - direction[1]*velocity[0]}
	norm := geometryNorm(position)
	out := []float64{1, 1 / (1 + norm/64), math.Min(4, geometryNorm(cross)/math.Max(1, norm)) / 4}
	wrap := func(v float64) float64 { return math.Atan2(math.Sin(v), math.Cos(v)) }
	for _, origin := range origins {
		r := [3]float64{position[0] - origin[0], position[1] - origin[1], position[2] - origin[2]}
		distance := math.Max(1, math.Hypot(r[0], r[1]))
		ty, tp := math.Atan2(r[1], r[0]), -math.Atan2(r[2]+height, distance)
		dy, dp := wrap(ty-ay), wrap(tp-ap)
		halfwidth := .001
		for _, i := range []float64{-1, 1} {
			for _, j := range []float64{-1, 1} {
				halfwidth = math.Max(halfwidth, math.Abs(wrap(math.Atan2(r[1]+j*hw, r[0]+i*hw)-ty)))
			}
		}
		pt, pb := -math.Atan2(r[2]+top, distance), -math.Atan2(r[2]+bottom, distance)
		halfheight := math.Max(.001, math.Abs(pb-pt)/2)
		lower, upper := [3]float64{r[0] - hw, r[1] - hw, r[2] + bottom}, [3]float64{r[0] + hw, r[1] + hw, r[2] + top}
		near, far := 0., math.Inf(1)
		var center, half [3]float64
		for i := range direction {
			d := direction[i]
			if math.Abs(d) < 1e-6 {
				if d < 0 {
					d = -1e-6
				} else {
					d = 1e-6
				}
			}
			a, b := lower[i]/d, upper[i]/d
			near = math.Max(near, math.Min(a, b))
			far = math.Min(far, math.Max(a, b))
			center[i], half[i] = (lower[i]+upper[i])/2, (upper[i]-lower[i])/2
		}
		dot := math.Max(0, geometryDot(center, direction))
		var excess [3]float64
		for i := range excess {
			excess[i] = math.Max(0, math.Abs(dot*direction[i]-center[i])-half[i])
		}
		outside := geometryNorm(excess) / math.Max(1, math.Max(half[0], math.Max(half[1], half[2])))
		gap := far - near
		hit := 0.
		if gap >= 0 {
			hit = 1
		}
		if version == ActionGeometrySmoothVersion || version == ActionGeometryCenteredVersion {
			hit = math.Max(0, math.Min(1, (gap+1)/2))
		}
		out = append(out, math.Sin(dy), math.Cos(dy), math.Sin(dp), math.Cos(dp), dy/math.Pi, dp/math.Pi,
			math.Min(8, math.Abs(dy)/halfwidth)/8, math.Min(8, math.Abs(dp)/halfheight)/8,
			math.Max(0, math.Abs(dy)-halfwidth)/math.Pi, (math.Max(0, pt-ap)+math.Max(0, ap-pb))/math.Pi,
			math.Min(4, outside)/4, math.Max(-4, math.Min(4, gap/distance))/4, hit)
	}
	for _, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite geometry output")
		}
	}
	return out, nil
}
