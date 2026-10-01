package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeDynamicBodyUsesEachNativeTick(t *testing.T) {
	// The target crosses into the second segment. Freezing its initial body
	// would miss contact; freezing its final body would report it too early.
	first := grenadeBody{quake.Vec3{60, 60, 0}, quake.Vec3{-8, -8, -32}, quake.Vec3{8, 8, 40}}
	second := first
	second.origin[1] = 0
	f := simulateGrenadeWithBodies(quake.Vec3{}, quake.Vec3{400, 0, 200}, 1, 800, emptyGrenadeTrace, func(tick int) []grenadeBody {
		if tick == 1 {
			return []grenadeBody{first}
		}
		return []grenadeBody{second}
	}, nil)
	if f.Event != "damageable_contact" || f.Seconds != .2 || f.End[0] != 52 || f.impactVelocity != (quake.Vec3{400, 0, 40}) {
		t.Fatal(f)
	}
	static := grenadeFlight(quake.Vec3{}, quake.Vec3{400, 0, 200}, 1, 800, emptyGrenadeTrace, []grenadeBody{first})
	if static.Event == "damageable_contact" {
		t.Fatal("Static body unexpectedly hit", static)
	}
}
