package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeFriendRelativeCrossing(t *testing.T) {
	friend := quake.Vec3{80, 60, 0}
	if got := grenadeFriendContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, emptyGrenadeTrace, friend, quake.Vec3{0, -200, 0}); got < 0.1 || got > 0.3 {
		t.Fatal(got)
	}
	if got := grenadeFriendContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, emptyGrenadeTrace, friend, quake.Vec3{0, 200, 0}); got != 0 {
		t.Fatal("Departing friend treated as crossing", got)
	}
	if got := grenadeFriendContact(quake.Vec3{}, quake.Vec3{400, 0, 200}, 800, func(_, _ quake.Vec3) quake.PointTrace { return quake.PointTrace{} }, friend, quake.Vec3{0, -200, 0}); got != 0 {
		t.Fatal("Unknown geometry certifies contact", got)
	}
}
