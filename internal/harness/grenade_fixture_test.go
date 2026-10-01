package harness

import "testing"

func TestGrenadeYawRequiresExplicitArm(t *testing.T) {
	s := fixture()
	s.Map = "base1"
	s.BotReleaseFrame = 60
	s.BotGrenadeYaw = 90
	if s.Validate() == nil {
		t.Fatal("Yaw accepted outside explicit arming fixture")
	}
	s.BotGrenadeArm = true
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	s.BotGrenadeYaw = 180
	if s.Validate() == nil {
		t.Fatal("Unsupported fixture yaw accepted")
	}
	s.BotGrenadeYaw = 90
	s.BotInvulnerable = true
	if s.Validate() == nil {
		t.Fatal("Masked-health grenade fixture accepted")
	}
}
