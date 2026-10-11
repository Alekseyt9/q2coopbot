package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

// Fixed commands exercise ownership and execution limits, never a network.
type weaponRecoveryProbe struct{ head string }

func (p weaponRecoveryProbe) Version() string           { return "weapon_recovery_probe" }
func (p weaponRecoveryProbe) WeaponHeadVersion() string { return p.head }
func (p weaponRecoveryProbe) Decide(o policy.Observation) (policy.Action, error) {
	return policy.Action{Version: policy.ActionVersion, Identity: o.Identity, Forward: .2, Side: .1,
		YawDelta: 30, Attack: true, Vertical: "release", Weapon: "Machinegun"}, nil
}

func TestWeaponRecoveryPreservesProviderControlAndUnsupportedFireGuard(t *testing.T) {
	for _, campaign := range []bool{false, true} {
		c := policyClient(t, "learned")
		c.combatControl.provider = weaponRecoveryProbe{head: policy.WeaponHeadVersion}
		c.combatControl.campaignEvaluation = campaign
		if !campaign {
			c.testSynchronous = true
			c.testWeaponSwitchFixture = "parasite_weapons"
		}
		s := &c.planner.World.Snapshot
		s.Weapon = "models/weapons/v_handgr/tris.md2"
		s.InventoryKnown, s.InventoryAgeFrames = true, 1
		s.Inventory = []quake.InventoryItem{{Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 20}, {Name: "Grenades", Count: 10}}
		cmd, proposed, selection, direct := c.combatCommand(c.combatObservation(time.Now()), time.Now())
		if !direct || selection.Owner != "provider" || selection.Fallback != "" || selection.Candidate.Weapon != "Machinegun" {
			t.Fatal("equipped grenades stole the policy's replacement request", campaign, selection)
		}
		if proposed.Buttons&1 == 0 || cmd.Buttons&1 != 0 || cmd.Yaw != proposed.Yaw || cmd.Pitch != proposed.Pitch {
			t.Fatal("unsupported fire guard or policy aim changed", proposed, cmd)
		}
		blocked := false
		for _, intervention := range selection.Interventions {
			blocked = blocked || intervention.Component == "attack" && intervention.Reason == "pilot_fixed_weapon"
		}
		if !blocked {
			t.Fatal("unsupported grenade firing was not reported")
		}
	}
}

func TestWeaponRecoveryRequiresDeclaredHeadAndExplicitContext(t *testing.T) {
	for _, scenario := range []string{"no_head", "wrong_head", "unscoped"} {
		c := policyClient(t, "learned")
		c.testSynchronous = true
		c.testWeaponSwitchFixture = "parasite_weapons"
		c.planner.World.Snapshot.Weapon = "models/weapons/v_handgr/tris.md2"
		switch scenario {
		case "wrong_head":
			c.combatControl.provider = weaponRecoveryProbe{head: "unknown"}
		case "unscoped":
			c.combatControl.provider = weaponRecoveryProbe{head: policy.WeaponHeadVersion}
			c.testWeaponSwitchFixture = ""
		}
		_, _, selection, direct := c.combatCommand(c.combatObservation(time.Now()), time.Now())
		if direct || selection.Fallback != "pilot_equip_not_ready" {
			t.Fatal("weapon recovery escaped its capability/context", scenario, selection)
		}
	}
}
