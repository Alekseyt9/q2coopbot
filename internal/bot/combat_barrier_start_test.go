package bot

import "testing"

func TestSynchronousCombatBudgetStartsAfterPreparation(t *testing.T) {
	for _, synchronous := range []bool{false, true} {
		c := &Client{testSynchronous: synchronous, firstMoveFrame: 20, lastMoveFrame: 97, latestFrame: 98}
		c.beginTestCombat()
		if !c.testCombatGo || c.testCombatGoFrame != 98 || c.lastMoveFrame != 97 {
			t.Fatal("bad barrier transition")
		}
		want := 20
		if synchronous {
			want = -1
		}
		if c.firstMoveFrame != want {
			t.Fatal("wrong budget start")
		}
	}
}
