package harness

import (
	"math"
	"q2coopbot/internal/quake"
)

func checkLastPlayerReturn(rows []Trace, death, respawn, recovery int) string {
	var player *quake.Vec3
	for _, row := range rows {
		if row.Frame < death && row.Teammate != nil {
			player = row.Teammate
		}
	}
	if player == nil {
		return "last_player_not_observed"
	}
	returns := 0
	arrived := false
	for _, row := range rows {
		if row.Frame <= respawn || row.Frame > respawn+recovery {
			continue
		}
		if row.Goal == "regroup_after_respawn" {
			if row.Teammate != nil || row.LastTeammate == nil || row.GoalPoint == nil {
				return "last_player_return_not_verified"
			}
			for axis := range *player {
				if math.Abs((*row.GoalPoint)[axis]-(*player)[axis]) > .125 || math.Abs((*row.LastTeammate)[axis]-(*player)[axis]) > .125 {
					return "wrong_last_player_target"
				}
			}
			returns++
		}
		if returns >= 3 && row.Teammate == nil && row.Self != nil && row.Health != nil && *row.Health > 0 && quake.Horizontal(*row.Self, *player) <= 64 && math.Abs((*row.Self)[2]-(*player)[2]) <= 40 {
			arrived = true
		}
	}
	if returns < 3 || !arrived {
		return "last_player_arrival_not_verified"
	}
	return ""
}
