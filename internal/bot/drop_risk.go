package bot

import "math"

// Dry upper estimate using server playerstate gravity. P_FallingDamage uses the previous
// server tick's vertical velocity squared, scaled by .0001. Include a full
// 100 ms gravity tick as a margin; do not credit armour or water mitigation.
func estimatedDropDamage(height, verticalSpeed float64, gravity int16) int {
	if gravity <= 0 || math.IsNaN(height) || math.IsInf(height, 0) || math.IsNaN(verticalSpeed) || math.IsInf(verticalSpeed, 0) || height < 0 {
		return 32767
	}
	g := float64(gravity)
	speed := math.Sqrt(verticalSpeed*verticalSpeed+2*g*height) + g*.1
	delta := speed * speed * .0001
	if math.IsInf(delta, 0) || delta > 65534 {
		return 32767
	}
	if delta <= 30 {
		return 0
	}
	return max(1, int(math.Ceil((delta-30)/2)))
}

func affordableDrop(health int16, damage int) bool {
	if health <= 0 || damage < 0 {
		return false
	}
	if damage == 0 {
		return true
	}
	return damage <= 20 && damage <= int(health)/3 && int(health)-damage >= 25
}
