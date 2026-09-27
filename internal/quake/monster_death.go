package quake

// Names are model directories from the network configstrings, not spawn
// classnames (for example, monster_chick uses the "bitch" model).
// Ranges come from baseq2 game/monster/*/*.h and the death mmove_t definitions.
// Unknown models remain eligible targets; solid alone cannot prove death.
var monsterDeathFrames = map[string][2]int{
	"soldier": {272, 474}, "infantry": {125, 178}, "gunner": {190, 200},
	"tank": {222, 253}, "parasite": {32, 38},
	"berserk": {223, 243}, "brain": {123, 145}, "bitch": {48, 82},
	"gladiatr": {61, 82}, "medic": {147, 176}, "mutant": {15, 33},
	"float": {104, 116}, "hover": {162, 172},
}

func monsterDeathAnimation(model string, frame int) bool {
	r, ok := monsterDeathFrames[model]
	return ok && frame >= r[0] && frame <= r[1]
}
