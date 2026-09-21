#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_OBJECTIVE_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_OBJECTIVE_H

#include <stdbool.h>

#include "bot_state.h"

#ifdef __cplusplus
extern "C" {
#endif

bool BotInterface_CoopProbeControlRoute(const bot_client_state_t *state,
	const vec3_t player_origin,
	int *bot_area_out,
	int *player_area_out);

void BotInterface_CoopSetControlObjectivePhase(
	bot_client_state_t *state,
	bool objective_enabled,
	bot_coop_control_phase_t phase,
	int control_entity,
	int goal_area);

void BotInterface_CoopSetObjectivePhase(bot_client_state_t *state,
	bot_coop_objective_phase_t phase,
	int player_entity,
	int goal_area,
	int traveltype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_OBJECTIVE_H */
