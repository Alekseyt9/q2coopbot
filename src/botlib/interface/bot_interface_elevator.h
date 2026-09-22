#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_ELEVATOR_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_ELEVATOR_H

#include <stdbool.h>

#include "bot_state.h"

#ifdef __cplusplus
extern "C" {
#endif

bool BotInterface_CoopResultUsesElevator(
	const bot_client_state_t *state,
	const bot_moveresult_t *result);

bool BotInterface_CoopElevatorRoute(
	const bot_client_state_t *state,
	int *reach_number,
	int *source_area,
	int *destination_area);

int BotInterface_CoopElevatorGoalArea(
	const vec3_t player_origin,
	int sampled_area);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_ELEVATOR_H */
