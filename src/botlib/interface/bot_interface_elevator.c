#include "botlib/aas/aas_local.h"
#include "bot_interface_elevator.h"

bool BotInterface_CoopResultUsesElevator(const bot_client_state_t *state,
	const bot_moveresult_t *result)
{
	bot_movestate_t *move_state;

	if (result != NULL &&
		(((result->traveltype & TRAVELTYPE_MASK) == TRAVEL_ELEVATOR) ||
		 result->type == RESULTTYPE_ELEVATORUP))
	{
		return true;
	}
	if (state == NULL || state->move_handle <= 0 ||
		aasworld.reachability == NULL)
	{
		return false;
	}

	move_state = BotMoveStateFromHandle(state->move_handle);
	if (move_state == NULL || move_state->lastreachnum <= 0 ||
		move_state->lastreachnum >= aasworld.numReachability)
	{
		return false;
	}
	return (aasworld.reachability[move_state->lastreachnum].traveltype &
		TRAVELTYPE_MASK) == TRAVEL_ELEVATOR;
}

bool BotInterface_CoopElevatorRoute(const bot_client_state_t *state,
	int *reach_number,
	int *source_area,
	int *destination_area)
{
	bot_movestate_t *move_state;
	int reach;

	if (state == NULL || state->move_handle <= 0 ||
		aasworld.reachability == NULL)
	{
		return false;
	}
	move_state = BotMoveStateFromHandle(state->move_handle);
	if (move_state == NULL)
	{
		return false;
	}
	reach = move_state->lastreachnum;
	if (reach <= 0 || reach >= aasworld.numReachability ||
		(aasworld.reachability[reach].traveltype & TRAVELTYPE_MASK) !=
			TRAVEL_ELEVATOR)
	{
		return false;
	}
	if (reach_number != NULL)
	{
		*reach_number = reach;
	}
	if (destination_area != NULL)
	{
		*destination_area = aasworld.reachability[reach].areanum;
	}
	if (source_area != NULL)
	{
		*source_area = aasworld.reachabilityFromArea != NULL
			? aasworld.reachabilityFromArea[reach]
			: AAS_PointAreaNum(state->last_client_update.origin);
	}
	return true;
}
