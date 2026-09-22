#include <math.h>

#include "botlib/aas/aas_local.h"
#include "bot_interface_elevator.h"

int BotInterface_CoopElevatorGoalArea(const vec3_t player_origin,
	int sampled_area)
{
	vec3_t mins;
	vec3_t maxs;
	int areas[64];
	int area_count;
	int best_area;
	float best_vertical_delta;
	int area_index;

	if (player_origin == NULL || sampled_area <= 0 ||
		!aasworld.loaded || aasworld.reachability == NULL)
	{
		return sampled_area;
	}

	/*
	 * AAS_PointAreaNum follows the BSP tree and can return the large lift
	 * volume (base2 area 806) even when the player is standing on the upper
	 * platform (area 751).  Use the live player bbox to retain the more
	 * specific elevator destination area when it overlaps the sampled volume.
	 */
	VectorSet(mins, player_origin[0] - 16.0f,
		player_origin[1] - 16.0f, player_origin[2] - 24.0f);
	VectorSet(maxs, player_origin[0] + 16.0f,
		player_origin[1] + 16.0f, player_origin[2] + 32.0f);
	area_count = AAS_BBoxAreas(mins, maxs, areas,
		(int)(sizeof(areas) / sizeof(areas[0])));
	best_area = sampled_area;
	best_vertical_delta = 0.0f;

	for (area_index = 0; area_index < area_count; ++area_index)
	{
		int candidate = areas[area_index];
		int reach;
		aas_areainfo_t info;
		float vertical_delta;

		if (candidate <= 0 || candidate == sampled_area ||
			!AAS_AreaInfo(candidate, &info) ||
			(info.presencetype & PRESENCE_NORMAL) == 0)
		{
			continue;
		}

		for (reach = 1; reach < aasworld.numReachability; ++reach)
		{
			if ((aasworld.reachability[reach].traveltype &
				TRAVELTYPE_MASK) != TRAVEL_ELEVATOR ||
				aasworld.reachability[reach].areanum != candidate)
			{
				continue;
			}
			vertical_delta = fabsf(info.center[2] - player_origin[2]);
			if (best_area == sampled_area ||
				vertical_delta < best_vertical_delta)
			{
				best_area = candidate;
				best_vertical_delta = vertical_delta;
			}
			break;
		}
	}

	return best_area;
}

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
