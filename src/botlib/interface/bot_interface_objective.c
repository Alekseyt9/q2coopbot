#include "botlib/aas/aas_local.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "bot_interface_objective.h"

static const char *BotInterface_CoopControlPhaseName(
	bot_coop_control_phase_t phase)
{
	switch (phase)
	{
		case BOT_COOP_CONTROL_NAVIGATE:
			return "NAVIGATE_CONTROL";
		case BOT_COOP_CONTROL_WAIT_PLAYER:
			return "WAIT_FOR_PLAYER";
		case BOT_COOP_CONTROL_COMPLETE:
			return "COMPLETE";
		case BOT_COOP_CONTROL_RETURN_PATH:
			return "RETURN_TO_PATH";
		case BOT_COOP_CONTROL_RETRY:
			return "RETRY";
		case BOT_COOP_CONTROL_NONE:
		default:
			return "NONE";
	}
}

static const char *BotInterface_CoopObjectivePhaseName(
	bot_coop_objective_phase_t phase)
{
	switch (phase)
	{
		case BOT_COOP_OBJECTIVE_APPROACH:
			return "APPROACH";
		case BOT_COOP_OBJECTIVE_WAIT_ELEVATOR:
			return "WAIT_ELEVATOR";
		case BOT_COOP_OBJECTIVE_TRAVEL_ELEVATOR:
			return "TRAVEL_ELEVATOR";
		case BOT_COOP_OBJECTIVE_RETRY:
			return "RETRY";
		case BOT_COOP_OBJECTIVE_REACQUIRE:
			return "REACQUIRE";
		case BOT_COOP_OBJECTIVE_NONE:
		default:
			return "NONE";
	}
}

bool BotInterface_CoopProbeControlRoute(const bot_client_state_t *state,
	const vec3_t player_origin,
	int *bot_area_out,
	int *player_area_out)
{
	int bot_area;
	int player_area;
	vec3_t route_origin;

	if (state == NULL || player_origin == NULL)
	{
		return false;
	}
	bot_area = AAS_PointAreaNum(state->last_client_update.origin);
	player_area = AAS_PointAreaNum(player_origin);
	if (bot_area <= 0 || player_area <= 0)
	{
		return false;
	}
	if (bot_area_out != NULL)
	{
		*bot_area_out = bot_area;
	}
	if (player_area_out != NULL)
	{
		*player_area_out = player_area;
	}
	if (bot_area == player_area)
	{
		return true;
	}

	VectorCopy(state->last_client_update.origin, route_origin);
	return AAS_AreaTravelTimeToGoalArea(
		bot_area,
		route_origin,
		player_area,
		TFL_DEFAULT) > 0;
}

void BotInterface_CoopSetControlObjectivePhase(
	bot_client_state_t *state,
	bool objective_enabled,
	bot_coop_control_phase_t phase,
	int control_entity,
	int goal_area)
{
	bot_coop_control_phase_t previous_phase;
	int previous_entity;
	int previous_goal_area;

	if (state == NULL || !objective_enabled)
	{
		return;
	}
	previous_phase = state->coop_control_phase;
	previous_entity = state->coop_control_entity;
	previous_goal_area = state->coop_control_goal_area;
	state->coop_control_entity = control_entity;
	state->coop_control_goal_area = goal_area;
	if (previous_phase == phase && previous_entity == control_entity &&
		previous_goal_area == goal_area)
	{
		return;
	}
	state->coop_control_phase = phase;
	state->coop_control_started = AAS_Time();
	if (phase == BOT_COOP_CONTROL_NAVIGATE ||
		phase == BOT_COOP_CONTROL_NONE ||
		phase == BOT_COOP_CONTROL_RETRY)
	{
		state->coop_control_route_confirmed = false;
	}
	if (LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_objective client=%d objective=OPEN_PATH phase=%s "
			"control_entity=%d goal_area=%d",
			state->client_number,
			BotInterface_CoopControlPhaseName(phase),
			control_entity,
			goal_area);
	}
}

void BotInterface_CoopSetObjectivePhase(bot_client_state_t *state,
	bot_coop_objective_phase_t phase,
	int player_entity,
	int goal_area,
	int traveltype)
{
	bot_coop_objective_phase_t previous_phase;

	if (state == NULL)
	{
		return;
	}
	previous_phase = state->coop_objective_phase;
	state->coop_objective_goal_area = goal_area;
	if (phase == BOT_COOP_OBJECTIVE_RETRY)
	{
		state->coop_objective_retries += 1;
	}
	if (previous_phase == phase)
	{
		return;
	}
	state->coop_objective_phase = phase;
	state->coop_objective_started = AAS_Time();
	if (LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_objective client=%d objective=REGROUP phase=%s "
			"player=%d goal_area=%d traveltype=%d retries=%d",
			state->client_number,
			BotInterface_CoopObjectivePhaseName(phase),
			player_entity,
			goal_area,
			traveltype,
			state->coop_objective_retries);
	}
}
