#include <math.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/ai_move/bot_move.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "botlib/common/l_utils.h"
#include "botlib/ea/ea_local.h"
#include "bot_interface_area.h"
#include "bot_interface_battle.h"
#include "bot_interface_combat.h"
#include "bot_interface_coop_overlay.h"
#include "bot_interface_coop_role.h"
#include "bot_interface_coop_state.h"
#include "bot_interface_elevator.h"
#include "bot_interface_map.h"
#include "bot_interface_objective.h"
#include "bot_interface_runtime.h"
#include "bot_state.h"

bool BotAI_ApplyCoopHardLeash(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t direction;
	float hard_leash;
	float distance;
	float vertical_separation;
	int player_entity;
	int player_area;
	int bot_area;
	int current_player_area;
	int known_player_area;
	float elevator_wait_timeout;
	float elevator_travel_timeout;
	bot_goal_t goal;
	bot_moveresult_t result;
	bool elevator_result;
	bool elevator_route_valid;
	bool elevator_travel_timeout_hit;
	bool was_elevator_waiting;
	bool needs_regroup;
	bot_coop_objective_phase_t previous_phase;
	int logged_traveltype;
	int elevator_reach;
	int elevator_source_area;
	int elevator_destination_area;
	int status;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_leash") == 0.0f)
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		return false;
	}

	VectorSubtract(player_origin, state->last_client_update.origin, direction);
	distance = sqrtf(DotProduct(direction, direction));
	hard_leash = LibVarGetValue("coopbot_hard_leash");
	current_player_area = AAS_PointAreaNum(player_origin);
	bot_area = AAS_PointAreaNum(state->last_client_update.origin);
	vertical_separation = fabsf(player_origin[2] -
		state->last_client_update.origin[2]);
	/*
	 * A player can be inside the lift volume, or on its moving brush, for a
	 * frame in which the point-area query returns zero. Preserve the last
	 * valid area so that a vertical split cannot turn into ordinary follow
	 * merely because the mover temporarily hides the AAS sample.
	 */
	if (current_player_area > 0)
	{
		state->coop_player_area = current_player_area;
		VectorCopy(player_origin, state->coop_player_origin);
	}
	known_player_area = current_player_area > 0
		? current_player_area : state->coop_player_area;
	/* A nearby player on another vertical AAS layer still requires regroup. */
	needs_regroup = hard_leash > 0.0f &&
		(distance > hard_leash ||
			(bot_area > 0 && known_player_area > 0 &&
				bot_area != known_player_area &&
				vertical_separation >= 64.0f) ||
			(current_player_area <= 0 && known_player_area > 0 &&
				vertical_separation >= 64.0f));
	previous_phase = state->coop_objective_phase;
	if (!needs_regroup)
	{
		if (state->coop_player_goal_valid)
		{
			if ((previous_phase == BOT_COOP_OBJECTIVE_WAIT_ELEVATOR ||
				previous_phase == BOT_COOP_OBJECTIVE_TRAVEL_ELEVATOR) &&
				LibVarGetValue("coopbot_log") >= 1.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_elevator_reacquired client=%d player=%d "
					"phase=ARRIVED bot_area=%d player_area=%d distance=%.1f",
					state->client_number, player_entity, bot_area,
					current_player_area, distance);
			}
			BotInterface_CoopSetObjectivePhase(state,
				BOT_COOP_OBJECTIVE_REACQUIRE,
				player_entity,
				current_player_area,
				0);
			if (LibVarGetValue("coopbot_log") >= 1.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_regroup_complete client=%d player=%d "
					"distance=%.1f vertical=%.1f bot_area=%d player_area=%d",
					state->client_number, player_entity, distance,
					vertical_separation, bot_area, current_player_area);
			}
			state->coop_player_goal_valid = false;
			state->coop_elevator_wait_started = 0.0f;
			state->coop_elevator_wait_area = 0;
			state->coop_elevator_travel_started = 0.0f;
			state->coop_elevator_travel_area = 0;
		}
		return false;
	}

	/* A destination area may be terminal and have no outgoing reachability. */
	if (current_player_area > 0)
	{
		state->coop_player_area = current_player_area;
		VectorCopy(player_origin, state->coop_player_origin);
		state->coop_player_goal_valid = true;
	}
	else if (known_player_area > 0)
	{
		/* Keep the current player origin, but route to the last valid area. */
		state->coop_player_area = known_player_area;
		VectorCopy(player_origin, state->coop_player_origin);
		state->coop_player_goal_valid = true;
	}
	if (!state->coop_player_goal_valid)
	{
		return false;
	}
	player_area = state->coop_player_area;
	if (state->coop_objective_phase == BOT_COOP_OBJECTIVE_NONE ||
		state->coop_objective_phase == BOT_COOP_OBJECTIVE_REACQUIRE)
	{
		if (state->coop_objective_phase == BOT_COOP_OBJECTIVE_REACQUIRE)
		{
			state->coop_objective_retries = 0;
		}
			BotInterface_CoopSetObjectivePhase(state,
			BOT_COOP_OBJECTIVE_APPROACH,
			player_entity,
			player_area,
			0);
	}

	memset(&goal, 0, sizeof(goal));
	goal.entitynum = player_entity;
	goal.areanum = player_area;
	VectorCopy(state->coop_player_origin, goal.origin);
	VectorSet(goal.mins, -16.0f, -16.0f, -24.0f);
	VectorSet(goal.maxs, 16.0f, 16.0f, 32.0f);

	status = BotInterface_PrepareMoveState(state, thinktime);
	if (status != BLERR_NOERROR)
	{
		return false;
	}

	BotClearMoveResult(&result);
	BotMoveToGoalHandle(&result,
		state->move_handle,
		&goal,
		BotAI_LongTermGoalTravelFlags(state));
	elevator_result = BotInterface_CoopResultUsesElevator(state, &result);
	elevator_reach = 0;
	elevator_source_area = 0;
	elevator_destination_area = 0;
	elevator_route_valid = elevator_result &&
		BotInterface_CoopElevatorRoute(state,
			&elevator_reach,
			&elevator_source_area,
			&elevator_destination_area);
	was_elevator_waiting = state->coop_objective_phase ==
		BOT_COOP_OBJECTIVE_WAIT_ELEVATOR;
	logged_traveltype = elevator_result ? TRAVEL_ELEVATOR : result.traveltype;
	elevator_wait_timeout = LibVarGetValue("coopbot_elevator_wait_timeout");
	if (elevator_wait_timeout <= 0.0f)
	{
		elevator_wait_timeout = 15.0f;
	}
	elevator_travel_timeout = LibVarGetValue("coopbot_elevator_travel_timeout");
	if (elevator_travel_timeout <= 0.0f)
	{
		elevator_travel_timeout = 30.0f;
	}
	elevator_travel_timeout_hit = false;
	if (elevator_result &&
		(result.flags & MOVERESULT_WAITING) != 0)
	{
		/* Waiting for the platform is a separate bounded phase. */
		state->coop_elevator_travel_started = 0.0f;
		state->coop_elevator_travel_area = 0;
			BotInterface_CoopSetObjectivePhase(state,
			BOT_COOP_OBJECTIVE_WAIT_ELEVATOR,
			player_entity,
			player_area,
			TRAVEL_ELEVATOR);
		if (state->coop_elevator_wait_area != player_area ||
			state->coop_elevator_wait_started <= 0.0f)
		{
			state->coop_elevator_wait_area = player_area;
			state->coop_elevator_wait_started = AAS_Time();
		}
		else if (AAS_Time() - state->coop_elevator_wait_started >
			elevator_wait_timeout)
		{
			result.failure = 1;
			BotResetAvoidReachHandle(state->move_handle);
			BotResetMoveStateHandle(state->move_handle);
			state->coop_elevator_wait_started = 0.0f;
			state->coop_elevator_wait_area = 0;
			state->coop_elevator_travel_started = 0.0f;
			state->coop_elevator_travel_area = 0;
			if (LibVarGetValue("coopbot_log") >= 2.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_elevator_failed client=%d player=%d reason=timeout timeout=%.1f",
					state->client_number, player_entity, elevator_wait_timeout);
			}
		}
	}
	else
	{
		if (elevator_result && !result.failure && !result.blocked)
		{
			if (was_elevator_waiting &&
				LibVarGetValue("coopbot_log") >= 1.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_elevator_boarded client=%d player=%d "
					"phase=TRAVEL goal_area=%d",
					state->client_number, player_entity, player_area);
			}
			BotInterface_CoopSetObjectivePhase(state,
				BOT_COOP_OBJECTIVE_TRAVEL_ELEVATOR,
				player_entity,
				player_area,
				TRAVEL_ELEVATOR);
			if (state->coop_elevator_travel_area != player_area ||
				state->coop_elevator_travel_started <= 0.0f)
			{
				state->coop_elevator_travel_area = player_area;
				state->coop_elevator_travel_started = AAS_Time();
			}
			else if (AAS_Time() - state->coop_elevator_travel_started >
				elevator_travel_timeout)
			{
				result.failure = 1;
				elevator_travel_timeout_hit = true;
				BotResetAvoidReachHandle(state->move_handle);
				BotResetMoveStateHandle(state->move_handle);
				state->coop_elevator_travel_started = 0.0f;
				state->coop_elevator_travel_area = 0;
				if (LibVarGetValue("coopbot_log") >= 2.0f)
				{
					BotLib_LogWriteTimeStamped(
						"coopbot_elevator_failed client=%d player=%d "
						"reason=travel_timeout timeout=%.1f",
						state->client_number, player_entity,
						elevator_travel_timeout);
				}
			}
		}
		else if (!result.failure && !result.blocked)
		{
			BotInterface_CoopSetObjectivePhase(state,
				BOT_COOP_OBJECTIVE_APPROACH,
				player_entity,
				player_area,
				result.traveltype);
		}
		state->coop_elevator_wait_started = 0.0f;
		state->coop_elevator_wait_area = 0;
		if (!elevator_result || result.failure || result.blocked)
		{
			state->coop_elevator_travel_started = 0.0f;
			state->coop_elevator_travel_area = 0;
		}
	}
	if (elevator_result &&
		(result.failure || result.blocked))
	{
			BotInterface_CoopSetObjectivePhase(state,
			BOT_COOP_OBJECTIVE_RETRY,
			player_entity,
			player_area,
			TRAVEL_ELEVATOR);
		/*
		 * A mover can become unavailable after the route was selected. Clear
		 * both the avoid entry and the retained reach so the next frame can
		 * select the elevator approach again instead of settling on the lower
		 * floor with a stale lastreachnum.
		 */
		BotResetAvoidReachHandle(state->move_handle);
		BotResetMoveStateHandle(state->move_handle);
		state->coop_elevator_wait_started = 0.0f;
		state->coop_elevator_wait_area = 0;
		state->coop_elevator_travel_started = 0.0f;
		state->coop_elevator_travel_area = 0;
		if (LibVarGetValue("coopbot_log") >= 2.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_elevator_retry client=%d player=%d failure=%d blocked=%d timeout=%d",
				state->client_number, player_entity, result.failure,
				result.blocked, elevator_travel_timeout_hit);
		}
	}
	else if (result.failure)
	{
			BotInterface_CoopSetObjectivePhase(state,
			BOT_COOP_OBJECTIVE_RETRY,
			player_entity,
			player_area,
			logged_traveltype);
		/*
		 * A failed regroup route must not leave the elevator reach in the
		 * one-entry avoid slot forever. The next frame is allowed to select
		 * the player route again, which is essential when the platform was
		 * temporarily unavailable or the bot entered the lower area late.
		 */
		BotResetAvoidReachHandle(state->move_handle);
	}
	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}
	BotInterface_ApplyMoveResult(&result, input);
	input->actionflags = 0;
	input->thinktime = thinktime;
	status = EA_SubmitInput(state->client_number, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}

	state->last_move_result = result;
	state->has_move_result = true;
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_regroup client=%d player=%d distance=%.1f hard_leash=%.1f "
			"vertical=%.1f bot_area=%d current_area=%d goal_area=%d "
			"failure=%d traveltype=%d elevator=%d "
			"mover_type=%d flags=0x%x blocked=%d blockentity=%d",
			state->client_number, player_entity, distance, hard_leash,
			vertical_separation, bot_area, current_player_area, player_area,
			result.failure, logged_traveltype, elevator_result,
			result.type, result.flags,
			result.blocked, result.blockentity);
		if (elevator_route_valid)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_elevator_route client=%d player=%d reach=%d "
				"source_area=%d destination_area=%d",
				state->client_number, player_entity, elevator_reach,
				elevator_source_area, elevator_destination_area);
		}
		if (elevator_result &&
			(result.flags & MOVERESULT_WAITING) != 0)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_elevator_wait client=%d player=%d distance=%.1f",
				state->client_number, player_entity, distance);
		}
		if (elevator_result &&
			(result.failure || result.blocked))
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_elevator_failed client=%d player=%d failure=%d blocked=%d blockentity=%d",
				state->client_number, player_entity, result.failure,
				result.blocked, result.blockentity);
		}
	}
	if ((result.failure || result.blocked) &&
		LibVarGetValue("coopbot_log") >= 1.0f)
	{
		/*
		 * Keep a failed regroup distinct from an ordinary idle/follow
		 * frame. Scenario 11 consumes this event when a platform is
		 * absent, busy, or the selected route cannot be completed.
		 */
		BotLib_LogWriteTimeStamped(
			"coopbot_path_failure phase=regroup client=%d player=%d "
			"failure=%d blocked=%d elevator=%d traveltype=%d "
			"goal_area=%d blockentity=%d",
			state->client_number, player_entity, result.failure,
			result.blocked, elevator_result, logged_traveltype,
			player_area, result.blockentity);
	}
	return true;
}

/*
=============
BotAI_ApplyCoopControlWait

Completes the first OpenPath HTN with an explicit WaitForPlayer step.  The
wait is only armed by a control entity that actually blocked the bot's route;
it is never a map-wide search or an unconditional pause after every door.
=============
*/
bool BotAI_ApplyCoopControlWait(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t direction;
	float distance;
	float wait_distance;
	float wait_timeout;
	int player_entity;
	int status;

	if (state == NULL || input == NULL ||
		state->coop_control_phase != BOT_COOP_CONTROL_WAIT_PLAYER)
	{
		return false;
	}
	if (!BotAI_CoopObjectiveControlEnabled())
	{
		state->coop_control_phase = BOT_COOP_CONTROL_NONE;
		state->coop_control_entity = 0;
		state->coop_control_goal_area = 0;
		state->coop_control_route_confirmed = false;
		return false;
	}
	wait_timeout = LibVarGetValue("coopbot_objective_wait_timeout");
	if (wait_timeout <= 0.0f)
	{
		wait_timeout = 30.0f;
	}
	if (AAS_Time() - state->coop_control_started >= wait_timeout)
	{
		BotAI_CoopSetControlObjectivePhase(state,
			BOT_COOP_CONTROL_RETRY,
			state->coop_control_entity,
			state->coop_control_goal_area);
		state->activation_goal_time = 0.0f;
		BotResetAvoidReachHandle(state->move_handle);
		if (LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_objective client=%d objective=OPEN_PATH "
				"phase=RETRY reason=wait_timeout timeout=%.1f "
				"route_confirmed=%d",
				state->client_number, wait_timeout,
				state->coop_control_route_confirmed ? 1 : 0);
		}
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		/* A missing player is not permission to continue through the
		 * objective.  Keep the last attack state but cancel movement. */
		status = EA_GetInput(state->client_number, thinktime, input);
		if (status != BLERR_NOERROR)
		{
			return false;
		}
		VectorClear(input->dir);
		input->speed = 0.0f;
		input->thinktime = thinktime;
		return EA_SubmitInput(state->client_number, input) == BLERR_NOERROR;
	}
	int bot_area = 0;
	int player_area = 0;
	if (!state->coop_control_route_confirmed &&
		BotInterface_CoopProbeControlRoute(state,
			player_origin,
			&bot_area,
			&player_area))
	{
		state->coop_control_route_confirmed = true;
		if (LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_objective client=%d objective=OPEN_PATH "
				"phase=ROUTE_CONFIRMED control_entity=%d goal_area=%d "
				"bot_area=%d player_area=%d source=aas_route_probe",
				state->client_number,
				state->coop_control_entity,
				state->coop_control_goal_area,
				bot_area,
				player_area);
		}
	}
	VectorSubtract(player_origin, state->last_client_update.origin,
		direction);
	distance = sqrtf(DotProduct(direction, direction));
	wait_distance = LibVarGetValue("coopbot_objective_wait_distance");
	if (wait_distance <= 0.0f)
	{
		wait_distance = 256.0f;
	}
	if (distance <= wait_distance)
	{
		BotAI_CoopSetControlObjectivePhase(state,
			BOT_COOP_CONTROL_COMPLETE,
			state->coop_control_entity,
			state->coop_control_goal_area);
		if (LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_objective client=%d objective=OPEN_PATH "
				"phase=PLAYER_REACQUIRED control_entity=%d goal_area=%d "
				"distance=%.1f route_confirmed=%d",
				state->client_number,
				state->coop_control_entity,
				state->coop_control_goal_area,
				distance,
				state->coop_control_route_confirmed ? 1 : 0);
		}
		/* Stop the overlay chain for this frame.  The next frame exposes
		 * RETURN_TO_PATH instead of immediately consuming COMPLETE here. */
		return true;
	}

	/* Preserve an attack command while removing only the committed movement. */
	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}
	VectorClear(input->dir);
	input->speed = 0.0f;
	input->thinktime = thinktime;
	return EA_SubmitInput(state->client_number, input) == BLERR_NOERROR;
}

/*
=============
BotAI_ApplyCoopControlReturnPath

Make the final OpenPath HTN step explicit. COMPLETE is the successful
WaitForPlayer result; RETURN_TO_PATH releases the stale control target on the
following frame so ordinary navigation can resume without keeping the button
or trigger objective armed.
=============
*/
bool BotAI_ApplyCoopControlReturnPath(bot_client_state_t *state)
{
	if (state == NULL || !BotAI_CoopObjectiveControlEnabled())
	{
		return false;
	}
	if (state->coop_control_phase == BOT_COOP_CONTROL_COMPLETE)
	{
		BotAI_CoopSetControlObjectivePhase(state,
			BOT_COOP_CONTROL_RETURN_PATH,
			state->coop_control_entity,
			state->coop_control_goal_area);
		return false;
	}
	if (state->coop_control_phase == BOT_COOP_CONTROL_RETURN_PATH)
	{
		BotAI_CoopSetControlObjectivePhase(state,
			BOT_COOP_CONTROL_NONE,
			0,
			0);
	}
	return false;
}

/*
=============
BotAI_ApplyCoopChangelevelGate

Do not let a companion touch a trigger whose target chain changes maps while
the human is still behind.  The trigger volume is resolved from the cached
BSP entity graph, so this remains a map-driven safety rule rather than a
hard-coded list of campaign maps.
=============
*/
bool BotAI_ApplyCoopChangelevelGate(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t direction;
	float distance;
	float wait_distance;
	int player_entity;
	int model_number = 0;
	int status;
	const aas_bspentity_t *trigger;

	if (state == NULL || input == NULL ||
		!BotAI_CoopObjectiveControlEnabled())
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		state->coop_changelevel_gate_active = false;
		state->coop_changelevel_gate_model = 0;
		return false;
	}
	trigger = BotInterface_MapFindChangelevelTrigger(
		BotInterface_CoopMapEntities(),
		state->last_client_update.origin, &model_number);
	if (trigger == NULL)
	{
		state->coop_changelevel_gate_active = false;
		state->coop_changelevel_gate_model = 0;
		return false;
	}

	VectorSubtract(player_origin, state->last_client_update.origin,
		direction);
	distance = sqrtf(DotProduct(direction, direction));
	wait_distance = LibVarGetValue("coopbot_objective_wait_distance");
	if (wait_distance <= 0.0f)
	{
		wait_distance = 256.0f;
	}
	if (distance <= wait_distance)
	{
		if (state->coop_changelevel_gate_active &&
			LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_changelevel_gate client=%d model=%d "
				"phase=RELEASE reason=player_near distance=%.1f",
				state->client_number, model_number, distance);
		}
		state->coop_changelevel_gate_active = false;
		state->coop_changelevel_gate_model = 0;
		return false;
	}

	if (!state->coop_changelevel_gate_active ||
		state->coop_changelevel_gate_model != model_number)
	{
		state->coop_changelevel_gate_active = true;
		state->coop_changelevel_gate_model = model_number;
		if (LibVarGetValue("coopbot_log") >= 1.0f)
		{
			const char *map = AAS_ValueForBSPEpairKey(trigger, "target");
			BotLib_LogWriteTimeStamped(
				"coopbot_changelevel_gate client=%d model=%d "
				"phase=WAIT_FOR_PLAYER player=%d distance=%.1f target=\"%s\"",
				state->client_number, model_number, player_entity, distance,
				map != NULL ? map : "");
		}
	}

	/* Preserve attack/view state but cancel the movement that would touch the
	 * transition trigger.  The next frame rechecks the player's distance. */
	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}
	VectorClear(input->dir);
	input->speed = 0.0f;
	input->thinktime = thinktime;
	return EA_SubmitInput(state->client_number, input) == BLERR_NOERROR;
}

/*
=============
BotAI_ApplyCoopAreaAdvanceGate

Do not let a companion silently turn a local path choice into solo
exploration.  Crossing into another AAS area is released only by a confident
ADVANCE/EXPLORE intent while the player is still close enough to follow.  A
hard leash/elevator regroup runs first and therefore retains priority for
vertical separation.
=============
*/
bool BotAI_ApplyCoopAreaAdvanceGate(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t direction;
	float distance;
	float advance_radius;
	int player_entity;
	int player_area;
	int bot_area;
	bool allowed;
	int status;

	if (state == NULL || input == NULL ||
		!BotAI_CoopObjectiveControlEnabled() ||
		!state->coop_bot_area_valid)
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		return false;
	}
	player_area = AAS_PointAreaNum(player_origin);
	bot_area = state->coop_current_area;
	if (player_area <= 0 || bot_area <= 0 || player_area == bot_area ||
		state->combat.current_enemy > aasworld.maxClients)
	{
		if (state->coop_area_gate_active)
		{
			state->coop_area_gate_active = false;
			if (LibVarGetValue("coopbot_log") >= 1.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_area_gate client=%d phase=RELEASE bot_area=%d "
					"player_area=%d reason=area_or_combat",
					state->client_number, bot_area, player_area);
			}
		}
		return false;
	}

	VectorSubtract(player_origin, state->last_client_update.origin,
		direction);
	distance = sqrtf(DotProduct(direction, direction));
	advance_radius = LibVarGetValue("coopbot_advance_radius");
	if (advance_radius <= 0.0f)
	{
		advance_radius = 256.0f;
	}
	allowed = distance <= advance_radius &&
		BotAI_CoopIntentAllowsForwardProgress(state);
	if (allowed)
	{
		if (state->coop_area_gate_active)
		{
			state->coop_area_gate_active = false;
			if (LibVarGetValue("coopbot_log") >= 1.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_area_gate client=%d phase=RELEASE bot_area=%d "
					"player_area=%d distance=%.1f intent=%s",
					state->client_number, bot_area, player_area, distance,
					BotAI_CoopPlayerIntentName(state->coop_player_intent));
			}
		}
		return false;
	}

	if (!state->coop_area_gate_active && LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_area_gate client=%d phase=WAIT_FOR_PLAYER bot_area=%d "
			"player_area=%d state=%s distance=%.1f intent=%s",
			state->client_number, bot_area, player_area,
			BotInterface_CoopAreaStateName(state->coop_area_state), distance,
			BotAI_CoopPlayerIntentName(state->coop_player_intent));
	}
	state->coop_area_gate_active = true;
	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}
	VectorClear(input->dir);
	input->speed = 0.0f;
	input->thinktime = thinktime;
	return EA_SubmitInput(state->client_number, input) == BLERR_NOERROR;
}

/*
=============
BotAI_ApplyCoopDirectionalMove

Submit one short, collision-checked lateral move for a coop overlay. The
reverse direction is tried when the first side is blocked, keeping avoidance
from turning a narrow corridor into a permanent stop.
=============
*/
bool BotAI_ApplyCoopDirectionalMove(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const vec3_t direction,
	bool preserve_actions,
	bool signal_intent)
{
	vec3_t move_direction;
	vec3_t signal_viewangles;
	vec3_t candidate;
	vec3_t mins = {-16.0f, -16.0f, -24.0f};
	vec3_t maxs = {16.0f, 16.0f, 32.0f};
	bsp_trace_t trace;
	float length;
	bool moved;
	int status;

	if (state == NULL || input == NULL || direction == NULL)
	{
		return false;
	}

	VectorCopy(direction, move_direction);
	move_direction[2] = 0.0f;
	length = sqrtf(DotProduct(move_direction, move_direction));
	if (length <= 0.1f)
	{
		return false;
	}
	VectorScale(move_direction, 1.0f / length, move_direction);
	VectorMA(state->last_client_update.origin,
		48.0f,
		move_direction,
		candidate);
	trace = AAS_Trace(state->last_client_update.origin,
		mins,
		maxs,
		candidate,
		state->entity_number,
		MASK_SOLID);
	if (trace.startsolid || trace.fraction < 0.75f)
	{
		VectorScale(move_direction, -1.0f, move_direction);
		VectorMA(state->last_client_update.origin,
			48.0f,
			move_direction,
			candidate);
		trace = AAS_Trace(state->last_client_update.origin,
			mins,
			maxs,
			candidate,
			state->entity_number,
			MASK_SOLID);
		if (trace.startsolid || trace.fraction < 0.75f)
		{
			return false;
		}
	}

	moved = BotMoveInDirectionHandle(state->move_handle,
		move_direction,
		300.0f,
		MOVE_WALK) != 0;
	if (!moved)
	{
		VectorScale(move_direction, -1.0f, move_direction);
		moved = BotMoveInDirectionHandle(state->move_handle,
			move_direction,
			300.0f,
			MOVE_WALK) != 0;
	}
	if (!moved)
	{
		return false;
	}

	EA_Move(state->client_number, move_direction, 300.0f);
	if (signal_intent &&
		LibVarGetValue("coopbot_intent_signal") != 0.0f)
	{
		Vector2Angles(move_direction, signal_viewangles);
		signal_viewangles[ROLL] *= 0.5f;
		EA_View(state->client_number, signal_viewangles);
	}
	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}
	if (!preserve_actions)
	{
		input->actionflags = 0;
	}
	input->thinktime = thinktime;
	return EA_SubmitInput(state->client_number, input) == BLERR_NOERROR;
}

/*
=============
BotAI_ApplyCoopCoverMove

Submit one strict cover step. Unlike the generic avoidance move, this helper
does not reverse direction when the requested side is blocked: reversing a
cover step could move the companion toward the enemy and make the danger
overlay counterproductive.
=============
*/
bool BotAI_ApplyCoopCoverMove(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const vec3_t direction,
	bool preserve_actions)
{
	vec3_t move_direction;
	vec3_t candidate;
	vec3_t mins = {-16.0f, -16.0f, -24.0f};
	vec3_t maxs = {16.0f, 16.0f, 32.0f};
	bsp_trace_t trace;
	float length;
	int status;

	if (state == NULL || input == NULL || direction == NULL)
	{
		return false;
	}

	VectorCopy(direction, move_direction);
	move_direction[2] = 0.0f;
	length = sqrtf(DotProduct(move_direction, move_direction));
	if (length <= 0.1f)
	{
		return false;
	}
	VectorScale(move_direction, 1.0f / length, move_direction);
	VectorMA(state->last_client_update.origin,
		96.0f,
		move_direction,
		candidate);
	trace = AAS_Trace(state->last_client_update.origin,
		mins,
		maxs,
		candidate,
		state->entity_number,
		MASK_SOLID);
	if (trace.startsolid || trace.fraction < 0.9f)
	{
		return false;
	}
	if (AAS_PointAreaNum(candidate) <= 0)
	{
		return false;
	}
	if (!BotMoveInDirectionHandle(state->move_handle,
		move_direction,
		300.0f,
		MOVE_WALK))
	{
		return false;
	}

	EA_Move(state->client_number, move_direction, 300.0f);
	status = EA_GetInput(state->client_number, thinktime, input);
	if (status != BLERR_NOERROR)
	{
		return false;
	}
	if (!preserve_actions)
	{
		input->actionflags = 0;
	}
	input->thinktime = thinktime;
	return EA_SubmitInput(state->client_number, input) == BLERR_NOERROR;
}

/*
=============
BotAI_ApplyCoopRolePositioning

Give the role model a concrete, bounded movement effect.  COVER and SUPPORT
move the companion off the player's direct line to the current enemy when the
bot is occupying the short player-to-enemy segment.  This is intentionally a
single lateral step with a cooldown: the role is allowed to create space, but
it must not fight the retail pathfinder or orbit the player indefinitely.
=============
*/
bool BotAI_ApplyCoopRolePositioning(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	aas_entityinfo_t player_info;
	aas_entityinfo_t enemy_info;
	vec3_t line;
	vec3_t bot_offset;
	vec3_t closest;
	vec3_t lateral;
	vec3_t direction;
	vec3_t right;
	float line_length;
	float projection;
	float lateral_distance;
	float radius;
	float interval;
	float lateral_dot;
	bool committed;
	int player_entity;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_roles") == 0.0f ||
		(state->coop_role != BOT_COOP_ROLE_COVER &&
			state->coop_role != BOT_COOP_ROLE_SUPPORT) ||
		state->combat.current_enemy <= aasworld.maxClients ||
		AAS_Time() < state->coop_role_next_position_time)
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntityInfo(state, &player_info);
	if (player_entity < 0)
	{
		return false;
	}
	memset(&enemy_info, 0, sizeof(enemy_info));
	AAS_EntityInfo(state->combat.current_enemy, &enemy_info);
	if (!enemy_info.valid || BotAI_EntityIsDead(&enemy_info))
	{
		return false;
	}

	VectorSubtract(enemy_info.origin, player_info.origin, line);
	line[2] = 0.0f;
	line_length = sqrtf(DotProduct(line, line));
	if (line_length <= 1.0f)
	{
		return false;
	}

	VectorSubtract(state->last_client_update.origin,
		player_info.origin,
		bot_offset);
	bot_offset[2] = 0.0f;
	projection = DotProduct(bot_offset, line) /
		(line_length * line_length);
	if (projection <= 0.05f || projection >= 0.95f)
	{
		return false;
	}

	VectorMA(player_info.origin, projection, line, closest);
	VectorSubtract(state->last_client_update.origin, closest, lateral);
	lateral[2] = 0.0f;
	lateral_distance = sqrtf(DotProduct(lateral, lateral));
	radius = LibVarGetValue("coopbot_role_position_radius");
	if (radius <= 0.0f)
	{
		radius = 64.0f;
	}
	if (lateral_distance >= radius)
	{
		return false;
	}

	committed = state->coop_action_valid &&
		state->coop_action == BOT_COOP_ACTION_POSITION &&
		AAS_Time() < state->coop_action_until;
	if (committed)
	{
		VectorCopy(state->coop_action_direction, direction);
	}
	else
	{
		right[0] = -line[1] / line_length;
		right[1] = line[0] / line_length;
		right[2] = 0.0f;
		lateral_dot = DotProduct(lateral, right);
		if (lateral_distance <= 1.0f || lateral_dot >= 0.0f)
		{
			VectorCopy(right, direction);
		}
		else
		{
			VectorScale(right, -1.0f, direction);
		}
		BotAI_CoopStartAction(state, BOT_COOP_ACTION_POSITION, direction);
	}

	if (!BotAI_ApplyCoopDirectionalMove(state,
		thinktime,
		input,
		direction,
		true,
		true))
	{
		BotAI_CoopClearActionCommitment(state);
		return false;
	}

	interval = LibVarGetValue("coopbot_role_position_interval");
	if (interval <= 0.0f)
	{
		interval = 0.5f;
	}
	state->coop_role_next_position_time = AAS_Time() + interval;
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_role_position client=%d player=%d enemy=%d role=%s "
			"lateral=%.1f projection=%.2f",
			state->client_number,
			player_entity,
			state->combat.current_enemy,
			BotAI_CoopRoleName(state->coop_role),
			lateral_distance,
			projection);
		BotLib_LogWriteTimeStamped(
			"coopbot_decision client=%d decision=ACTION_SELECT action=POSITION "
			"target=%d role=%s player_intent=%s confidence=%.2f reason=role_position",
			state->client_number,
			state->combat.current_enemy,
			BotAI_CoopRoleName(state->coop_role),
			BotAI_CoopPlayerIntentName(state->coop_player_intent),
			state->coop_player_intent_confidence);
	}
	return true;
}

/*
=============
BotAI_ApplyCoopRescuePositioning

When the game has supplied a confirmed critical-health signal for the human,
close the gap before the companion spends initiative on ordinary exploration.
The bot's own danger overlay remains a veto, and hard regroup/elevator travel
still runs first.
=============
*/
bool BotAI_ApplyCoopRescuePositioning(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	aas_entityinfo_t player_info;
	vec3_t direction;
	float distance;
	float radius;
	float interval;
	int player_entity;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_rescue") == 0.0f ||
		state->coop_role != BOT_COOP_ROLE_RESCUER ||
		AAS_Time() < state->coop_role_next_position_time ||
		BotAI_CoopDangerRequiresRetreat(state))
	{
		return false;
	}

	memset(&player_info, 0, sizeof(player_info));
	player_entity = BotAI_CoopPlayerEntityInfo(state, &player_info);
	if (player_entity < 0)
	{
		return false;
	}

	VectorSubtract(player_info.origin,
		state->last_client_update.origin,
		direction);
	direction[2] = 0.0f;
	distance = sqrtf(DotProduct(direction, direction));
	radius = LibVarGetValue("coopbot_rescue_radius");
	if (radius <= 0.0f)
	{
		radius = 160.0f;
	}
	if (distance <= 64.0f || distance > radius)
	{
		return false;
	}

	if (!BotAI_ApplyCoopDirectionalMove(state,
		thinktime,
		input,
		direction,
		true,
		true))
	{
		return false;
	}

	interval = LibVarGetValue("coopbot_role_position_interval");
	if (interval <= 0.0f)
	{
		interval = 0.5f;
	}
	state->coop_role_next_position_time = AAS_Time() + interval;
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_rescue_position client=%d player=%d distance=%.1f "
			"health=%d/%d",
			state->client_number,
			player_entity,
			distance,
			state->coop_player_health,
			state->coop_player_max_health);
		BotLib_LogWriteTimeStamped(
			"coopbot_decision client=%d decision=ACTION_SELECT action=RESCUE "
			"target=%d role=%s player_intent=%s confidence=%.2f reason=player_critical",
			state->client_number,
			player_entity,
			BotAI_CoopRoleName(state->coop_role),
			BotAI_CoopPlayerIntentName(state->coop_player_intent),
			state->coop_player_intent_confidence);
	}
	return true;
}

/*
=============
BotAI_ApplyCoopBasicCover

Adds the P1 survival fallback: if the companion is in danger and has a clear
shot on the current enemy, prefer a short reachable step that puts solid
geometry between them. This is deliberately local and deterministic; the
full cover-position utility belongs to the later cooperative-role stages.
=============
*/
bool BotAI_ApplyCoopBasicCover(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t enemy_delta;
	vec3_t away;
	vec3_t left;
	vec3_t right;
	vec3_t bot_eye;
	vec3_t candidate_eye;
	vec3_t candidates[3];
	aas_entityinfo_t enemy_info;
	float horizontal_length;
	int candidate_index;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_basic_cover") == 0.0f ||
		state->combat.current_enemy <= aasworld.maxClients ||
		!BotAI_CoopDangerRequiresRetreat(state) ||
		!aasworld.initialized || !aasworld.loaded ||
		aasworld.entities == NULL)
	{
		return false;
	}

	memset(&enemy_info, 0, sizeof(enemy_info));
	AAS_EntityInfo(state->combat.current_enemy, &enemy_info);
	if (!enemy_info.valid || BotAI_EntityIsDead(&enemy_info))
	{
		return false;
	}

	BotInterface_ClientEyePosition(state, bot_eye);
	if (!BotInterface_HasLineOfSight(bot_eye,
		enemy_info.origin,
		state->entity_number,
		state->combat.current_enemy))
	{
		return false;
	}

	VectorSubtract(state->last_client_update.origin,
		enemy_info.origin,
		enemy_delta);
	enemy_delta[2] = 0.0f;
	horizontal_length = sqrtf(DotProduct(enemy_delta, enemy_delta));
	if (horizontal_length <= 1.0f)
	{
		return false;
	}
	VectorScale(enemy_delta, 1.0f / horizontal_length, away);
	left[0] = -away[1];
	left[1] = away[0];
	left[2] = 0.0f;
	VectorScale(left, -1.0f, right);
	VectorCopy(away, candidates[0]);
	VectorCopy(left, candidates[1]);
	VectorCopy(right, candidates[2]);

	for (candidate_index = 0; candidate_index < 3; ++candidate_index)
	{
		vec3_t candidate_origin;

		VectorMA(state->last_client_update.origin,
			96.0f,
			candidates[candidate_index],
			candidate_origin);
		VectorCopy(candidate_origin, candidate_eye);
		candidate_eye[2] += state->last_client_update.viewoffset[2];
		if (BotInterface_HasLineOfSight(candidate_eye,
			enemy_info.origin,
			state->entity_number,
			state->combat.current_enemy))
		{
			continue;
		}
		if (!BotAI_ApplyCoopCoverMove(state,
			thinktime,
			input,
			candidates[candidate_index],
			true))
		{
			continue;
		}

		if (LibVarGetValue("coopbot_log") >= 2.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_cover client=%d enemy=%d danger=%.2f direction=%d",
				state->client_number,
				state->combat.current_enemy,
				BotAI_CoopDangerScore(state),
				candidate_index);
		}
		return true;
	}

	return false;
}

/*
=============
BotAI_ApplyCoopFirelineAvoidance

Moves the bot sideways when its current enemy is visible in the player's
forward line and the bot occupies the short player-to-enemy segment. This is
an opt-in coop overlay; it does not replace combat or hard regroup movement.
=============
*/
bool BotAI_ApplyCoopFirelineAvoidance(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t enemy_direction;
	vec3_t bot_offset;
	vec3_t closest;
	vec3_t lateral;
	vec3_t right;
	aas_entityinfo_t player_info;
	aas_entityinfo_t enemy_info;
	float enemy_length;
	float bot_length;
	float projection;
	float lateral_distance;
	float radius;
	float yaw;
	int player_entity;
	int side;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_fireline_avoid") == 0.0f ||
		state->combat.current_enemy <= aasworld.maxClients)
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		return false;
	}
	memset(&player_info, 0, sizeof(player_info));
	memset(&enemy_info, 0, sizeof(enemy_info));
	AAS_EntityInfo(player_entity, &player_info);
	AAS_EntityInfo(state->combat.current_enemy, &enemy_info);
	if (!player_info.valid || !enemy_info.valid)
	{
		return false;
	}

	VectorSubtract(enemy_info.origin, player_origin, enemy_direction);
	enemy_direction[2] = 0.0f;
	enemy_length = sqrtf(DotProduct(enemy_direction, enemy_direction));
	if (enemy_length <= 1.0f)
	{
		return false;
	}
	VectorScale(enemy_direction, 1.0f / enemy_length, enemy_direction);

	VectorSubtract(state->last_client_update.origin,
		player_origin,
		bot_offset);
	bot_offset[2] = 0.0f;
	bot_length = sqrtf(DotProduct(bot_offset, bot_offset));
	if (bot_length <= 1.0f)
	{
		return false;
	}
	projection = DotProduct(bot_offset, enemy_direction) / enemy_length;
	radius = LibVarGetValue("coopbot_fireline_radius");
	if (radius <= 0.0f)
	{
		radius = 128.0f;
	}
	if (projection <= 0.05f || projection >= 0.95f ||
		bot_length > radius)
	{
		return false;
	}

	VectorMA(player_origin, projection * enemy_length, enemy_direction, closest);
	VectorSubtract(state->last_client_update.origin, closest, lateral);
	lateral[2] = 0.0f;
	lateral_distance = sqrtf(DotProduct(lateral, lateral));
	if (lateral_distance >= 48.0f)
	{
		return false;
	}

	if (!AAS_EntityVisible(player_entity,
		player_origin,
		player_info.angles,
		70.0f,
		state->combat.current_enemy))
	{
		return false;
	}

	yaw = player_info.angles[YAW] * ((float)M_PI / 180.0f);
	right[0] = -sinf(yaw);
	right[1] = cosf(yaw);
	right[2] = 0.0f;
	if (lateral_distance > 1.0f)
	{
		side = DotProduct(lateral, right) >= 0.0f ? 1 : -1;
		VectorScale(right, (float)side, lateral);
	}
	else
	{
		VectorCopy(right, lateral);
	}

	if (!BotAI_ApplyCoopDirectionalMove(state,
		thinktime,
		input,
		lateral,
		true,
		false))
	{
		return false;
	}
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_fireline_avoid client=%d player=%d enemy=%d distance=%.1f "
			"lateral=%.1f",
			state->client_number,
			player_entity,
			state->combat.current_enemy,
			bot_length,
			lateral_distance);
	}
	return true;
}

/*
=============
BotAI_ApplyCoopDoorwayAvoidance

Detects the narrow doorway case from live AAS areas and the player's recent
movement vector. When the player is crossing into the bot's area and the bot
occupies the short forward corridor, the bot gives way laterally.
=============
*/
bool BotAI_ApplyCoopDoorwayAvoidance(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t player_delta;
	vec3_t movement;
	vec3_t relative;
	vec3_t lateral;
	vec3_t right;
	aas_entityinfo_t player_info;
	float movement_length;
	float forward_distance;
	float lateral_distance;
	float radius;
	float yaw;
	int player_entity;
	int player_area;
	int bot_area;
	int side;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_doorway_avoid") == 0.0f ||
		state->combat.current_enemy > 0)
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		return false;
	}
	memset(&player_info, 0, sizeof(player_info));
	AAS_EntityInfo(player_entity, &player_info);
	if (!player_info.valid)
	{
		return false;
	}

	player_area = AAS_PointAreaNum(player_origin);
	bot_area = AAS_PointAreaNum(state->last_client_update.origin);
	if (player_area <= 0 || bot_area <= 0 || player_area == bot_area)
	{
		return false;
	}

	VectorSubtract(player_origin, player_info.lastvisorigin, player_delta);
	player_delta[2] = 0.0f;
	movement_length = sqrtf(DotProduct(player_delta, player_delta));
	if (movement_length <= 2.0f)
	{
		return false;
	}
	VectorScale(player_delta, 1.0f / movement_length, movement);

	VectorSubtract(state->last_client_update.origin,
		player_origin,
		relative);
	relative[2] = 0.0f;
	forward_distance = DotProduct(relative, movement);
	lateral[0] = relative[0] - movement[0] * forward_distance;
	lateral[1] = relative[1] - movement[1] * forward_distance;
	lateral[2] = 0.0f;
	lateral_distance = sqrtf(DotProduct(lateral, lateral));
	radius = LibVarGetValue("coopbot_doorway_radius");
	if (radius <= 0.0f)
	{
		radius = 160.0f;
	}
	if (forward_distance < 24.0f || forward_distance > radius ||
		lateral_distance >= 64.0f)
	{
		return false;
	}

	yaw = player_info.angles[YAW] * ((float)M_PI / 180.0f);
	right[0] = -sinf(yaw);
	right[1] = cosf(yaw);
	right[2] = 0.0f;
	if (lateral_distance > 1.0f)
	{
		side = DotProduct(lateral, right) >= 0.0f ? 1 : -1;
		VectorScale(right, (float)side, lateral);
	}
	else
	{
		VectorCopy(right, lateral);
	}

	if (!BotAI_ApplyCoopDirectionalMove(state,
		thinktime,
		input,
		lateral,
		false,
		false))
	{
		return false;
	}
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_doorway_avoid client=%d player=%d player_area=%d "
			"bot_area=%d forward=%.1f lateral=%.1f",
			state->client_number,
			player_entity,
			player_area,
			bot_area,
			forward_distance,
			lateral_distance);
	}
	return true;
}

/*
=============
BotAI_ApplyCoopPersonalSpace

Keeps an idle companion out of the player's immediate path. When the player
is looking at the bot, the lateral escape direction is preferred so the bot
does not stand in the crosshair; elsewhere it backs away from the player.
The overlay is opt-in and runs only after the hard leash, so a vertical
regroup or elevator route always has priority.
=============
*/
bool BotAI_ApplyCoopPersonalSpace(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input)
{
	vec3_t player_origin;
	vec3_t direction;
	vec3_t relative;
	vec3_t forward;
	vec3_t right;
	aas_entityinfo_t player_info;
	float radius;
	float horizontal_distance;
	float forward_dot;
	float right_dot;
	float yaw;
	int player_entity;
	int side;

	if (state == NULL || input == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_personal_space") == 0.0f ||
		state->combat.current_enemy > 0)
	{
		return false;
	}

	player_entity = BotAI_CoopPlayerEntity(state, player_origin);
	if (player_entity < 0)
	{
		return false;
	}

	memset(&player_info, 0, sizeof(player_info));
	AAS_EntityInfo(player_entity, &player_info);
	if (!player_info.valid)
	{
		return false;
	}

	VectorSubtract(state->last_client_update.origin, player_origin, relative);
	relative[2] = 0.0f;
	horizontal_distance = sqrtf(DotProduct(relative, relative));
	radius = LibVarGetValue("coopbot_personal_space_radius");
	if (radius <= 0.0f || horizontal_distance >= radius ||
		fabsf(state->last_client_update.origin[2] - player_origin[2]) >= 64.0f)
	{
		return false;
	}

	yaw = player_info.angles[YAW] * ((float)M_PI / 180.0f);
	forward[0] = cosf(yaw);
	forward[1] = sinf(yaw);
	forward[2] = 0.0f;
	right[0] = -forward[1];
	right[1] = forward[0];
	right[2] = 0.0f;
	if (horizontal_distance > 0.1f)
	{
		VectorScale(relative, 1.0f / horizontal_distance, relative);
	}
	else
	{
		VectorCopy(right, relative);
	}
	forward_dot = DotProduct(relative, forward);
	right_dot = DotProduct(relative, right);
	if (forward_dot > 0.35f)
	{
		/* Leave the player's forward cone by moving farther toward one side. */
		side = right_dot >= 0.0f ? 1 : -1;
		VectorScale(right, (float)side, direction);
	}
	else if (horizontal_distance > 0.1f)
	{
		VectorCopy(relative, direction);
	}
	else
	{
		VectorCopy(right, direction);
	}

	if (!BotAI_ApplyCoopDirectionalMove(state,
		thinktime,
		input,
		direction,
		false,
		false))
	{
		return false;
	}
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_personal_space client=%d player=%d distance=%.1f radius=%.1f "
			"forward=%.2f lateral=%.2f",
			state->client_number, player_entity, horizontal_distance, radius,
			forward_dot, right_dot);
	}
	return true;
}

/*
=============
BotAI_Think

Runs one reconstructed per-client AI frame, including retail console-message
processing before node-equivalent movement work.
=============
*/
