#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/common/l_utils.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "bot_interface_area.h"
#include "bot_interface_combat.h"
#include "bot_interface_coop_state.h"
#include "bot_state.h"

/*
=============
BotAI_CoopPlayerEntity

Finds the first active non-bot client entity. Bot state slots are created only
for bots, so the inactive slot is the game-side human companion we should
follow in coop mode.
=============
*/
int BotAI_CoopPlayerEntityInfo(const bot_client_state_t *state,
	aas_entityinfo_t *player_info)
{
	int max_clients;
	int client;

	if (state == NULL || player_info == NULL || !BotAI_CoopMode())
	{
		return -1;
	}

	max_clients = aasworld.maxClients;
	if (max_clients > MAX_CLIENTS)
	{
		max_clients = MAX_CLIENTS;
	}
	for (client = 0; client < max_clients; ++client)
	{
		int entity = client + 1;
		bot_client_state_t *candidate_state;
		aas_entityinfo_t entity_info;

		if (client == state->client_number)
		{
			continue;
		}
		candidate_state = BotState_Get(client);
		if (candidate_state != NULL && candidate_state->active)
		{
			continue;
		}
		memset(&entity_info, 0, sizeof(entity_info));
		AAS_EntityInfo(entity, &entity_info);
		if (!entity_info.valid || entity_info.number != entity)
		{
			continue;
		}

		*player_info = entity_info;
		return entity;
	}

	return -1;
}

int BotAI_CoopPlayerEntity(const bot_client_state_t *state,
	vec3_t origin)
{
	aas_entityinfo_t player_info;
	int entity;

	if (origin == NULL)
	{
		return -1;
	}
	memset(&player_info, 0, sizeof(player_info));
	entity = BotAI_CoopPlayerEntityInfo(state, &player_info);
	if (entity >= 0)
	{
		VectorCopy(player_info.origin, origin);
	}
	return entity;
}

/*
=============
BotAI_ResetCoopPlayerState

Drop all map-local follow, regroup, rescue, role, and focus state.  A coop
player can keep the same client slot across death, so a new alive frame must
not inherit the previous spawn's AAS area or objective route.
=============
*/
void BotAI_ResetCoopPlayerState(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	state->coop_player_entity = -1;
	state->coop_player_dead = true;
	state->coop_player_area = 0;
	VectorClear(state->coop_player_origin);
	state->coop_player_goal_valid = false;
	state->coop_elevator_wait_started = 0.0f;
	state->coop_elevator_wait_area = 0;
	state->coop_elevator_travel_started = 0.0f;
	state->coop_elevator_travel_area = 0;
	state->coop_player_intent = BOT_COOP_INTENT_UNKNOWN;
	state->coop_player_intent_confidence = 0.0f;
	state->coop_player_intent_time = 0.0f;
	VectorClear(state->coop_player_last_origin);
	VectorClear(state->coop_player_last_velocity);
	state->coop_player_last_yaw = 0.0f;
	state->coop_player_last_threat_distance = 0.0f;
	state->coop_player_last_area = 0;
	state->coop_player_motion_valid = false;
	state->coop_player_threat_valid = false;
	state->coop_player_area_valid = false;
	state->coop_player_focus_entity = 0;
	state->coop_player_focus_confidence = 0.0f;
	state->coop_player_focus_time = 0.0f;
	state->coop_target_candidate_entity = 0;
	state->coop_target_candidate_time = 0.0f;
	state->coop_bot_last_area = 0;
	state->coop_bot_area_valid = false;
	state->coop_role = BOT_COOP_ROLE_FOLLOWER;
	state->coop_initiative_budget = 0.0f;
	state->coop_role_confidence = 0.0f;
	state->coop_role_time = 0.0f;
	state->coop_role_next_position_time = 0.0f;
	state->coop_action = BOT_COOP_ACTION_NONE;
	state->coop_action_started = 0.0f;
	state->coop_action_until = 0.0f;
	VectorClear(state->coop_action_direction);
	state->coop_action_valid = false;
	state->coop_objective_phase = BOT_COOP_OBJECTIVE_NONE;
	state->coop_objective_started = 0.0f;
	state->coop_objective_goal_area = 0;
	state->coop_objective_retries = 0;
	state->coop_control_phase = BOT_COOP_CONTROL_NONE;
	state->coop_control_entity = 0;
	state->coop_control_goal_area = 0;
	state->coop_control_started = 0.0f;
	state->coop_control_route_confirmed = false;
	state->coop_changelevel_gate_active = false;
	state->coop_changelevel_gate_model = 0;
	state->coop_current_area = 0;
	state->coop_area_state = BOT_COOP_AREA_UNKNOWN;
	state->coop_area_enemy_count = 0;
	state->coop_area_combat_seen = false;
	state->coop_area_gate_active = false;
	state->coop_console_status_time = 0.0f;
	state->coop_evasive_next_time = 0.0f;
	state->coop_evasive_until = 0.0f;
	state->coop_evasive_enemy = 0;
	state->coop_evasive_side = 1;
	state->coop_last_safe_area = 0;
	VectorClear(state->coop_last_safe_origin);
	state->coop_last_safe_time = 0.0f;
	state->coop_last_safe_valid = false;
	state->coop_joint_retreat_until = 0.0f;
	state->coop_joint_retreat_active = false;
}

static void BotAI_ObserveCoopPlayerLifecycle(bot_client_state_t *state,
	int player_entity)
{
	bool player_dead;
	bool player_changed;
	bool player_revived;

	if (state == NULL || player_entity <= 0)
	{
		return;
	}

	player_dead = state->coop_player_telemetry_valid &&
		state->coop_player_health <= 0;
	player_changed = state->coop_player_entity > 0 &&
		state->coop_player_entity != player_entity;
	player_revived = state->coop_player_entity == player_entity &&
		state->coop_player_dead && !player_dead;
	if (player_changed || player_revived)
	{
		BotAI_ResetCoopPlayerState(state);
		if (LibVarGetValue("coopbot_log") >= 1.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_player_rebind client=%d player=%d reason=%s",
				state->client_number, player_entity,
				player_changed ? "entity_changed" : "respawn");
		}
	}
	state->coop_player_entity = player_entity;
	state->coop_player_dead = player_dead;
}

/*
=============
BotAI_CoopPlayerIntentName

Keep the decision log readable without exposing enum values as an external
contract.
=============
*/
const char *BotAI_CoopPlayerIntentName(bot_coop_player_intent_t intent)
{
	switch (intent)
	{
		case BOT_COOP_INTENT_ADVANCE:
			return "ADVANCE";
		case BOT_COOP_INTENT_HOLD:
			return "HOLD";
		case BOT_COOP_INTENT_RETREAT:
			return "RETREAT";
		case BOT_COOP_INTENT_ENGAGE_TARGET:
			return "ENGAGE_TARGET";
		case BOT_COOP_INTENT_SEARCH:
			return "SEARCH";
		case BOT_COOP_INTENT_EXPLORE:
			return "EXPLORE";
		case BOT_COOP_INTENT_INTERACT:
			return "INTERACT";
		case BOT_COOP_INTENT_LOOT:
			return "LOOT";
		case BOT_COOP_INTENT_WAIT:
			return "WAIT";
		case BOT_COOP_INTENT_UNKNOWN:
		default:
			return "UNKNOWN";
	}
}

bool BotAI_CoopIntentIsConfident(const bot_client_state_t *state,
	bot_coop_player_intent_t intent)
{
	float threshold;

	if (state == NULL || state->coop_player_intent != intent)
	{
		return false;
	}
	threshold = LibVarGetValue("coopbot_intent_confidence");
	if (threshold <= 0.0f)
	{
		threshold = 0.60f;
	}
	return state->coop_player_intent_confidence >= threshold;
}

bool BotAI_CoopIntentAllowsForwardProgress(
	const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return false;
	}
	return (state->coop_player_intent == BOT_COOP_INTENT_ADVANCE ||
		state->coop_player_intent == BOT_COOP_INTENT_EXPLORE) &&
		state->coop_player_intent_confidence >=
			(LibVarGetValue("coopbot_intent_confidence") > 0.0f
				? LibVarGetValue("coopbot_intent_confidence")
				: 0.60f);
}

/*
=============
BotAI_UpdateCoopPlayerTelemetry

The retail BotUpdateEntity contract intentionally contains no player-state
health or inventory.  The game-side coop bridge publishes those few values
through namespaced libvars, keeping the 20-pointer Gladiator ABI unchanged.
Missing telemetry is not treated as zero health.
=============
*/
void BotAI_UpdateCoopPlayerTelemetry(bot_client_state_t *state,
	int player_entity)
{
	char variable_name[64];
	const char *value;
	char *end;
	long parsed;
	int health;
	int max_health;

	if (state == NULL || player_entity <= 0)
	{
		return;
	}

	snprintf(variable_name,
		sizeof(variable_name),
		"coopbot_player_health_%d",
		player_entity);
	value = LibVarGetString(variable_name);
	if (value == NULL || value[0] == '\0')
	{
		state->coop_player_telemetry_valid = false;
		return;
	}
	parsed = strtol(value, &end, 10);
	if (end == value)
	{
		state->coop_player_telemetry_valid = false;
		return;
	}
	health = (int)parsed;

	snprintf(variable_name,
		sizeof(variable_name),
		"coopbot_player_max_health_%d",
		player_entity);
	value = LibVarGetString(variable_name);
	if (value == NULL || value[0] == '\0')
	{
		state->coop_player_telemetry_valid = false;
		return;
	}
	parsed = strtol(value, &end, 10);
	if (end == value || parsed <= 0)
	{
		state->coop_player_telemetry_valid = false;
		return;
	}
	max_health = (int)parsed;

	snprintf(variable_name,
		sizeof(variable_name),
		"coopbot_player_armor_%d",
		player_entity);
	state->coop_player_armor = (int)LibVarGetValue(variable_name);
	snprintf(variable_name,
		sizeof(variable_name),
		"coopbot_player_damage_blood_%d",
		player_entity);
	state->coop_player_damage_blood = (int)LibVarGetValue(variable_name);
	state->coop_player_health = health;
	state->coop_player_max_health = max_health;
	state->coop_player_telemetry_valid = true;
}

bool BotAI_CoopPlayerNeedsRescue(const bot_client_state_t *state)
{
	float critical_health;
	float damage_threshold;

	if (state == NULL || !state->coop_player_telemetry_valid ||
		LibVarGetValue("coopbot_rescue") == 0.0f)
	{
		return false;
	}

	critical_health = LibVarGetValue("coopbot_player_critical_health");
	if (critical_health <= 0.0f)
	{
		critical_health = 25.0f;
	}
	damage_threshold = LibVarGetValue("coopbot_rescue_damage_threshold");
	if (damage_threshold <= 0.0f)
	{
		damage_threshold = 15.0f;
	}

	return state->coop_player_health <= critical_health ||
		(state->coop_player_health < state->coop_player_max_health * 0.5f &&
			state->coop_player_damage_blood >= damage_threshold);
}

/*
=============
BotAI_UpdateCoopPlayerIntent

Infer the human's short-horizon intent from the data the game actually sends
to botlib: entity displacement, player animation, player-facing direction,
visible monster pressure, AAS area changes, and optional coop health telemetry.
Objective interactions are still intentionally not guessed because the current
entity/player feed does not expose a reliable use target. Unknown is therefore
preferred over a fabricated intent.
=============
*/
void BotAI_UpdateCoopPlayerIntent(bot_client_state_t *state)
{
	aas_entityinfo_t player_info;
	bot_coop_player_intent_t previous_intent;
	bot_coop_player_intent_t intent;
	vec3_t velocity;
	vec3_t horizontal_velocity;
	vec3_t threat_direction;
	float speed;
	float threat_distance = 0.0f;
	float yaw_delta = 0.0f;
	float confidence;
	float focus_age;
	float focus_memory;
	float advance_speed;
	float hold_speed;
	int player_entity;
	int player_area;
	int visible_count = 0;
	int enemy_count = 0;
	int player_focus_entity = 0;
	float player_focus_distance = 0.0f;
	bool player_shooting;
	bool player_target_visible = false;
	bool area_changed = false;
	bool retreating = false;
	bool has_threat = false;

	if (state == NULL || !BotAI_CoopMode())
	{
		if (state != NULL)
		{
			BotAI_ResetCoopPlayerState(state);
			state->coop_player_intent = BOT_COOP_INTENT_UNKNOWN;
			state->coop_player_intent_confidence = 0.0f;
			state->coop_player_motion_valid = false;
			state->coop_player_threat_valid = false;
			state->coop_player_area_valid = false;
			state->coop_player_focus_entity = 0;
			state->coop_player_focus_confidence = 0.0f;
			state->coop_player_focus_time = 0.0f;
		}
		return;
	}

	memset(&player_info, 0, sizeof(player_info));
	player_entity = BotAI_CoopPlayerEntityInfo(state, &player_info);
	if (player_entity < 0 || !aasworld.initialized ||
		aasworld.entities == NULL)
	{
		BotAI_ResetCoopPlayerState(state);
		state->coop_player_intent = BOT_COOP_INTENT_UNKNOWN;
		state->coop_player_intent_confidence = 0.0f;
		state->coop_player_motion_valid = false;
		state->coop_player_threat_valid = false;
		state->coop_player_area_valid = false;
		state->coop_player_focus_entity = 0;
		state->coop_player_focus_confidence = 0.0f;
		state->coop_player_focus_time = 0.0f;
		return;
	}
	BotAI_UpdateCoopPlayerTelemetry(state, player_entity);
	BotAI_ObserveCoopPlayerLifecycle(state, player_entity);
	if (LibVarGetValue("coopbot_player_intent") == 0.0f)
	{
		state->coop_player_intent = BOT_COOP_INTENT_UNKNOWN;
		state->coop_player_intent_confidence = 0.0f;
		state->coop_player_motion_valid = false;
		state->coop_player_threat_valid = false;
		state->coop_player_area_valid = false;
		state->coop_player_focus_entity = 0;
		state->coop_player_focus_confidence = 0.0f;
		state->coop_player_focus_time = 0.0f;
		return;
	}

	VectorSubtract(player_info.origin, player_info.old_origin, velocity);
	VectorCopy(velocity, horizontal_velocity);
	horizontal_velocity[2] = 0.0f;
	speed = sqrtf(DotProduct(horizontal_velocity, horizontal_velocity));
	player_area = AAS_PointAreaNum(player_info.origin);
	if (state->coop_player_area_valid && player_area > 0 &&
		player_area != state->coop_player_last_area)
	{
		area_changed = true;
	}
	if (state->coop_player_motion_valid)
	{
		yaw_delta = fabsf(player_info.angles[YAW] -
			state->coop_player_last_yaw);
		while (yaw_delta > 180.0f)
		{
			yaw_delta -= 360.0f;
		}
		yaw_delta = fabsf(yaw_delta);
	}

	player_shooting = BotAI_EntityIsShooting(&player_info) != 0;
	if (player_entity > 0)
	{
		int visible_entities[32];
		visible_count = AAS_VisibleEntities(player_entity,
			player_info.origin,
			player_info.angles,
			360.0f,
			(int)(sizeof(visible_entities) / sizeof(visible_entities[0])),
			visible_entities);
		for (int index = 0; index < visible_count; ++index)
		{
			aas_entityinfo_t entity_info;
			vec3_t direction;
			vec3_t target_angles;
			float distance;

			memset(&entity_info, 0, sizeof(entity_info));
			AAS_EntityInfo(visible_entities[index], &entity_info);
			if (BotAI_EntityIsDead(&entity_info) ||
				entity_info.number <= aasworld.maxClients ||
				entity_info.number == state->entity_number)
			{
				continue;
			}
			VectorSubtract(entity_info.origin, player_info.origin, direction);
			distance = sqrtf(DotProduct(direction, direction));
			Vector2Angles(direction, target_angles);
			if (BotInterface_InFieldOfVision(player_info.angles,
				120.0f,
				target_angles))
			{
				player_target_visible = true;
				if (player_focus_entity == 0 ||
					distance < player_focus_distance)
				{
					player_focus_entity = entity_info.number;
					player_focus_distance = distance;
				}
			}
			if (!has_threat || distance < threat_distance)
			{
				threat_distance = distance;
				has_threat = true;
				VectorSubtract(player_info.origin,
					entity_info.origin,
					threat_direction);
			}
			enemy_count += 1;
		}
	}

	if (has_threat && state->coop_player_threat_valid &&
		threat_distance > state->coop_player_last_threat_distance + 4.0f)
	{
		retreating = true;
	}
	if (has_threat && speed >= 32.0f)
	{
		threat_direction[2] = 0.0f;
		float threat_length = sqrtf(DotProduct(threat_direction,
			threat_direction));
		if (threat_length > 1.0f)
		{
			VectorScale(threat_direction, 1.0f / threat_length,
				threat_direction);
			VectorScale(horizontal_velocity, 1.0f / speed,
				horizontal_velocity);
			if (DotProduct(horizontal_velocity, threat_direction) > 0.25f)
			{
				retreating = true;
			}
		}
	}

	advance_speed = LibVarGetValue("coopbot_intent_advance_speed");
	if (advance_speed <= 0.0f)
	{
		advance_speed = 48.0f;
	}
	hold_speed = LibVarGetValue("coopbot_intent_hold_speed");
	if (hold_speed <= 0.0f)
	{
		hold_speed = 24.0f;
	}

	intent = BOT_COOP_INTENT_UNKNOWN;
	confidence = 0.25f;
	if (retreating)
	{
		intent = BOT_COOP_INTENT_RETREAT;
		confidence = 0.90f;
	}
	else if (player_shooting && player_target_visible)
	{
		intent = BOT_COOP_INTENT_ENGAGE_TARGET;
		confidence = 0.85f;
	}
	else if (speed >= advance_speed)
	{
		intent = area_changed ? BOT_COOP_INTENT_ADVANCE :
			BOT_COOP_INTENT_EXPLORE;
		confidence = area_changed ? 0.85f : 0.65f;
	}
	else if (enemy_count > 0 && speed <= hold_speed &&
		(!state->coop_player_motion_valid || yaw_delta < 20.0f))
	{
		intent = BOT_COOP_INTENT_HOLD;
		confidence = 0.78f;
	}
	else if (enemy_count == 0 && yaw_delta >= 20.0f)
	{
		intent = BOT_COOP_INTENT_SEARCH;
		confidence = 0.62f;
	}
	else if (speed < hold_speed)
	{
		intent = BOT_COOP_INTENT_WAIT;
		confidence = 0.55f;
	}

	previous_intent = state->coop_player_intent;
	state->coop_player_intent = intent;
	state->coop_player_intent_confidence = confidence;
	state->coop_player_intent_time = AAS_Time();
	VectorCopy(player_info.origin, state->coop_player_last_origin);
	VectorCopy(velocity, state->coop_player_last_velocity);
	state->coop_player_last_yaw = player_info.angles[YAW];
	state->coop_player_motion_valid = true;
	if (player_shooting && player_target_visible)
	{
		state->coop_player_focus_entity = player_focus_entity;
		state->coop_player_focus_confidence = 0.85f;
		state->coop_player_focus_time = AAS_Time();
		if (LibVarGetValue("coopbot_log") >= 2.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_player_focus client=%d player=%d target=%d "
				"distance=%.1f confidence=%.2f",
				state->client_number,
				player_entity,
				player_focus_entity,
				player_focus_distance,
				state->coop_player_focus_confidence);
		}
	}
	else if (state->coop_player_focus_entity > 0)
	{
		focus_memory = LibVarGetValue("coopbot_focus_memory");
		if (focus_memory <= 0.0f)
		{
			focus_memory = 0.75f;
		}
		focus_age = AAS_Time() - state->coop_player_focus_time;
		if (focus_age >= 0.0f && focus_age <= focus_memory)
		{
			state->coop_player_focus_confidence = 0.85f *
				(1.0f - focus_age / focus_memory);
		}
		else
		{
			state->coop_player_focus_entity = 0;
			state->coop_player_focus_confidence = 0.0f;
			state->coop_player_focus_time = 0.0f;
		}
	}
	else
	{
		state->coop_player_focus_entity = 0;
		state->coop_player_focus_confidence = 0.0f;
		state->coop_player_focus_time = 0.0f;
	}
	if (has_threat)
	{
		state->coop_player_last_threat_distance = threat_distance;
		state->coop_player_threat_valid = true;
	}
	else
	{
		state->coop_player_last_threat_distance = 0.0f;
		state->coop_player_threat_valid = false;
	}
	if (player_area > 0)
	{
		if (area_changed)
		{
			BotInterface_LogCoopAreaTransition(state, "player",
				state->coop_player_last_area,
				player_area,
				enemy_count > 0 ? "ACTIVE_COMBAT" : "VISITED");
		}
		state->coop_player_last_area = player_area;
		state->coop_player_area_valid = true;
	}

	if (previous_intent != intent && LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_player_intent client=%d player=%d intent=%s "
			"confidence=%.2f speed=%.1f enemies=%d area=%d changed=%d",
			state->client_number,
			player_entity,
			BotAI_CoopPlayerIntentName(intent),
			confidence,
			speed,
			enemy_count,
			player_area,
			area_changed);
	}
}

static void BotAI_CoopResetPlayerStyle(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}
	state->coop_player_style_aggression = 0.5f;
	state->coop_player_style_pace = 0.5f;
	state->coop_player_style_preferred_range = 256.0f;
	state->coop_player_style_risk_tolerance = 0.5f;
	state->coop_player_style_retreat_frequency = 0.0f;
	state->coop_player_style_exploration = 0.0f;
	state->coop_player_style_confidence = 0.0f;
	state->coop_player_style_time = 0.0f;
	state->coop_player_style_observations = 0;
	state->coop_player_style_valid = false;
}

static float BotAI_CoopStyleClamp01(float value)
{
	return fminf(fmaxf(value, 0.0f), 1.0f);
}
/*
=============
BotAI_UpdateCoopPlayerStyle

Accumulate a small, inspectable style model from observations already exposed
by the coop bridge.  This deliberately does not invent accuracy or inventory
usage: those signals are not part of the current ABI.  Each sample is
smoothed with an EMA and confidence grows only with actual observations.
=============
*/
void BotAI_UpdateCoopPlayerStyle(bot_client_state_t *state)
{
	aas_entityinfo_t player_info;
	aas_entityinfo_t target_info;
	vec3_t velocity;
	vec3_t horizontal_velocity;
	vec3_t to_target;
	float speed;
	float target_distance = 0.0f;
	float health_ratio = 0.5f;
	float learning_rate;
	float interval;
	float now;
	float aggression;
	float pace;
	float risk_tolerance;
	float retreat;
	float exploration;
	bool has_target = false;
	bool player_shooting;
	bool moving_toward_target = false;
	int player_entity;
	int visible_entities[32];
	int visible_count;

	if (state == NULL)
	{
		return;
	}
	if (LibVarGetValue("coopbot_player_style") == 0.0f ||
		!BotAI_CoopMode())
	{
		BotAI_CoopResetPlayerStyle(state);
		return;
	}

	now = AAS_Time();
	interval = LibVarGetValue("coopbot_style_update_interval");
	if (interval < 0.0f)
	{
		interval = 1.0f;
	}
	if (state->coop_player_style_valid &&
		now - state->coop_player_style_time < interval)
	{
		return;
	}

	memset(&player_info, 0, sizeof(player_info));
	player_entity = BotAI_CoopPlayerEntityInfo(state, &player_info);
	if (player_entity < 0)
	{
		BotAI_CoopResetPlayerStyle(state);
		return;
	}
	BotAI_UpdateCoopPlayerTelemetry(state, player_entity);

	VectorSubtract(player_info.origin, player_info.old_origin, velocity);
	VectorCopy(velocity, horizontal_velocity);
	horizontal_velocity[2] = 0.0f;
	speed = sqrtf(DotProduct(horizontal_velocity, horizontal_velocity));
	player_shooting = BotAI_EntityIsShooting(&player_info) != 0;

	memset(&target_info, 0, sizeof(target_info));
	if (state->coop_player_focus_entity > aasworld.maxClients)
	{
		AAS_EntityInfo(state->coop_player_focus_entity, &target_info);
		if (target_info.valid && !BotAI_EntityIsDead(&target_info))
		{
			has_target = true;
		}
	}
	if (!has_target && player_entity > 0)
	{
		/* The style model remains useful when the optional intent model is off. */
		visible_count = AAS_VisibleEntities(player_entity,
			player_info.origin,
			player_info.angles,
			360.0f,
			(int)(sizeof(visible_entities) / sizeof(visible_entities[0])),
			visible_entities);
		for (int index = 0; index < visible_count; ++index)
		{
			aas_entityinfo_t candidate;
			memset(&candidate, 0, sizeof(candidate));
			AAS_EntityInfo(visible_entities[index], &candidate);
			if (candidate.number <= aasworld.maxClients ||
				candidate.number == state->entity_number ||
				!candidate.valid || BotAI_EntityIsDead(&candidate))
			{
				continue;
			}
			if (!has_target)
			{
				target_info = candidate;
				has_target = true;
			}
			else
			{
				vec3_t current_direction;
				vec3_t candidate_direction;
				VectorSubtract(target_info.origin, player_info.origin,
					current_direction);
				VectorSubtract(candidate.origin, player_info.origin,
					candidate_direction);
				if (DotProduct(candidate_direction, candidate_direction) <
					DotProduct(current_direction, current_direction))
				{
					target_info = candidate;
				}
			}
		}
	}
	if (has_target)
	{
		VectorSubtract(target_info.origin, player_info.origin, to_target);
		target_distance = sqrtf(DotProduct(to_target, to_target));
		to_target[2] = 0.0f;
		float target_length = sqrtf(DotProduct(to_target, to_target));
		if (speed >= 32.0f && target_length > 1.0f)
		{
			VectorScale(to_target, 1.0f / target_length, to_target);
			VectorScale(horizontal_velocity, 1.0f / speed,
				horizontal_velocity);
			moving_toward_target =
				DotProduct(horizontal_velocity, to_target) > 0.25f;
		}
	}

	if (state->coop_player_telemetry_valid &&
		state->coop_player_max_health > 0)
	{
		health_ratio = BotAI_CoopStyleClamp01(
			(float)state->coop_player_health /
			(float)state->coop_player_max_health);
	}

	/* Aggression is inferred from target engagement and forward pressure. */
	aggression = has_target ? 0.65f : 0.30f;
	if (player_shooting && has_target)
	{
		aggression += 0.25f;
	}
	if (moving_toward_target)
	{
		aggression += 0.10f;
	}
	if (state->coop_player_intent == BOT_COOP_INTENT_RETREAT)
	{
		aggression -= 0.25f;
	}
	aggression = BotAI_CoopStyleClamp01(aggression);

	pace = BotAI_CoopStyleClamp01(speed / 160.0f);
	risk_tolerance = 0.50f;
	if (moving_toward_target)
	{
		risk_tolerance += 0.25f;
	}
	if (health_ratio < 0.35f && moving_toward_target)
	{
		risk_tolerance += 0.15f;
	}
	if (state->coop_player_intent == BOT_COOP_INTENT_RETREAT)
	{
		risk_tolerance -= 0.35f;
	}
	risk_tolerance = BotAI_CoopStyleClamp01(risk_tolerance);
	retreat = state->coop_player_intent == BOT_COOP_INTENT_RETREAT
		? 1.0f : 0.0f;
	exploration = (state->coop_player_intent == BOT_COOP_INTENT_ADVANCE ||
		state->coop_player_intent == BOT_COOP_INTENT_EXPLORE ||
		state->coop_player_intent == BOT_COOP_INTENT_SEARCH) ? 1.0f : 0.0f;

	learning_rate = LibVarGetValue("coopbot_style_learning_rate");
	if (learning_rate <= 0.0f)
	{
		learning_rate = 0.10f;
	}
	learning_rate = fminf(learning_rate, 0.50f);
	if (!state->coop_player_style_valid)
	{
		state->coop_player_style_aggression = aggression;
		state->coop_player_style_pace = pace;
		state->coop_player_style_risk_tolerance = risk_tolerance;
		state->coop_player_style_retreat_frequency = retreat;
		state->coop_player_style_exploration = exploration;
		if (has_target)
		{
			state->coop_player_style_preferred_range =
				fminf(fmaxf(target_distance, 64.0f), 768.0f);
		}
		state->coop_player_style_valid = true;
	}
	else
	{
		state->coop_player_style_aggression =
			state->coop_player_style_aggression * (1.0f - learning_rate) +
			aggression * learning_rate;
		state->coop_player_style_pace =
			state->coop_player_style_pace * (1.0f - learning_rate) +
			pace * learning_rate;
		state->coop_player_style_risk_tolerance =
			state->coop_player_style_risk_tolerance * (1.0f - learning_rate) +
			risk_tolerance * learning_rate;
		state->coop_player_style_retreat_frequency =
			state->coop_player_style_retreat_frequency * (1.0f - learning_rate) +
			retreat * learning_rate;
		state->coop_player_style_exploration =
			state->coop_player_style_exploration * (1.0f - learning_rate) +
			exploration * learning_rate;
		if (has_target)
		{
			state->coop_player_style_preferred_range =
				state->coop_player_style_preferred_range *
					(1.0f - learning_rate) +
				fminf(fmaxf(target_distance, 64.0f), 768.0f) *
					learning_rate;
		}
	}
	state->coop_player_style_time = now;
	state->coop_player_style_observations += 1;
	state->coop_player_style_confidence = fminf(1.0f,
		(float)state->coop_player_style_observations / 20.0f);
	if (LibVarGetValue("coopbot_log") >= 1.0f &&
		(state->coop_player_style_observations == 1 ||
			state->coop_player_style_observations % 5 == 0))
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_player_style client=%d player=%d aggression=%.2f "
			"pace=%.2f preferred_range=%.1f risk=%.2f retreat=%.2f "
			"exploration=%.2f confidence=%.2f observations=%d",
			state->client_number, player_entity,
			state->coop_player_style_aggression,
			state->coop_player_style_pace,
			state->coop_player_style_preferred_range,
			state->coop_player_style_risk_tolerance,
			state->coop_player_style_retreat_frequency,
			state->coop_player_style_exploration,
			state->coop_player_style_confidence,
			state->coop_player_style_observations);
	}
}
