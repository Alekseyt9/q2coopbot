#include <math.h>
#include <string.h>

#include "botlib/aas/aas_local.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "bot_interface_area.h"

void BotInterface_LogCoopAreaTransition(bot_client_state_t *state,
	const char *actor,
	int from_area,
	int to_area,
	const char *area_state)
{
	if (state == NULL || actor == NULL || area_state == NULL ||
		LibVarGetValue("coopbot_map_model") == 0.0f)
	{
		return;
	}
	BotLib_LogWriteTimeStamped(
		"coopbot_area_transition client=%d actor=%s from=%d to=%d state=%s",
		state->client_number,
		actor,
		from_area,
		to_area,
		area_state);
}

const char *BotInterface_CoopAreaStateName(bot_coop_area_state_t state)
{
	switch (state)
	{
		case BOT_COOP_AREA_VISITED:
			return "VISITED";
		case BOT_COOP_AREA_ACTIVE_COMBAT:
			return "ACTIVE_COMBAT";
		case BOT_COOP_AREA_PARTIALLY_CLEARED:
			return "PARTIALLY_CLEARED";
		case BOT_COOP_AREA_CLEARED:
			return "CLEARED";
		case BOT_COOP_AREA_DANGEROUS:
			return "DANGEROUS";
		case BOT_COOP_AREA_UNKNOWN:
		default:
			return "UNKNOWN";
	}
}

/*
 * Find or allocate the small episode-local record for an AAS area.  Replacing
 * the least recently observed record keeps the memory bounded on maps with
 * many areas while preserving the useful history near the bot's route.
 */
bot_coop_area_memory_t *BotInterface_FindCoopAreaMemory(
	bot_client_state_t *state,
	int area,
	bool create)
{
	bot_coop_area_memory_t *memory;
	int index;
	int slot;
	float oldest_time;

	if (state == NULL || area <= 0)
	{
		return NULL;
	}

	for (index = 0; index < state->coop_area_memory_count; index += 1)
	{
		memory = &state->coop_area_memory[index];
		if (memory->area == area)
		{
			return memory;
		}
	}
	if (!create)
	{
		return NULL;
	}

	if (state->coop_area_memory_count < BOT_COOP_AREA_MEMORY_MAX)
	{
		slot = state->coop_area_memory_count;
		state->coop_area_memory_count += 1;
	}
	else
	{
		slot = 0;
		oldest_time = state->coop_area_memory[0].last_observed;
		for (index = 1; index < BOT_COOP_AREA_MEMORY_MAX; index += 1)
		{
			if (state->coop_area_memory[index].last_observed < oldest_time)
			{
				slot = index;
				oldest_time =
					state->coop_area_memory[index].last_observed;
			}
		}
	}

	memory = &state->coop_area_memory[slot];
	memset(memory, 0, sizeof(*memory));
	memory->area = area;
	memory->state = BOT_COOP_AREA_VISITED;
	return memory;
}

/*
=============
BotInterface_CoopAreaIsDangerous

Keep the area label conservative.  Missing health telemetry is not treated as
critical, but an explicitly critical health sample or recent incoming damage
marks a live encounter as dangerous until the area is clear.
=============
*/
bool BotInterface_CoopAreaIsDangerous(const bot_client_state_t *state,
	int enemy_count)
{
	float critical_health;
	int health;

	if (state == NULL || enemy_count <= 0)
	{
		return false;
	}

	/* BotInterface_UpdateBattleInventory runs later in the frame; use the raw
	 * client stat here so a synthetic/missing inventory slot cannot look like
	 * one point of health. */
	health = state->last_client_update.stats[STAT_HEALTH];
	critical_health = LibVarGetValue("coopbot_danger_critical_health");
	if (critical_health <= 0.0f)
	{
		critical_health = 25.0f;
	}
	if (health > 0 && health <= critical_health)
	{
		return true;
	}

	return state->combat.took_damage &&
		AAS_Time() - state->combat.last_damage_time <= 2.0f;
}

/*
=============
BotInterface_RecordCoopSafeArea

Remember an actual no-enemy position rather than an abstract area center.  The
position is later usable as a normal bot_goal_t, so the mover/path planner can
decide whether the remembered spot is still reachable.
=============
*/
void BotInterface_RecordCoopSafeArea(bot_client_state_t *state,
	int area,
	int enemy_count)
{
	vec3_t delta;
	float distance;
	int health;
	float critical_health;

	if (state == NULL || area <= 0 || enemy_count != 0)
	{
		return;
	}

	/* The area observer runs before the derived battle inventory is refreshed. */
	health = state->last_client_update.stats[STAT_HEALTH];
	critical_health = LibVarGetValue("coopbot_danger_critical_health");
	if (critical_health <= 0.0f)
	{
		critical_health = 25.0f;
	}
	if ((health > 0 && health <= critical_health) ||
		(state->combat.took_damage &&
		 AAS_Time() - state->combat.last_damage_time <= 2.0f))
	{
		return;
	}

	VectorSubtract(state->last_client_update.origin,
		state->coop_last_safe_origin, delta);
	distance = sqrtf(DotProduct(delta, delta));
	if (state->coop_last_safe_valid &&
		state->coop_last_safe_area == area && distance < 64.0f)
	{
		return;
	}

	state->coop_last_safe_area = area;
	VectorCopy(state->last_client_update.origin,
		state->coop_last_safe_origin);
	state->coop_last_safe_time = AAS_Time();
	state->coop_last_safe_valid = true;
	if (LibVarGetValue("coopbot_log") >= 1.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_safe_area client=%d area=%d origin=(%.1f %.1f %.1f) "
			"state=%s",
			state->client_number,
			area,
			state->coop_last_safe_origin[0],
			state->coop_last_safe_origin[1],
			state->coop_last_safe_origin[2],
			BotInterface_CoopAreaStateName(state->coop_area_state));
	}
}
