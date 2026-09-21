#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "shared/q_platform.h"
#include "q2bridge/aas_translation.h"
#include "botlib/aas/aas_local.h"
#include "botlib/aas/aas_map.h"
#include "botlib/ea/ea_local.h"
#include "botlib/common/l_libvar.h"
#include "bot_interface.h"
#include "bot_interface_behavior.h"
#include "bot_state.h"

#define BOT_CONSOLE_SYNONYM_BASE 3UL
#define BOT_CONSOLE_SYNONYM_CTF_RED 7UL
#define BOT_CONSOLE_SYNONYM_CTF_BLUE 11UL

/*
=============
BotAI_LongTermGoalRandom

Returns the exact low-15-bit fraction emitted by the retail long-term-goal
branches, whose `1 / 32768` scale is distinct from console reply selection.
=============
*/
float BotAI_LongTermGoalRandom(void)
{
	return (float)(rand() & 0x7fff) * 3.05185094e-05f;
}

/*
=============
BotAI_TryRecentEnemyDeathWave

Replays Seek-LTG's short post-death gesture trial before it scans for a new
enemy.
=============
*/
void BotAI_TryRecentEnemyDeathWave(bot_client_state_t *state,
	float thinktime)
{
	if (state == NULL ||
		AAS_Time() - 5.0f >= state->combat.enemy_death_time ||
		BotAI_LongTermGoalRandom() >= thinktime)
	{
		return;
	}

	EA_Gesture(state->client_number,
		BotAI_LongTermGoalRandom() < 0.5f ? 0 : 2);
}

/*
=============
BotAI_RoamGoal

Reconstructs Gladiator's short-lived safe roam point used only for the
accompany formation idle-view branch. The candidate gates and distances are
the retail values rather than Quake III's later defaults.
=============
*/
void BotAI_RoamGoal(const bot_client_state_t *state, vec3_t goal)
{
	if (state == NULL || goal == NULL)
	{
		return;
	}

	vec3_t best_origin;
	VectorCopy(state->last_client_update.origin, best_origin);
	for (int attempt = 0; attempt < 10; ++attempt)
	{
		VectorCopy(state->last_client_update.origin, best_origin);
		float random_value = BotAI_LongTermGoalRandom();
		if (random_value < 0.8f)
		{
			float direction = BotAI_LongTermGoalRandom() < 0.5f ? -1.0f : 1.0f;
			best_origin[0] +=
				BotAI_LongTermGoalRandom() * direction * 700.0f + 50.0f;
		}
		if (random_value > 0.2f)
		{
			float direction = BotAI_LongTermGoalRandom() < 0.5f ? -1.0f : 1.0f;
			best_origin[1] +=
				BotAI_LongTermGoalRandom() * direction * 700.0f + 50.0f;
		}
		best_origin[2] += BotAI_LongTermGoalRandom() * 144.0f - 97.0f;

		bsp_trace_t trace = AAS_Trace(state->last_client_update.origin,
			NULL,
			NULL,
			best_origin,
			state->entity_number,
			MASK_SOLID);
		vec3_t direction;
		/*
		 * 0x10022bf1 subtracts the bot origin from the RANDOMIZED endpoint,
		 * before the trace result is copied out at 0x10022c03, so the length
		 * driving both the 100-unit gate and the 0x10022c5b rescale is the
		 * full untraced ray length rather than fraction * length.
		 */
		VectorSubtract(best_origin, state->last_client_update.origin, direction);
		float length = sqrtf(DotProduct(direction, direction));
		if (length <= 100.0f)
		{
			continue;
		}

		VectorScale(direction, 1.0f / length, direction);
		VectorScale(direction, length * trace.fraction - 40.0f, direction);
		VectorAdd(state->last_client_update.origin, direction, best_origin);

		vec3_t below_best_origin;
		VectorCopy(best_origin, below_best_origin);
		below_best_origin[2] -= 800.0f;
		trace = AAS_Trace(best_origin,
			NULL,
			NULL,
			below_best_origin,
			state->entity_number,
			MASK_SOLID);
		if (!trace.startsolid)
		{
			trace.endpos[2] += 1.0f;
			if ((AAS_PointContents(trace.endpos) &
				(CONTENTS_LAVA | CONTENTS_SLIME)) == 0)
			{
				break;
			}
		}
	}
	VectorCopy(best_origin, goal);
}

/*
=============
BotAI_CTFTeam

Return the bot's CTF team the way retail sub_10023510 does: 1 when its skin
contains "ctf_r", 2 otherwise.

0x10023523 calls strstr(ClientSkin(bs->client), "ctf_r") and 0x1002352b turns
the result into 1 or 2 with the neg/sbb idiom.  Retail derives this on demand
and never caches it in bot_state_t, so nothing here writes state->team.
=============
*/
int BotAI_CTFTeam(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return 2;
	}

	const char *skin = BotState_ClientSkin(state->client_number);
	return (skin != NULL && strstr(skin, "ctf_r") != NULL) ? 1 : 2;
}

/*
=============
BotAI_ConsoleSynonymContext

Builds Gladiator's normal/nearby synonym context and adds the CTF team bit
selected from the retail skin-name convention.
=============
*/
/*
=============
BotAI_ConsoleRandom

Returns the retail 15-bit random fraction used by console-message timing and
reply selection.
=============
*/
float BotAI_ConsoleRandom(void)
{
	return (float)(rand() & 0x7fff) / 32767.0f;
}


unsigned long BotAI_ConsoleSynonymContext(const bot_client_state_t *state)
{
	if (state == NULL || LibVarGetValue("ctf") == 0.0f)
	{
		return BOT_CONSOLE_SYNONYM_BASE;
	}

	const char *skin = BotState_ClientSkin(state->client_number);
	if (skin != NULL && strstr(skin, "ctf_r") != NULL)
	{
		return BOT_CONSOLE_SYNONYM_CTF_RED;
	}

	return BOT_CONSOLE_SYNONYM_CTF_BLUE;
}
