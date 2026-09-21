#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#include "shared/q_platform.h"
#include "q2bridge/aas_translation.h"
#include "q2bridge/bridge.h"
#include "botlib/aas/aas_map.h"
#include "botlib/aas/aas_local.h"
#include "botlib/aas/aas_sound.h"
#include "botlib/ai_character/bot_character.h"
#include "botlib/ai/ai_dm.h"
#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "botlib/common/l_utils.h"
#include "botlib/ea/ea_local.h"
#include "bot_interface.h"
#include "bot_interface_assets.h"
#include "bot_interface_combat.h"
#include "bot_interface_console.h"
#include "bot_interface_coop_state.h"
#include "bot_interface_runtime.h"
#include "bot_state.h"

#define CHARACTERISTIC_3D_ACCELERATOR 45
#define CHARACTERISTIC_WEAPONJUMPING 26
#define BOT_BATTLE_POWER_ARMOR_GRACE 0.9
#define BOT_CONSOLE_SKIN_TEAMS 0x40
#define BOT_CONSOLE_MODEL_TEAMS 0x80

/*
=============
BotInterface_SynchroniseCombatState

Copies the private DM metrics into the public combat state consumed by the
cooperative overlay and by the retained battle-node transitions.
=============
*/
void BotInterface_SynchroniseCombatState(bot_client_state_t *state)
{
	if (state == NULL || state->dm_state == NULL)
	{
		return;
	}

	ai_dm_metrics_t metrics = {0};
	AI_DMState_GetMetrics(state->dm_state, &metrics);

	bot_combat_state_t *combat = &state->combat;
	combat->revenge_enemy = metrics.revenge_enemy;
	combat->revenge_kills = metrics.revenge_kills;
	combat->enemy_visible = metrics.enemy_visible;
	combat->enemy_visible_time = metrics.enemyvisible_time;
	combat->enemy_death_time = metrics.enemydeath_time;
	combat->enemy_last_seen_time = metrics.enemyposition_time;
	VectorCopy(metrics.last_enemy_origin, combat->last_enemy_origin);
	VectorCopy(metrics.last_enemy_velocity, combat->last_enemy_velocity);
}

/*
=============
BotAI_UpdateBattleInventory

Reconstructs retail sub_10021020: health, timed powerups, and the short
power-armor activity window are projected into the fuzzy-weight inventory.
=============
*/
void BotAI_UpdateBattleInventory(bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	bot_updateclient_t *update = &state->last_client_update;
	int *inventory = update->inventory;
	const float now = BotInterface_CurrentFrameTime();

	inventory[BOT_BATTLE_INVENTORY_HEALTH] =
		(int)update->stats[STAT_HEALTH];

	int timer_image = (int)update->stats[STAT_TIMER_ICON];
	if (timer_image != 0)
	{
		const char *image_name = BotInterface_ImageNameForIndex(timer_image);
		float expiry = now + (float)update->stats[STAT_TIMER];
		if (Q_stricmp(image_name, "p_quad") == 0)
		{
			state->quad_time = expiry;
		}
		else if (Q_stricmp(image_name, "p_invulnerability") == 0)
		{
			state->invulnerability_time = expiry;
		}
		else if (Q_stricmp(image_name, "p_rebreather") == 0)
		{
			state->rebreather_time = expiry;
		}
		else if (Q_stricmp(image_name, "p_envirosuit") == 0)
		{
			state->environmentsuit_time = expiry;
		}
	}

	inventory[BOT_BATTLE_USING_QUAD] =
		(int)((double)state->quad_time - (double)now);
	if (inventory[BOT_BATTLE_USING_QUAD] <= 0)
	{
		inventory[BOT_BATTLE_USING_QUAD] = 0;
	}

	inventory[BOT_BATTLE_USING_INVULNERABILITY] =
		(int)((double)state->invulnerability_time - (double)now);
	if (inventory[BOT_BATTLE_USING_INVULNERABILITY] <= 0)
	{
		inventory[BOT_BATTLE_USING_INVULNERABILITY] = 0;
	}

	inventory[BOT_BATTLE_USING_REBREATHER] =
		(int)((double)state->rebreather_time - (double)now);
	if (inventory[BOT_BATTLE_USING_REBREATHER] <= 0)
	{
		inventory[BOT_BATTLE_USING_REBREATHER] = 0;
	}

	inventory[BOT_BATTLE_USING_ENVIRONMENTSUIT] =
		(int)((double)state->environmentsuit_time - (double)now);
	if (inventory[BOT_BATTLE_USING_ENVIRONMENTSUIT] <= 0)
	{
		inventory[BOT_BATTLE_USING_ENVIRONMENTSUIT] = 0;
	}

	int armor_image = (int)update->stats[STAT_ARMOR_ICON];
	if (armor_image != 0)
	{
		const char *image_name = BotInterface_ImageNameForIndex(armor_image);
		if (Q_stricmp(image_name, "i_powershield") == 0)
		{
			state->power_armor_time = now;
		}

		if ((double)state->power_armor_time >
			(double)now - BOT_BATTLE_POWER_ARMOR_GRACE)
		{
			int cells = inventory[BOT_BATTLE_INVENTORY_CELLS];
			inventory[BOT_BATTLE_USING_POWERSCREEN] = cells;
			inventory[BOT_BATTLE_USING_POWERSHIELD] = cells;
		}
		else
		{
			inventory[BOT_BATTLE_USING_POWERSCREEN] = 0;
			inventory[BOT_BATTLE_USING_POWERSHIELD] = 0;
		}
	}
}

/*
=============
BotAI_UpdateEnemyBattleInventory

Reconstructs retail sub_10021290: enemy displacement, Quake II weapon byte,
and the three observed effect flags are projected into battle inventory.
=============
*/
int BotAI_UpdateEnemyBattleInventory(bot_client_state_t *state,
	int enemy_entity)
{
	if (state == NULL)
	{
		return qfalse;
	}

	aas_entityinfo_t entity_info;
	AAS_EntityInfo(enemy_entity, &entity_info);

	vec3_t displacement;
	VectorSubtract(entity_info.origin,
		state->last_client_update.origin,
		displacement);
	int *inventory = state->last_client_update.inventory;
	inventory[BOT_BATTLE_ENEMY_HEIGHT] = (int)displacement[2];
	displacement[2] = 0.0f;
	double horizontal_square =
		(double)displacement[0] * (double)displacement[0] +
		(double)displacement[1] * (double)displacement[1];
	inventory[BOT_BATTLE_ENEMY_HORIZONTAL_DIST] =
		(int)sqrt(horizontal_square);

	memset(&inventory[BOT_BATTLE_ENEMY_BLASTER],
		0,
		12U * sizeof(inventory[0]));

	int weapon = (entity_info.skinnum >> 8) & 0xff;
	switch (weapon)
	{
		case 1:
			inventory[BOT_BATTLE_ENEMY_BLASTER] = 1;
			break;
		case 2:
			inventory[BOT_BATTLE_ENEMY_SHOTGUN] = 1;
			break;
		case 3:
			inventory[BOT_BATTLE_ENEMY_SUPERSHOTGUN] = 1;
			break;
		case 4:
			inventory[BOT_BATTLE_ENEMY_MACHINEGUN] = 1;
			break;
		case 5:
			inventory[BOT_BATTLE_ENEMY_CHAINGUN] = 1;
			break;
		case 6:
			inventory[BOT_BATTLE_ENEMY_GRENADES] = 1;
			break;
		case 7:
			inventory[BOT_BATTLE_ENEMY_GRENADELAUNCHER] = 1;
			break;
		case 8:
			inventory[BOT_BATTLE_ENEMY_ROCKETLAUNCHER] = 1;
			break;
		case 9:
			inventory[BOT_BATTLE_ENEMY_HYPERBLASTER] = 1;
			break;
		case 10:
			inventory[BOT_BATTLE_ENEMY_RAILGUN] = 1;
			break;
		case 11:
			inventory[BOT_BATTLE_ENEMY_BFG10K] = 1;
			break;
		case 12:
			inventory[BOT_BATTLE_ENEMY_GRAPPLE] = 1;
			break;
		default:
			break;
	}

	inventory[BOT_BATTLE_ENEMY_INVULNERABILITY] =
		(entity_info.effects & EF_PENT) != 0;
	inventory[BOT_BATTLE_ENEMY_QUAD] =
		(entity_info.effects & EF_QUAD) != 0;
	inventory[BOT_BATTLE_ENEMY_POWERSCREEN] =
		(entity_info.effects & EF_POWERSCREEN) != 0;
	return qtrue;
}

/*
=============
BotAI_UseItems

Reconstructs retail sub_10021500's four independent item-use branches and
their exact Silencer, liquid/Rebreather, Power Shield, Power Screen order.
=============
*/
void BotAI_UseItems(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	const int *inventory = state->last_client_update.inventory;
	if (inventory[BOT_BATTLE_INVENTORY_SILENCER] > 0)
	{
		EA_UseItem(state->client_number, "Silencer");
	}

	vec3_t eye;
	BotInterface_ClientEyePosition(state, eye);
	if ((Q2_PointContents(eye) & 0x38) != 0 &&
		inventory[BOT_BATTLE_USING_REBREATHER] == 0 &&
		inventory[BOT_BATTLE_INVENTORY_REBREATHER] > 0)
	{
		EA_UseItem(state->client_number, "Rebreather");
	}

	if (inventory[BOT_BATTLE_USING_POWERSHIELD] == 0 &&
		inventory[BOT_BATTLE_INVENTORY_POWERSHIELD] > 0)
	{
		EA_UseItem(state->client_number, "Power Shield");
	}

	if (inventory[BOT_BATTLE_USING_POWERSCREEN] == 0 &&
		inventory[BOT_BATTLE_INVENTORY_POWERSCREEN] > 0)
	{
		EA_UseItem(state->client_number, "Power Screen");
	}
}

/*
=============
BotAI_BattleUseItems

Reconstructs retail sub_100215e0's Quad-first early return and subsequent
Invulnerability fallback over raw battle-inventory slots.
=============
*/
void BotAI_BattleUseItems(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return;
	}

	const int *inventory = state->last_client_update.inventory;
	if (inventory[BOT_BATTLE_USING_QUAD] == 0 &&
		inventory[BOT_BATTLE_INVENTORY_QUAD] > 0)
	{
		EA_UseItem(state->client_number, "Quad Damage");
		return;
	}

	if (inventory[BOT_BATTLE_USING_INVULNERABILITY] == 0 &&
		inventory[BOT_BATTLE_INVENTORY_INVULNERABILITY] > 0)
	{
		EA_UseItem(state->client_number, "Invulnerability");
	}
}

/*
=============
BotAI_CarryingFlag

Reconstructs retail sub_10021650, including the ordered-nonzero CTF gate and
the distinct flag-one/flag-two return values.
=============
*/
int BotAI_CarryingFlag(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return 0;
	}

	float ctf = LibVarGetValue("ctf");
	if (ctf == 0.0f || isnan(ctf))
	{
		return 0;
	}

	const int *inventory = state->last_client_update.inventory;
	if (inventory[BOT_BATTLE_INVENTORY_FLAG1] > 0)
	{
		return 1;
	}
	if (inventory[BOT_BATTLE_INVENTORY_FLAG2] > 0)
	{
		return 2;
	}

	return 0;
}

/*
=============
BotAI_Aggression

Reconstructs retail sub_100226c0's ordered powerup, height, health, armor,
weapon, and ammunition gates over the battle inventory.
=============
*/
float BotAI_Aggression(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return 0.0f;
	}

	const int *inventory = state->last_client_update.inventory;
	if (inventory[BOT_BATTLE_USING_INVULNERABILITY] != 0)
	{
		return 100.0f;
	}

	if (inventory[BOT_BATTLE_ENEMY_INVULNERABILITY] != 0)
	{
		return 0.0f;
	}
	if (inventory[BOT_BATTLE_ENEMY_QUAD] != 0 &&
		inventory[BOT_BATTLE_USING_QUAD] == 0)
	{
		return 0.0f;
	}
	if (inventory[BOT_BATTLE_ENEMY_POWERSCREEN] != 0 &&
		(inventory[BOT_BATTLE_USING_POWERSCREEN] == 0 ||
		inventory[BOT_BATTLE_INVENTORY_CELLS] < 50))
	{
		return 0.0f;
	}

	if (inventory[BOT_BATTLE_ENEMY_HEIGHT] > 200)
	{
		return 0.0f;
	}

	int health = inventory[BOT_BATTLE_INVENTORY_HEALTH];
	if (health < 40)
	{
		return 0.0f;
	}
	if (health < 70 &&
		inventory[BOT_BATTLE_INVENTORY_ARMORBODY] < 40 &&
		inventory[BOT_BATTLE_INVENTORY_ARMORCOMBAT] < 50 &&
		inventory[BOT_BATTLE_INVENTORY_ARMORJACKET] < 60)
	{
		return 0.0f;
	}

	if (inventory[BOT_BATTLE_INVENTORY_BFG10K] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_CELLS] > 50)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_RAILGUN] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_SLUGS] > 5)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_HYPERBLASTER] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_CELLS] > 50)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_ROCKETLAUNCHER] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_ROCKETS] > 5)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_GRENADELAUNCHER] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_GRENADES] > 10)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_CHAINGUN] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_BULLETS] > 100)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_MACHINEGUN] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_BULLETS] > 75)
	{
		return 100.0f;
	}
	if (inventory[BOT_BATTLE_INVENTORY_SUPERSHOTGUN] > 0 &&
		inventory[BOT_BATTLE_INVENTORY_SHELLS] > 20)
	{
		return 100.0f;
	}

	return 0.0f;
}

/*
=============
BotAI_CoopDangerScore

Builds the opt-in companion danger estimate from survivability, recent
incoming damage, visible enemy pressure, and separation from the player.
The retail aggression result remains authoritative until the coop overlay is
enabled.
=============
*/
float BotAI_CoopDangerScore(const bot_client_state_t *state)
{
	const int *inventory;
	float score = 0.0f;
	float health_score;
	float armor_score;
	float now;
	float critical_health;
	int health;
	int armor;

	if (state == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_danger_retreat") == 0.0f)
	{
		return 0.0f;
	}
	if (state->combat.current_enemy <= 0)
	{
		return 0.0f;
	}

	inventory = state->last_client_update.inventory;
	if (inventory[BOT_BATTLE_USING_INVULNERABILITY] != 0)
	{
		return 0.0f;
	}

	health = inventory[BOT_BATTLE_INVENTORY_HEALTH];
	if (health <= 0)
	{
		return 1.0f;
	}
	critical_health = LibVarGetValue("coopbot_danger_critical_health");
	if (critical_health <= 0.0f)
	{
		critical_health = 25.0f;
	}
	if (health <= critical_health)
	{
		return 1.0f;
	}

	health_score = health < 70 ? (70.0f - (float)health) / 70.0f : 0.0f;
	armor = inventory[BOT_BATTLE_INVENTORY_ARMORBODY] +
		inventory[BOT_BATTLE_INVENTORY_ARMORCOMBAT] +
		inventory[BOT_BATTLE_INVENTORY_ARMORJACKET];
	armor_score = armor < 80 ? (80.0f - (float)armor) / 80.0f : 0.0f;
	score += fminf(fmaxf(health_score, 0.0f), 1.0f) * 0.35f;
	score += fminf(fmaxf(armor_score, 0.0f), 1.0f) * 0.15f;

	now = AAS_Time();
	if (state->combat.took_damage &&
		now - state->combat.last_damage_time <= 2.0f)
	{
		float damage_score = (float)state->combat.last_damage_amount / 50.0f;
		score += fminf(fmaxf(damage_score, 0.0f), 1.0f) * 0.20f;
	}

	if (aasworld.loaded && aasworld.entities != NULL)
	{
		vec3_t eye;
		vec3_t viewangles;
		int visible_entities[16];
		int visible_count;
		int enemy_count = 0;
		int shooting_count = 0;

		BotInterface_ClientEyePosition(state, eye);
		VectorClear(viewangles);
		visible_count = AAS_VisibleEntities(state->entity_number,
			eye,
			viewangles,
			360.0f,
			16,
			visible_entities);
		for (int index = 0; index < visible_count; ++index)
		{
			aas_entityinfo_t entity_info;

			AAS_EntityInfo(visible_entities[index], &entity_info);
			if (BotAI_EntityIsDead(&entity_info) ||
				entity_info.number <= aasworld.maxClients ||
				entity_info.number == state->entity_number)
			{
				continue;
			}
			enemy_count += 1;
			if (BotAI_EntityIsShooting(&entity_info))
			{
				shooting_count += 1;
			}
		}

		float enemy_pressure = (float)enemy_count / 4.0f;
		score += fminf(enemy_pressure, 1.0f) * 0.25f;
		score += fminf((float)shooting_count, 1.0f) * 0.10f;
	}

	if (aasworld.initialized && aasworld.entities != NULL)
	{
		aas_entityinfo_t enemy_info;
		memset(&enemy_info, 0, sizeof(enemy_info));
		AAS_EntityInfo(state->combat.current_enemy, &enemy_info);
		if (!BotAI_EntityIsDead(&enemy_info))
		{
			vec3_t direction;
			VectorSubtract(enemy_info.origin,
				state->last_client_update.origin,
				direction);
			float distance = sqrtf(DotProduct(direction, direction));
			if (distance < 256.0f)
			{
				score += (256.0f - distance) / 256.0f * 0.10f;
			}
			if (BotAI_EntityIsShooting(&enemy_info))
			{
				score += 0.10f;
			}
		}
	}

	if (aasworld.initialized && aasworld.entities != NULL)
	{
		vec3_t player_origin;
		int player_entity = BotAI_CoopPlayerEntity(state, player_origin);
		if (player_entity >= 0)
		{
			vec3_t direction;
			float soft_leash = LibVarGetValue("coopbot_soft_leash");
			float hard_leash = LibVarGetValue("coopbot_hard_leash");
			VectorSubtract(state->last_client_update.origin,
				player_origin,
				direction);
			float distance = sqrtf(DotProduct(direction, direction));
			if (hard_leash <= soft_leash)
			{
				hard_leash = soft_leash + 1.0f;
			}
			if (distance > soft_leash)
			{
				score += fminf((distance - soft_leash) /
					(hard_leash - soft_leash), 1.0f) * 0.15f;
			}
		}
	}

	return fminf(fmaxf(score, 0.0f), 1.0f);
}

/*
=============
BotAI_CoopDangerRequiresRetreat

Applies the configured danger threshold and preserves the critical-health
override as a hard safety boundary for the companion overlay.
=============
*/
bool BotAI_CoopDangerRequiresRetreat(const bot_client_state_t *state)
{
	float threshold;

	if (state == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_danger_retreat") == 0.0f)
	{
		return false;
	}
	threshold = LibVarGetValue("coopbot_danger_threshold");
	if (threshold <= 0.0f)
	{
		threshold = 0.65f;
	}
	return BotAI_CoopDangerScore(state) >= threshold;
}

/*
=============
BotAI_WantsToRetreat

Reconstructs retail sub_100228c0's flag, get-flag LTG, and strict aggression
threshold gates.
=============
*/
int BotAI_WantsToRetreat(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return qfalse;
	}

	if (BotAI_CarryingFlag(state) != 0)
	{
		return qtrue;
	}
	if (state->ltg_type == BOT_LTG_GET_FLAG)
	{
		return qtrue;
	}
	if (BotAI_CoopDangerRequiresRetreat(state))
	{
		return qtrue;
	}

	return BotAI_Aggression(state) < 50.0f;
}

/*
=============
BotAI_WantsToChase

Reconstructs retail sub_10022930's strict aggression threshold without the
additional flag and LTG special cases introduced by the Quake III successor.
=============
*/
int BotAI_WantsToChase(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return qfalse;
	}
	if (BotAI_CoopDangerRequiresRetreat(state))
	{
		return qfalse;
	}

	return BotAI_Aggression(state) > 50.0f;
}

/*
=============
BotAI_CanAndWantsToRocketJump

Reconstructs retail sub_10022990's ordered rocket, powerup, survivability,
and weapon-jumping characteristic gates over the raw battle inventory.
=============
*/
int BotAI_CanAndWantsToRocketJump(const bot_client_state_t *state)
{
	if (state == NULL)
	{
		return qfalse;
	}

	const int *inventory = state->last_client_update.inventory;
	if (inventory[BOT_BATTLE_INVENTORY_ROCKETLAUNCHER] <= 0)
	{
		return qfalse;
	}
	if (inventory[BOT_BATTLE_INVENTORY_ROCKETS] < 3)
	{
		return qfalse;
	}
	if (inventory[BOT_BATTLE_USING_QUAD] != 0)
	{
		return qfalse;
	}

	if (inventory[BOT_BATTLE_USING_INVULNERABILITY] != 0)
	{
		return qtrue;
	}

	int health = inventory[BOT_BATTLE_INVENTORY_HEALTH];
	if (health < 60)
	{
		return qfalse;
	}
	if (health < 90 &&
		inventory[BOT_BATTLE_INVENTORY_ARMORBODY] < 40 &&
		inventory[BOT_BATTLE_INVENTORY_ARMORCOMBAT] < 50 &&
		inventory[BOT_BATTLE_INVENTORY_ARMORJACKET] < 60)
	{
		return qfalse;
	}

	float weapon_jumping = Characteristic_BFloat(state->character,
		CHARACTERISTIC_WEAPONJUMPING,
		0.0f,
		1.0f);
	return weapon_jumping >= 0.5f;
}
void BotAI_InitEnemyInfo(ai_dm_enemy_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    info->valid = false;
    info->visible = false;
    info->entity = -1;
    VectorClear(info->origin);
    VectorClear(info->velocity);
    VectorClear(info->lastvisorigin);
    info->update_time = 0.0f;
    info->distance = 0.0f;
    info->last_seen_time = -FLT_MAX;
    info->field_of_view = 0.0f;
    info->is_invisible = false;
    info->is_chatting = false;
    info->is_shooting = false;
    info->triggered_by_damage = false;
    info->in_field_of_view = false;
    info->has_line_of_sight = false;
}

/*
=============
BotInterface_InFieldOfVision

Applies retail's 16-bit angle quantization and inclusive pitch/yaw half-FOV.
=============
*/
bool BotInterface_InFieldOfVision(const vec3_t viewangles,
	float field_of_view,
	const vec3_t target_angles)
{
	for (int axis = 0; axis < 2; ++axis)
	{
		float view_angle = AngleMod(viewangles[axis]);
		float target_angle = AngleMod(target_angles[axis]);
		float difference = target_angle - view_angle;
		if (target_angle < view_angle)
		{
			if (difference < -180.0f)
			{
				difference += 360.0f;
			}
		}
		else if (difference > 180.0f)
		{
			difference -= 360.0f;
		}

		if (difference < -field_of_view * 0.5f ||
			difference > field_of_view * 0.5f)
		{
			return false;
		}
	}

	return true;
}

/*
=============
BotInterface_ClientEyePosition

Builds the eye position retained at bot-state offset 0x6b0 in retail.
=============
*/
void BotInterface_ClientEyePosition(const bot_client_state_t *state,
	vec3_t out)
{
	if (state == NULL || out == NULL)
	{
		return;
	}

	VectorCopy(state->last_client_update.origin, out);
	for (int axis = 0; axis < 3; ++axis)
	{
		out[axis] += state->last_client_update.viewoffset[axis];
	}
}

/*
=============
BotInterface_HasLineOfSight

Runs the shared point trace used by reconstructed console-goal visibility.
=============
*/
bool BotInterface_HasLineOfSight(const vec3_t from,
	const vec3_t to,
	int viewer,
	int target)
{
	if (from == NULL || to == NULL)
	{
		return false;
	}

	vec3_t start;
	vec3_t end;
	VectorCopy(from, start);
	VectorCopy(to, end);

	vec3_t mins = {0.0f, 0.0f, 0.0f};
	vec3_t maxs = {0.0f, 0.0f, 0.0f};
	bsp_trace_t trace = Q2_Trace(start, mins, maxs, end, viewer, MASK_SHOT);
	return trace.fraction >= 1.0f || trace.ent == target;
}

/*
=============
BotAI_EntityIsDead

Reconstructs retail sub_10021710's Quake II client/live-frame predicate.
=============
*/
int BotAI_EntityIsDead(const aas_entityinfo_t *entity_info)
{
	if (entity_info == NULL || !entity_info->valid)
	{
		return qtrue;
	}

	if (LibVarGetValue("coop") != 0.0f)
	{
		/* Monsters are bbox entities. This excludes pickups, triggers and
		 * brush movers from the coop enemy scan. */
		return entity_info->number < 1 ||
			entity_info->modelindex <= 0 ||
			entity_info->solid != SOLID_BBOX ||
			(entity_info->effects & (EF_GIB | EF_FLIES)) != 0;
	}

	if ((entity_info->effects & (EF_GIB | EF_FLIES)) != 0 ||
		entity_info->number < 1 ||
		entity_info->number > aasworld.maxClients ||
		entity_info->modelindex != 255)
	{
		return qtrue;
	}

	return entity_info->frame >= 173 && entity_info->frame <= 197;
}

/*
=============
BotAI_EntityIsShooting

Reconstructs retail sub_10021780's player attack-animation predicate.
=============
*/
int BotAI_EntityIsShooting(const aas_entityinfo_t *entity_info)
{
	return entity_info->modelindex == 255 &&
		entity_info->frame >= 46 && entity_info->frame <= 53;
}

/*
=============
BotAI_ModelTeamMatches

Compares the model prefix before the Quake II model/skin separator.

Retail's DF_MODELTEAMS arm (0x100236ec..0x100237ab) ends in StrCompareN @
0x100456b0 - the raw repe cmpsb strncmp - so this compare is CASE-SENSITIVE,
unlike the DF_SKINTEAMS/ctf and teamplay arms, which both go through the
case-folding sub_10045cb0 (_strcmpi).  ref be_ai2_dmq2.c:1214 vs :1202/:1190.
=============
*/
static int BotAI_ModelTeamMatches(const char *left, const char *right)
{
	const char *left_separator = strchr(left, '/');
	const char *right_separator = strchr(right, '/');
	size_t left_length = left_separator != NULL
		? (size_t)(left_separator - left)
		: strlen(left);
	size_t right_length = right_separator != NULL
		? (size_t)(right_separator - right)
		: strlen(right);
	return left_length == right_length &&
		strncmp(left, right, left_length) == 0;
}

/*
=============
BotAI_SkinTeamMatches

Compares the suffix beginning at the Quake II model/skin separator.
=============
*/
static int BotAI_SkinTeamMatches(const char *left, const char *right)
{
	const char *left_separator = strchr(left, '/');
	const char *right_separator = strchr(right, '/');
	return Q_stricmp(left_separator != NULL ? left_separator : left,
		right_separator != NULL ? right_separator : right) == 0;
}

/*
=============
BotAI_LibVarOrderedNonZero

Mirrors retail's ordered x87 comparison so an unordered NaN is treated as off.
=============
*/
int BotAI_LibVarOrderedNonZero(const char *name)
{
	float value = LibVarGetValue(name);
	return value < 0.0f || value > 0.0f;
}

int BotAI_CoopMode(void)
{
	return BotAI_LibVarOrderedNonZero("coop");
}

/*
=============
BotAI_CoopRejectNewGroup

The first coop new-group guard is deliberately conservative: when the bot
has lost the human's soft leash, do not let a non-urgent distant monster start
a fresh encounter. Damage and an observed attack animation remain immediate
threat overrides. This is a distance-based P1 guard, not full encounter
clustering; the latter still needs runtime event/group data.
=============
*/
static bool BotAI_CoopRejectNewGroup(const bot_client_state_t *state,
	const aas_entityinfo_t *candidate,
	int health_decrease,
	bool active_group)
{
	vec3_t player_origin;
	vec3_t direction;
	float distance;
	float soft_leash;

	if (state == NULL || candidate == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_new_group_guard") == 0.0f ||
		state->combat.current_enemy > 0)
	{
		return false;
	}
	if (health_decrease || active_group || BotAI_EntityIsShooting(candidate))
	{
		return false;
	}
	if (BotAI_CoopPlayerEntity(state, player_origin) < 0)
	{
		return false;
	}

	soft_leash = LibVarGetValue("coopbot_soft_leash");
	if (soft_leash <= 0.0f)
	{
		return false;
	}

	VectorSubtract(state->last_client_update.origin, player_origin, direction);
	distance = sqrtf(DotProduct(direction, direction));
	if (distance <= soft_leash)
	{
		return false;
	}
	if (BotAI_CoopRoleBlocksNewGroup(state))
	{
		return true;
	}
	if (BotAI_CoopIntentAllowsForwardProgress(state))
	{
		/* A confident advance/explore is permission to join the next group. */
		return false;
	}

	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"coopbot_new_group_blocked client=%d entity=%d distance=%.1f soft_leash=%.1f",
			state->client_number, candidate->number, distance, soft_leash);
	}
	return true;
}

/*
=============
BotAI_CoopCandidateInActiveGroup

Treats a visible monster near another visible shooting monster as part of the
current encounter. This is the first runtime enemy-group approximation: it
does not invent persistent map groups, but it prevents the soft-leash guard
from rejecting a nearby member of an already active fight.
=============
*/
static bool BotAI_CoopCandidateInActiveGroup(
	const aas_entityinfo_t *candidate,
	const int *visible_entities,
	int visible_count)
{
	float group_radius;
	int index;

	if (candidate == NULL || visible_entities == NULL ||
		visible_count <= 0 || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_new_group_guard") == 0.0f)
	{
		return false;
	}
	if (BotAI_EntityIsShooting(candidate))
	{
		return true;
	}

	group_radius = LibVarGetValue("coopbot_enemy_group_radius");
	if (group_radius <= 0.0f)
	{
		group_radius = 384.0f;
	}
	for (index = 0; index < visible_count; ++index)
	{
		aas_entityinfo_t other;
		vec3_t direction;
		float distance;

		if (visible_entities[index] <= 0 ||
			visible_entities[index] == candidate->number)
		{
			continue;
		}
		memset(&other, 0, sizeof(other));
		AAS_EntityInfo(visible_entities[index], &other);
		if (BotAI_EntityIsDead(&other) ||
			!BotAI_EntityIsShooting(&other))
		{
			continue;
		}
		VectorSubtract(candidate->origin, other.origin, direction);
		distance = sqrtf(DotProduct(direction, direction));
		if (distance <= group_radius)
		{
			return true;
		}
	}

	return false;
}

/*
=============
BotAI_SameTeam

Reconstructs retail sub_10023550's shell, CH, teamplay, CTF, skin-team, and
model-team precedence for a candidate entity number.
=============
*/
int BotAI_SameTeam(const bot_client_state_t *state, int entity)
{
	if (state == NULL)
	{
		return qfalse;
	}

	aas_entityinfo_t candidate_info;
	AAS_EntityInfo(entity, &candidate_info);
	if (candidate_info.number == 0)
	{
		return qfalse;
	}
	if (BotAI_CoopMode() && candidate_info.number > aasworld.maxClients)
	{
		return qfalse;
	}

	aas_entityinfo_t self_info;
	if (BotAI_LibVarOrderedNonZero("teamplay_shell"))
	{
		AAS_EntityInfo(state->entity_number, &self_info);
		return ((candidate_info.renderfx ^ self_info.renderfx) & 0x1c00) == 0;
	}

	if (BotAI_LibVarOrderedNonZero("ch"))
	{
		AAS_EntityInfo(state->entity_number, &self_info);
		return self_info.modelindex3 != candidate_info.modelindex3;
	}

	const char *self_skin = BotState_ClientSkin(state->client_number);
	const char *candidate_skin = BotState_ClientSkin(candidate_info.number - 1);
	if (BotAI_LibVarOrderedNonZero("teamplay"))
	{
		return Q_stricmp(self_skin, candidate_skin) == 0;
	}

	int dmflags = (int)LibVarGetValue("dmflags");
	if ((dmflags & BOT_CONSOLE_SKIN_TEAMS) != 0 ||
		BotAI_LibVarOrderedNonZero("ctf"))
	{
		return BotAI_SkinTeamMatches(self_skin, candidate_skin);
	}
	if ((dmflags & BOT_CONSOLE_MODEL_TEAMS) != 0)
	{
		return BotAI_ModelTeamMatches(self_skin, candidate_skin);
	}

	return qfalse;
}

/*
=============
BotAI_AcceptEnemy

Copies the accepted AAS record into the local DM handoff and performs the two
state writes made by retail BotFindEnemy.
=============
*/
static int BotAI_AcceptEnemy(bot_client_state_t *state,
	const aas_entityinfo_t *entity_info,
	float distance,
	float field_of_view,
	int health_decrease,
	ai_dm_enemy_info_t *enemy)
{
	float now = AAS_Time();
	int previous_enemy = state != NULL ? state->combat.current_enemy : 0;
	if (enemy != NULL)
	{
		enemy->valid = true;
		enemy->visible = true;
		enemy->entity = entity_info->number;
		VectorCopy(entity_info->origin, enemy->origin);
		VectorSubtract(entity_info->origin,
			entity_info->old_origin,
			enemy->velocity);
		VectorCopy(entity_info->lastvisorigin, enemy->lastvisorigin);
		enemy->update_time = entity_info->update_time;
		enemy->distance = distance;
		enemy->last_seen_time = now;
		enemy->field_of_view = field_of_view;
		enemy->is_invisible = false;
		enemy->is_chatting = false;
		enemy->is_shooting = BotAI_EntityIsShooting(entity_info) != 0;
		enemy->triggered_by_damage = health_decrease != 0;
		enemy->in_field_of_view = true;
		enemy->has_line_of_sight = true;
	}

	state->combat.current_enemy = entity_info->number;
	state->combat.enemy_sight_time = now;
	if (LibVarGetValue("coopbot_log") >= 2.0f)
	{
		BotLib_LogWriteTimeStamped(
			"event=enemy client=%d name=\"%s\" entity=%d distance=%.1f fov=%.1f damaged=%d",
			state->client_number,
			BotState_ClientName(state->client_number),
			entity_info->number, distance, field_of_view, health_decrease);
		if (BotAI_CoopMode() && previous_enemy != entity_info->number)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_decision client=%d decision=TARGET_SELECT target=%d "
				"current_target=%d player_intent=%s confidence=%.2f reason=%s",
				state->client_number,
				entity_info->number,
				previous_enemy,
				BotAI_CoopPlayerIntentName(state->coop_player_intent),
				state->coop_player_intent_confidence,
				health_decrease ? "damaged" :
				BotAI_EntityIsShooting(entity_info) ? "threat" : "scan");
		}
	}
	return qtrue;
}

/*
=============
BotAI_CoopTargetUtility

Returns the first coop target utility approximation: proximity is the base,
while a shooting monster and membership in an active group increase urgency.
The value is intentionally local to the coop overlay and does not alter the
retail enemy scan when target hysteresis is disabled.
=============
*/
static float BotAI_CoopTargetUtility(const bot_client_state_t *state,
	const aas_entityinfo_t *candidate,
	float distance,
	bool active_group)
{
	float utility;

	if (state == NULL || candidate == NULL)
	{
		return 0.0f;
	}

	utility = 1.0f / (distance + 64.0f);
	if (BotAI_EntityIsShooting(candidate))
	{
		utility += 0.02f;
	}
	if (active_group)
	{
		utility += 0.01f;
	}
	if (LibVarGetValue("coopbot_shared_focus") != 0.0f &&
		candidate->number == state->coop_player_focus_entity)
	{
		utility += 0.03f;
	}
	if (candidate->number == state->combat.current_enemy)
	{
		utility += 0.01f;
	}
	return utility;
}

static bool BotAI_CoopYieldPlayerFocus(const bot_client_state_t *state,
	const aas_entityinfo_t *candidate,
	float distance,
	int health_decrease)
{
	float radius;

	if (state == NULL || candidate == NULL ||
		LibVarGetValue("coopbot_shared_focus") == 0.0f ||
		LibVarGetValue("coopbot_kill_steal_control") == 0.0f ||
		candidate->number != state->coop_player_focus_entity ||
		candidate->number == state->combat.current_enemy)
	{
		return false;
	}
	if (health_decrease || BotAI_EntityIsShooting(candidate))
	{
		return false;
	}
	radius = LibVarGetValue("coopbot_kill_steal_radius");
	if (radius <= 0.0f)
	{
		radius = 192.0f;
	}
	return distance > radius;
}

/*
=============
BotAI_CoopTargetAcquisitionAllowed

Keeps a newly noticed coop target pending for a bounded human-like reaction
interval.  Damage and an actively shooting threat remain immediate so the
delay cannot make the companion ignore an urgent attack.
=============
*/
static bool BotAI_CoopTargetAcquisitionAllowed(
	bot_client_state_t *state,
	const aas_entityinfo_t *candidate,
	int health_decrease)
{
	float delay;
	float now;

	if (state == NULL || candidate == NULL || !BotAI_CoopMode())
	{
		return true;
	}

	delay = LibVarGetValue("coopbot_target_acquisition_delay");
	if (delay <= 0.0f || health_decrease ||
		BotAI_EntityIsShooting(candidate) ||
		candidate->number == state->combat.current_enemy)
	{
		state->coop_target_candidate_entity = 0;
		state->coop_target_candidate_time = 0.0f;
		return true;
	}

	now = AAS_Time();
	if (state->coop_target_candidate_entity != candidate->number)
	{
		if (state->coop_target_candidate_entity == 0 ||
			now - state->coop_target_candidate_time >= delay)
		{
			state->coop_target_candidate_entity = candidate->number;
			state->coop_target_candidate_time = now;
			if (LibVarGetValue("coopbot_log") >= 2.0f)
			{
				BotLib_LogWriteTimeStamped(
					"coopbot_target_pending client=%d target=%d delay=%.3f",
					state->client_number,
					candidate->number,
					delay);
			}
		}
		return false;
	}

	if (now - state->coop_target_candidate_time < delay)
	{
		return false;
	}

	state->coop_target_candidate_entity = 0;
	state->coop_target_candidate_time = 0.0f;
	return true;
}

/*
=============
BotAI_CoopTargetSwitchAllowed

Applies target hysteresis only when a coop scan is trying to replace an
existing enemy. Immediate threats always interrupt; otherwise the candidate
must beat the retained target utility by the configured ratio.
=============
*/
static bool BotAI_CoopTargetSwitchAllowed(bot_client_state_t *state,
	const aas_entityinfo_t *candidate,
	float distance,
	bool active_group,
	int health_decrease)
{
	aas_entityinfo_t current_info;
	float current_distance;
	float candidate_utility;
	float current_utility;
	float switch_ratio;
	vec3_t direction;

	if (!BotAI_CoopTargetAcquisitionAllowed(state, candidate, health_decrease))
	{
		return false;
	}
	if (state == NULL || candidate == NULL || !BotAI_CoopMode() ||
		LibVarGetValue("coopbot_target_hysteresis") == 0.0f ||
		state->combat.current_enemy <= 0 ||
		candidate->number == state->combat.current_enemy)
	{
		return true;
	}
	if (health_decrease || BotAI_EntityIsShooting(candidate))
	{
		return true;
	}

	memset(&current_info, 0, sizeof(current_info));
	AAS_EntityInfo(state->combat.current_enemy, &current_info);
	if (!current_info.valid || BotAI_EntityIsDead(&current_info))
	{
		return true;
	}

	VectorSubtract(current_info.origin,
		state->last_client_update.origin,
		direction);
	current_distance = sqrtf(DotProduct(direction, direction));
	if (current_distance <= 0.0f)
	{
		VectorSubtract(state->combat.last_enemy_origin,
			state->last_client_update.origin,
			direction);
		current_distance = sqrtf(DotProduct(direction, direction));
	}
	candidate_utility = BotAI_CoopTargetUtility(state,
		candidate,
		distance,
		active_group);
	current_utility = BotAI_CoopTargetUtility(state,
		&current_info,
		current_distance,
		false);
	switch_ratio = LibVarGetValue("coopbot_target_switch_ratio");
	if (switch_ratio <= 1.0f)
	{
		switch_ratio = 1.25f;
	}
	if (candidate_utility <= current_utility * switch_ratio)
	{
		if (LibVarGetValue("coopbot_log") >= 2.0f)
		{
			BotLib_LogWriteTimeStamped(
				"coopbot_target_switch_blocked client=%d current=%d candidate=%d "
				"current_utility=%.5f candidate_utility=%.5f ratio=%.2f",
				state->client_number,
				state->combat.current_enemy,
				candidate->number,
				current_utility,
				candidate_utility,
				switch_ratio);
			BotLib_LogWriteTimeStamped(
				"coopbot_decision client=%d decision=TARGET_YIELD target=%d "
				"current_target=%d candidate_utility=%.5f "
				"current_utility=%.5f ratio=%.2f reason=hysteresis_block",
				state->client_number,
				candidate->number,
				state->combat.current_enemy,
				candidate_utility,
				current_utility,
				switch_ratio);
		}
		return false;
	}

	return true;
}

/*
=============
BotAI_FindEnemy

Reconstructs retail sub_10023970's ascending visible-client scan, exact
distance/FOV/team/light gates, and retreat fallback.
=============
*/
int BotAI_FindEnemy(bot_client_state_t *state, ai_dm_enemy_info_t *enemy)
{
	BotAI_InitEnemyInfo(enemy);
	if (state == NULL)
	{
		return qfalse;
	}

	int accelerator_3d = Characteristic_BInteger(state->character,
		CHARACTERISTIC_3D_ACCELERATOR,
		0,
		1);
	int current_health = state->last_client_update.inventory[
		BOT_BATTLE_INVENTORY_HEALTH];
	int health_decrease = state->combat.last_health_valid &&
		state->combat.last_known_health > current_health;
	if (health_decrease)
	{
		state->combat.last_damage_amount =
			state->combat.last_known_health - current_health;
		state->combat.last_damage_time = AAS_Time();
		state->combat.took_damage = true;
	}
	else if (state->combat.took_damage &&
		AAS_Time() - state->combat.last_damage_time > 2.0f)
	{
		state->combat.took_damage = false;
	}
	state->combat.last_known_health = current_health;
	state->combat.last_health_valid = true;
	vec3_t eye;
	BotInterface_ClientEyePosition(state, eye);
	vec3_t viewangles;
	if (!AI_DMState_GetViewAngles(state->dm_state, viewangles))
	{
		VectorClear(viewangles);
	}

	int visible_entities[16];
	int visible_count = AAS_VisibleEntities(state->entity_number,
		eye,
		viewangles,
		360.0f,
		16,
		visible_entities);
	bool shared_focus = BotAI_CoopMode() &&
		LibVarGetValue("coopbot_shared_focus") != 0.0f &&
		state->coop_player_focus_entity > 0;
	int scan_passes = shared_focus ? 2 : 1;
	for (int pass = 0; pass < scan_passes; ++pass)
	{
		for (int index = 0; index < visible_count; ++index)
		{
			aas_entityinfo_t entity_info;
			bool active_group;
			AAS_EntityInfo(visible_entities[index], &entity_info);
			if (shared_focus &&
				((pass == 0 && entity_info.number !=
					state->coop_player_focus_entity) ||
					(pass == 1 && entity_info.number ==
					state->coop_player_focus_entity)))
			{
				continue;
			}
		if (BotAI_EntityIsDead(&entity_info) ||
			entity_info.number == state->entity_number ||
			(BotAI_CoopMode() && entity_info.number <= aasworld.maxClients))
		{
			continue;
		}

		vec3_t direction;
		VectorSubtract(entity_info.origin,
			state->last_client_update.origin,
			direction);
		float distance = sqrtf(DotProduct(direction, direction));
		if (BotAI_CoopYieldPlayerFocus(state,
			&entity_info,
			distance,
			health_decrease))
		{
			continue;
		}
		if (!accelerator_3d && distance > 900.0f)
		{
			continue;
		}

		float field_of_view = health_decrease
			? 360.0f
			: 90.0f + (distance > 810.0f ? 810.0f : distance) / 3.0f;
		vec3_t target_angles;
		Vector2Angles(direction, target_angles);
		if (!BotInterface_InFieldOfVision(viewangles,
			field_of_view,
			target_angles) ||
			BotAI_SameTeam(state, entity_info.number))
		{
			continue;
		}
		active_group = BotAI_CoopCandidateInActiveGroup(&entity_info,
			visible_entities,
			visible_count);
		if (BotAI_CoopRejectNewGroup(state,
			&entity_info,
			health_decrease,
			active_group))
		{
			continue;
		}

		if (health_decrease && !(distance > 300.0f))
		{
			if (!BotAI_CoopTargetSwitchAllowed(state,
				&entity_info,
				distance,
				active_group,
				health_decrease))
			{
				continue;
			}
			return BotAI_AcceptEnemy(state,
				&entity_info,
				distance,
				field_of_view,
				health_decrease,
				enemy);
		}

		if (AAS_PointLight(entity_info.origin, NULL, NULL, NULL) < 5)
		{
			continue;
		}
		if (!(distance > 300.0f) || BotAI_EntityIsShooting(&entity_info))
		{
			if (!BotAI_CoopTargetSwitchAllowed(state,
				&entity_info,
				distance,
				active_group,
				health_decrease))
			{
				continue;
			}
			return BotAI_AcceptEnemy(state,
				&entity_info,
				distance,
				field_of_view,
				health_decrease,
				enemy);
		}

		vec3_t candidate_to_bot;
		VectorSubtract(state->last_client_update.origin,
			entity_info.origin,
			candidate_to_bot);
		vec3_t bot_angles;
		Vector2Angles(candidate_to_bot, bot_angles);
		if (BotInterface_InFieldOfVision(entity_info.angles,
			160.0f,
			bot_angles))
		{
			if (!BotAI_CoopTargetSwitchAllowed(state,
				&entity_info,
				distance,
				active_group,
				health_decrease))
			{
				continue;
			}
			return BotAI_AcceptEnemy(state,
				&entity_info,
				distance,
				field_of_view,
				health_decrease,
				enemy);
		}

		BotAI_UpdateEnemyBattleInventory(state, entity_info.number);
		if (!BotAI_WantsToRetreat(state))
		{
			if (!BotAI_CoopTargetSwitchAllowed(state,
				&entity_info,
				distance,
				active_group,
				health_decrease))
			{
				continue;
			}
			return BotAI_AcceptEnemy(state,
				&entity_info,
				distance,
				field_of_view,
				health_decrease,
				enemy);
		}
		}
	}

	return qfalse;
}
