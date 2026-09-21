#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_COMBAT_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_COMBAT_H

#include <stdbool.h>

#include "bot_interface.h"

typedef struct aas_entityinfo_s aas_entityinfo_t;

enum bot_battle_inventory_slot_e
{
	BOT_BATTLE_INVENTORY_ARMORBODY = 1,
	BOT_BATTLE_INVENTORY_ARMORCOMBAT = 2,
	BOT_BATTLE_INVENTORY_ARMORJACKET = 3,
	BOT_BATTLE_INVENTORY_POWERSCREEN = 5,
	BOT_BATTLE_INVENTORY_POWERSHIELD = 6,
	BOT_BATTLE_INVENTORY_SUPERSHOTGUN = 9,
	BOT_BATTLE_INVENTORY_MACHINEGUN = 10,
	BOT_BATTLE_INVENTORY_CHAINGUN = 11,
	BOT_BATTLE_INVENTORY_GRENADES = 12,
	BOT_BATTLE_INVENTORY_GRENADELAUNCHER = 13,
	BOT_BATTLE_INVENTORY_ROCKETLAUNCHER = 14,
	BOT_BATTLE_INVENTORY_HYPERBLASTER = 15,
	BOT_BATTLE_INVENTORY_RAILGUN = 16,
	BOT_BATTLE_INVENTORY_BFG10K = 17,
	BOT_BATTLE_INVENTORY_SHELLS = 18,
	BOT_BATTLE_INVENTORY_BULLETS = 19,
	BOT_BATTLE_INVENTORY_CELLS = 20,
	BOT_BATTLE_INVENTORY_ROCKETS = 21,
	BOT_BATTLE_INVENTORY_SLUGS = 22,
	BOT_BATTLE_INVENTORY_QUAD = 23,
	BOT_BATTLE_INVENTORY_INVULNERABILITY = 24,
	BOT_BATTLE_INVENTORY_SILENCER = 25,
	BOT_BATTLE_INVENTORY_REBREATHER = 26,
	BOT_BATTLE_INVENTORY_HEALTH = 41,
	BOT_BATTLE_INVENTORY_FLAG1 = 43,
	BOT_BATTLE_INVENTORY_FLAG2 = 44,
	BOT_BATTLE_INVENTORY_TECH1 = 45,
	BOT_BATTLE_INVENTORY_TECH2 = 46,
	BOT_BATTLE_INVENTORY_TECH3 = 47,
	BOT_BATTLE_INVENTORY_TECH4 = 48,
	BOT_BATTLE_ENEMY_HORIZONTAL_DIST = 200,
	BOT_BATTLE_ENEMY_HEIGHT = 201,
	BOT_BATTLE_USING_QUAD = 204,
	BOT_BATTLE_USING_INVULNERABILITY = 205,
	BOT_BATTLE_USING_REBREATHER = 207,
	BOT_BATTLE_USING_ENVIRONMENTSUIT = 208,
	BOT_BATTLE_USING_POWERSCREEN = 210,
	BOT_BATTLE_USING_POWERSHIELD = 211,
	BOT_BATTLE_ENEMY_BLASTER = 230,
	BOT_BATTLE_ENEMY_SHOTGUN = 231,
	BOT_BATTLE_ENEMY_SUPERSHOTGUN = 232,
	BOT_BATTLE_ENEMY_MACHINEGUN = 233,
	BOT_BATTLE_ENEMY_CHAINGUN = 234,
	BOT_BATTLE_ENEMY_GRENADELAUNCHER = 235,
	BOT_BATTLE_ENEMY_ROCKETLAUNCHER = 236,
	BOT_BATTLE_ENEMY_HYPERBLASTER = 237,
	BOT_BATTLE_ENEMY_RAILGUN = 238,
	BOT_BATTLE_ENEMY_BFG10K = 239,
	BOT_BATTLE_ENEMY_GRENADES = 240,
	BOT_BATTLE_ENEMY_GRAPPLE = 241,
	BOT_BATTLE_ENEMY_QUAD = 245,
	BOT_BATTLE_ENEMY_INVULNERABILITY = 246,
	BOT_BATTLE_ENEMY_POWERSCREEN = 247,
};

#ifdef __cplusplus
extern "C" {
#endif

void BotAI_InitEnemyInfo(ai_dm_enemy_info_t *info);
void BotInterface_SynchroniseCombatState(bot_client_state_t *state);
void BotAI_UpdateBattleInventory(bot_client_state_t *state);
bool BotInterface_InFieldOfVision(const vec3_t viewangles,
	float field_of_view,
	const vec3_t target_angles);
void BotInterface_ClientEyePosition(const bot_client_state_t *state,
	vec3_t out);
bool BotInterface_HasLineOfSight(const vec3_t from,
	const vec3_t to,
	int viewer,
	int target);
int BotAI_EntityIsDead(const aas_entityinfo_t *entity_info);
int BotAI_EntityIsShooting(const aas_entityinfo_t *entity_info);
int BotAI_LibVarOrderedNonZero(const char *name);
int BotAI_CoopMode(void);
float BotAI_CoopDangerScore(const bot_client_state_t *state);
bool BotAI_CoopDangerRequiresRetreat(const bot_client_state_t *state);
int BotAI_CoopPlayerEntity(const bot_client_state_t *state, vec3_t origin);
bool BotAI_CoopIntentAllowsForwardProgress(const bot_client_state_t *state);
bool BotAI_CoopRoleBlocksNewGroup(const bot_client_state_t *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_COMBAT_H */
