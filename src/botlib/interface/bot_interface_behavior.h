#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_BEHAVIOR_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_BEHAVIOR_H

#include "bot_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

float BotAI_ConsoleRandom(void);
float BotAI_LongTermGoalRandom(void);
void BotAI_TryRecentEnemyDeathWave(bot_client_state_t *state,
	float thinktime);
void BotAI_RoamGoal(const bot_client_state_t *state, vec3_t goal);
int BotAI_CTFTeam(const bot_client_state_t *state);
unsigned long BotAI_ConsoleSynonymContext(const bot_client_state_t *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_BEHAVIOR_H */

