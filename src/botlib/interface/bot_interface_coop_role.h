#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_COOP_ROLE_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_COOP_ROLE_H

#include <stdbool.h>

#include "bot_interface.h"
#include "bot_state.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *BotAI_CoopRoleName(bot_coop_role_t role);
bool BotAI_CoopRoleBreaksChase(const bot_client_state_t *state);
void BotAI_UpdateCoopJointRetreat(bot_client_state_t *state);
void BotAI_CoopClearActionCommitment(bot_client_state_t *state);
void BotAI_CoopStartAction(bot_client_state_t *state,
	bot_coop_action_t action,
	const vec3_t direction);
void BotAI_UpdateCoopRole(bot_client_state_t *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_COOP_ROLE_H */
