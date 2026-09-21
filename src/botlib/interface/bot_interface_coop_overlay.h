#ifndef BOTLIB_INTERFACE_BOT_INTERFACE_COOP_OVERLAY_H
#define BOTLIB_INTERFACE_BOT_INTERFACE_COOP_OVERLAY_H

#include <stdbool.h>

#include "bot_interface.h"
#include "bot_state.h"

#ifdef __cplusplus
extern "C" {
#endif

bool BotAI_CoopObjectiveControlEnabled(void);
void BotAI_CoopSetControlObjectivePhase(
	bot_client_state_t *state,
	bot_coop_control_phase_t phase,
	int control_entity,
	int goal_area);

bool BotAI_ApplyCoopHardLeash(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopControlWait(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopControlReturnPath(bot_client_state_t *state);
bool BotAI_ApplyCoopChangelevelGate(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopAreaAdvanceGate(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopDirectionalMove(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const vec3_t direction,
	bool preserve_actions,
	bool signal_intent);
bool BotAI_ApplyCoopCoverMove(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input,
	const vec3_t direction,
	bool preserve_actions);
bool BotAI_ApplyCoopRolePositioning(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopRescuePositioning(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopBasicCover(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopFirelineAvoidance(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopDoorwayAvoidance(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);
bool BotAI_ApplyCoopPersonalSpace(bot_client_state_t *state,
	float thinktime,
	bot_input_t *input);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BOTLIB_INTERFACE_BOT_INTERFACE_COOP_OVERLAY_H */
