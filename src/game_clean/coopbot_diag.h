#ifndef COOPBOT_DIAG_H
#define COOPBOT_DIAG_H

void CoopBotDiag_Init(void);
void CoopBotDiag_Shutdown(void);
void CoopBotDiag_FrameBegin(void);
void CoopBotDiag_FrameEnd(void);
void CoopBotDiag_BotAIStart(edict_t *bot);
void CoopBotDiag_BotAIEnd(edict_t *bot, int status);
void CoopBotDiag_RecordInput(edict_t *bot, const bot_input_t *input);
void CoopBotDiag_RecordBotSnapshot(edict_t *bot, const bot_input_t *input);
void CoopBotDiag_RecordPlayerContact(edict_t *first,
	edict_t *second,
	const trace_t *trace);
void CoopBotDiag_RecordEntity(edict_t *ent);
void CoopBotDiag_RecordMapEntities(const char *mapname);
void CoopBotDiag_RecordMapLoad(const char *mapname,
	const char *library,
	int status);
void CoopBotDiag_RecordBotSpawn(edict_t *bot);
void CoopBotDiag_RecordBotRemove(edict_t *bot);
void CoopBotDiag_RecordBotState(const char *phase, edict_t *bot);
void CoopBotDiag_RecordShot(const char *kind,
	edict_t *attacker,
	const trace_t *trace,
	int damage,
	int mod);
void CoopBotDiag_RecordProjectileLaunch(edict_t *projectile,
	edict_t *owner,
	int damage,
	int mod);
void CoopBotDiag_RecordProjectileTouch(edict_t *projectile,
	edict_t *other,
	int damage,
	int mod);
void CoopBotDiag_RecordDamageAttempt(edict_t *targ,
	edict_t *inflictor,
	edict_t *attacker,
	int damage,
	int dflags,
	int mod);
void CoopBotDiag_RecordDamageApplied(edict_t *targ,
	edict_t *attacker,
	int requested,
	int applied,
	int health_before,
	int mod);
int CoopBotDiag_Command(char *cmd, edict_t *ent, int server);

#endif
