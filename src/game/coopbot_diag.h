#ifndef COOPBOT_DIAG_H
#define COOPBOT_DIAG_H

void CoopBotDiag_Init(void);
void CoopBotDiag_Shutdown(void);
void CoopBotDiag_FrameBegin(void);
void CoopBotDiag_FrameEnd(void);
void CoopBotDiag_BotAIStart(edict_t *bot);
void CoopBotDiag_BotAIEnd(edict_t *bot, int status);
void CoopBotDiag_RecordInput(edict_t *bot, const bot_input_t *input);
void CoopBotDiag_RecordEntity(edict_t *ent);
void CoopBotDiag_RecordMapLoad(const char *mapname,
	const char *library,
	int status);
void CoopBotDiag_RecordBotSpawn(edict_t *bot);
void CoopBotDiag_RecordBotRemove(edict_t *bot);
int CoopBotDiag_Command(char *cmd, edict_t *ent, int server);

#endif
