#include "g_local.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "bl_main.h"
#include "coopbot_diag.h"

#define COOPBOT_DIAG_MAX_CLIENTS MAX_CLIENTS
#define COOPBOT_DIAG_NAME_SIZE MAX_NETNAME

typedef struct coopbot_diag_client_s
{
	qboolean active;
	char name[COOPBOT_DIAG_NAME_SIZE];
	unsigned long long ai_calls;
	unsigned long long ai_errors;
	unsigned long long input_calls;
	unsigned long long attack_actions;
	unsigned long long use_actions;
	unsigned long long jump_actions;
	unsigned long long crouch_actions;
	int last_actionflags;
	clock_t ai_start;
} coopbot_diag_client_t;

typedef struct coopbot_diag_state_s
{
	cvar_t *log_level;
	cvar_t *metrics_enabled;
	cvar_t *metrics_interval;
	cvar_t *slow_ai_ms;
	unsigned long long frames;
	unsigned long long ai_calls;
	unsigned long long ai_errors;
	unsigned long long input_calls;
	unsigned long long entity_updates;
	unsigned long long monster_updates;
	unsigned long long attack_actions;
	unsigned long long use_actions;
	unsigned long long jump_actions;
	unsigned long long crouch_actions;
	unsigned long long map_loads;
	unsigned long long map_load_errors;
	unsigned long long reports;
	clock_t ai_ticks_total;
	clock_t ai_ticks_max;
	float next_report_time;
	char map_name[64];
	coopbot_diag_client_t clients[COOPBOT_DIAG_MAX_CLIENTS];
} coopbot_diag_state_t;

static coopbot_diag_state_t coopbot_diag;

static void CoopBotDiag_Dump(void);

static int CoopBotDiag_ClientNumber(const edict_t *bot)
{
	if (bot == NULL || g_edicts == NULL)
	{
		return -1;
	}

	return (int)(bot - g_edicts) - 1;
}

static int CoopBotDiag_LogLevel(void)
{
	if (coopbot_diag.log_level == NULL)
	{
		return 0;
	}

	return (int)coopbot_diag.log_level->value;
}

static unsigned long CoopBotDiag_ClockMilliseconds(clock_t ticks)
{
	if (ticks <= 0)
	{
		return 0;
	}

	return (unsigned long)((1000.0 * (double)ticks) / CLOCKS_PER_SEC);
}

static void CoopBotDiag_Log(int level, const char *fmt, ...)
{
	if (CoopBotDiag_LogLevel() < level || fmt == NULL)
	{
		return;
	}

	char message[1024];
	va_list args;
	va_start(args, fmt);
	vsnprintf(message, sizeof(message), fmt, args);
	va_end(args);
	message[sizeof(message) - 1] = '\0';
	gi.dprintf("[coopbot] %s\n", message);
}

static const char *CoopBotDiag_Name(const edict_t *bot)
{
	if (bot != NULL && bot->client != NULL &&
		bot->client->pers.netname[0] != '\0')
	{
		return bot->client->pers.netname;
	}

	return "<unnamed>";
}

static void CoopBotDiag_ResetCounters(void)
{
	clock_t zero = (clock_t)0;
	unsigned int index;

	coopbot_diag.frames = 0;
	coopbot_diag.ai_calls = 0;
	coopbot_diag.ai_errors = 0;
	coopbot_diag.input_calls = 0;
	coopbot_diag.entity_updates = 0;
	coopbot_diag.monster_updates = 0;
	coopbot_diag.attack_actions = 0;
	coopbot_diag.use_actions = 0;
	coopbot_diag.jump_actions = 0;
	coopbot_diag.crouch_actions = 0;
	coopbot_diag.map_loads = 0;
	coopbot_diag.map_load_errors = 0;
	coopbot_diag.reports = 0;
	coopbot_diag.ai_ticks_total = zero;
	coopbot_diag.ai_ticks_max = zero;
	coopbot_diag.next_report_time = 0.0f;

	for (index = 0; index < COOPBOT_DIAG_MAX_CLIENTS; ++index)
	{
		coopbot_diag.clients[index].ai_calls = 0;
		coopbot_diag.clients[index].ai_errors = 0;
		coopbot_diag.clients[index].input_calls = 0;
		coopbot_diag.clients[index].attack_actions = 0;
		coopbot_diag.clients[index].use_actions = 0;
		coopbot_diag.clients[index].jump_actions = 0;
		coopbot_diag.clients[index].crouch_actions = 0;
		coopbot_diag.clients[index].last_actionflags = 0;
		coopbot_diag.clients[index].ai_start = zero;
	}
}

void CoopBotDiag_Init(void)
{
	memset(&coopbot_diag, 0, sizeof(coopbot_diag));
	coopbot_diag.log_level = gi.cvar("coopbot_log", "1", 0);
	coopbot_diag.metrics_enabled = gi.cvar("coopbot_metrics", "0", 0);
	coopbot_diag.metrics_interval = gi.cvar("coopbot_metrics_interval", "10", 0);
	coopbot_diag.slow_ai_ms = gi.cvar("coopbot_slow_ai_ms", "100", 0);
	CoopBotDiag_ResetCounters();
	CoopBotDiag_Log(1,
		"diagnostics initialized log=%d metrics=%d interval=%.1f slow_ai_ms=%.1f",
		CoopBotDiag_LogLevel(),
		(int)coopbot_diag.metrics_enabled->value,
		coopbot_diag.metrics_interval->value,
		coopbot_diag.slow_ai_ms->value);
}

void CoopBotDiag_Shutdown(void)
{
	CoopBotDiag_Log(1, "diagnostics shutdown");
}

void CoopBotDiag_FrameBegin(void)
{
	coopbot_diag.frames += 1;
}

void CoopBotDiag_FrameEnd(void)
{
	float interval;

	if (coopbot_diag.metrics_enabled == NULL ||
		coopbot_diag.metrics_enabled->value == 0.0f)
	{
		return;
	}

	interval = coopbot_diag.metrics_interval->value;
	if (interval < 0.1f)
	{
		interval = 0.1f;
	}

	if (level.time >= coopbot_diag.next_report_time)
	{
		CoopBotDiag_Dump();
		coopbot_diag.next_report_time = level.time + interval;
	}
}

void CoopBotDiag_BotAIStart(edict_t *bot)
{
	int client = CoopBotDiag_ClientNumber(bot);

	if (client >= 0 && client < COOPBOT_DIAG_MAX_CLIENTS)
	{
		coopbot_diag.clients[client].ai_start = clock();
	}
}

void CoopBotDiag_BotAIEnd(edict_t *bot, int status)
{
	int client = CoopBotDiag_ClientNumber(bot);
	clock_t elapsed = (clock_t)0;

	if (client >= 0 && client < COOPBOT_DIAG_MAX_CLIENTS)
	{
		coopbot_diag.clients[client].ai_calls += 1;
		coopbot_diag.clients[client].ai_errors += status != BLERR_NOERROR;
		if (coopbot_diag.clients[client].ai_start != (clock_t)0)
		{
			elapsed = clock() - coopbot_diag.clients[client].ai_start;
		}
	}

	coopbot_diag.ai_calls += 1;
	coopbot_diag.ai_errors += status != BLERR_NOERROR;
	coopbot_diag.ai_ticks_total += elapsed;
	if (elapsed > coopbot_diag.ai_ticks_max)
	{
		coopbot_diag.ai_ticks_max = elapsed;
	}

	if (status != BLERR_NOERROR)
	{
		CoopBotDiag_Log(1, "ai_error client=%d name=\"%s\" status=%d",
			client, CoopBotDiag_Name(bot), status);
	}
	else if (CoopBotDiag_LogLevel() >= 3)
	{
		CoopBotDiag_Log(3, "ai client=%d name=\"%s\" elapsed_ms=%lu",
			client, CoopBotDiag_Name(bot),
			CoopBotDiag_ClockMilliseconds(elapsed));
	}

	if (coopbot_diag.slow_ai_ms != NULL &&
		CoopBotDiag_ClockMilliseconds(elapsed) >=
			(unsigned long)coopbot_diag.slow_ai_ms->value)
	{
		CoopBotDiag_Log(1, "slow_ai client=%d name=\"%s\" elapsed_ms=%lu",
			client, CoopBotDiag_Name(bot),
			CoopBotDiag_ClockMilliseconds(elapsed));
	}
}

void CoopBotDiag_RecordInput(edict_t *bot, const bot_input_t *input)
{
	int client = CoopBotDiag_ClientNumber(bot);
	int previous;

	if (input == NULL)
	{
		return;
	}

	coopbot_diag.input_calls += 1;
	coopbot_diag.attack_actions += (input->actionflags & ACTION_ATTACK) != 0;
	coopbot_diag.use_actions += (input->actionflags & ACTION_USE) != 0;
	coopbot_diag.jump_actions += (input->actionflags & ACTION_JUMP) != 0;
	coopbot_diag.crouch_actions += (input->actionflags & ACTION_CROUCH) != 0;

	if (client < 0 || client >= COOPBOT_DIAG_MAX_CLIENTS)
	{
		return;
	}

	coopbot_diag.clients[client].input_calls += 1;
	coopbot_diag.clients[client].attack_actions +=
		(input->actionflags & ACTION_ATTACK) != 0;
	coopbot_diag.clients[client].use_actions +=
		(input->actionflags & ACTION_USE) != 0;
	coopbot_diag.clients[client].jump_actions +=
		(input->actionflags & ACTION_JUMP) != 0;
	coopbot_diag.clients[client].crouch_actions +=
		(input->actionflags & ACTION_CROUCH) != 0;

	previous = coopbot_diag.clients[client].last_actionflags;
	coopbot_diag.clients[client].last_actionflags = input->actionflags;
	if (CoopBotDiag_LogLevel() >= 3 ||
		(CoopBotDiag_LogLevel() >= 2 &&
			previous != input->actionflags))
	{
		CoopBotDiag_Log(2,
			"input client=%d name=\"%s\" flags=0x%x speed=%.1f dir=(%.2f %.2f %.2f) view=(%.1f %.1f %.1f)",
			client, CoopBotDiag_Name(bot), input->actionflags, input->speed,
			input->dir[0], input->dir[1], input->dir[2],
			input->viewangles[0], input->viewangles[1], input->viewangles[2]);
	}
}

void CoopBotDiag_RecordEntity(edict_t *ent)
{
	coopbot_diag.entity_updates += 1;
	if (ent != NULL && (ent->svflags & SVF_MONSTER) != 0)
	{
		coopbot_diag.monster_updates += 1;
	}
}

void CoopBotDiag_RecordMapLoad(const char *mapname,
	const char *library,
	int status)
{
	coopbot_diag.map_loads += 1;
	if (status != BLERR_NOERROR)
	{
		coopbot_diag.map_load_errors += 1;
	}

	strncpy(coopbot_diag.map_name,
		mapname != NULL ? mapname : "<unknown>",
		sizeof(coopbot_diag.map_name) - 1);
	coopbot_diag.map_name[sizeof(coopbot_diag.map_name) - 1] = '\0';

	CoopBotDiag_Log(status == BLERR_NOERROR ? 1 : 0,
		"map_load map=\"%s\" library=\"%s\" status=%d",
		coopbot_diag.map_name,
		library != NULL ? library : "<unknown>",
		status);
}

void CoopBotDiag_RecordBotSpawn(edict_t *bot)
{
	int client = CoopBotDiag_ClientNumber(bot);

	if (client >= 0 && client < COOPBOT_DIAG_MAX_CLIENTS)
	{
		coopbot_diag.clients[client].active = true;
		strncpy(coopbot_diag.clients[client].name, CoopBotDiag_Name(bot),
			sizeof(coopbot_diag.clients[client].name) - 1);
		coopbot_diag.clients[client].name[
			sizeof(coopbot_diag.clients[client].name) - 1] = '\0';
	}

	CoopBotDiag_Log(1, "bot_spawn client=%d name=\"%s\"",
		client, CoopBotDiag_Name(bot));
}

void CoopBotDiag_RecordBotRemove(edict_t *bot)
{
	int client = CoopBotDiag_ClientNumber(bot);

	CoopBotDiag_Log(1, "bot_remove client=%d name=\"%s\"",
		client, CoopBotDiag_Name(bot));
	if (client >= 0 && client < COOPBOT_DIAG_MAX_CLIENTS)
	{
		coopbot_diag.clients[client].active = false;
	}
}

static void CoopBotDiag_Dump(void)
{
	unsigned long total_ms = CoopBotDiag_ClockMilliseconds(
		coopbot_diag.ai_ticks_total);
	unsigned long max_ms = CoopBotDiag_ClockMilliseconds(
		coopbot_diag.ai_ticks_max);
	double average_ms = coopbot_diag.ai_calls == 0
		? 0.0
		: (double)total_ms / (double)coopbot_diag.ai_calls;
	int active_bots = 0;
	int client;

	for (client = 0; client < game.maxclients; ++client)
	{
		edict_t *bot = DF_CLIENTENT(client);
		if (bot->inuse && (bot->flags & FL_BOT) != 0)
		{
			active_bots += 1;
		}
	}

	coopbot_diag.reports += 1;
	gi.dprintf(
		"[coopbot metrics] report=%llu map=\"%s\" time=%.1f frames=%llu bots=%d "
		"ai_calls=%llu ai_errors=%llu ai_avg_ms=%.3f ai_max_ms=%lu "
		"inputs=%llu attack=%llu use=%llu jump=%llu crouch=%llu "
		"entity_updates=%llu monster_updates=%llu map_loads=%llu map_errors=%llu\n",
		coopbot_diag.reports,
		coopbot_diag.map_name[0] != '\0' ? coopbot_diag.map_name : "<none>",
		level.time, coopbot_diag.frames, active_bots,
		coopbot_diag.ai_calls, coopbot_diag.ai_errors, average_ms, max_ms,
		coopbot_diag.input_calls, coopbot_diag.attack_actions,
		coopbot_diag.use_actions, coopbot_diag.jump_actions,
		coopbot_diag.crouch_actions, coopbot_diag.entity_updates,
		coopbot_diag.monster_updates, coopbot_diag.map_loads,
		coopbot_diag.map_load_errors);

	for (client = 0; client < game.maxclients; ++client)
	{
		coopbot_diag_client_t *stats = &coopbot_diag.clients[client];
		if (!stats->active)
		{
			continue;
		}

		gi.dprintf(
			"[coopbot metrics bot] client=%d name=\"%s\" ai=%llu errors=%llu "
			"inputs=%llu attack=%llu use=%llu jump=%llu crouch=%llu\n",
			client, stats->name, stats->ai_calls, stats->ai_errors,
			stats->input_calls, stats->attack_actions, stats->use_actions,
			stats->jump_actions, stats->crouch_actions);
	}
}

int CoopBotDiag_Command(char *cmd, edict_t *ent, int server)
{
	(void)ent;
	(void)server;

	if (cmd != NULL && Q_stricmp(cmd, "coopbot_metrics") == 0)
	{
		CoopBotDiag_Dump();
		return true;
	}

	return false;
}
