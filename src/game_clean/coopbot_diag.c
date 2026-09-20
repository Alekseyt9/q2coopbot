#include "g_local.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
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
	int last_target_entity;
	int last_visible_enemies;
	clock_t ai_start;
} coopbot_diag_client_t;

typedef struct coopbot_diag_state_s
{
	cvar_t *log_level;
	cvar_t *metrics_enabled;
	cvar_t *metrics_interval;
	cvar_t *slow_ai_ms;
	cvar_t *map_dump;
	cvar_t *jsonl_enabled;
	cvar_t *episode_id_override;
	cvar_t *seed;
	FILE *log_file;
	FILE *event_file;
	char log_path[256];
	char event_path[256];
	char episode_id[96];
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
	unsigned long long episode_sequence;
	clock_t ai_ticks_total;
	clock_t ai_ticks_max;
	float next_report_time;
	char map_name[64];
	coopbot_diag_client_t clients[COOPBOT_DIAG_MAX_CLIENTS];
} coopbot_diag_state_t;

static coopbot_diag_state_t coopbot_diag;

static void CoopBotDiag_Dump(void);
static void CoopBotDiag_DumpMapEntities(void);

//===========================================================================
//
// Derive a stable event type from the first token of the existing log line.
//
//===========================================================================
static const char *CoopBotDiag_EventName(const char *message)
{
	static char event_name[48];
	const char *cursor = message;
	size_t length = 0;

	if (message == NULL)
	{
		return "diagnostic";
	}

	while (cursor[length] != '\0' && cursor[length] != ' ' &&
		cursor[length] != '\t' && length < sizeof(event_name) - 1)
	{
		length += 1;
	}
	memcpy(event_name, message, length);
	event_name[length] = '\0';
	return event_name[0] != '\0' ? event_name : "diagnostic";
}

//===========================================================================
//
// Escape a value for the JSONL event stream without changing the text log.
//
//===========================================================================
static void CoopBotDiag_JSONEscape(FILE *file, const char *value)
{
	const unsigned char *cursor = (const unsigned char *)value;

	if (file == NULL)
	{
		return;
	}

	if (cursor == NULL)
	{
		fputs("<none>", file);
		return;
	}

	while (*cursor != '\0')
	{
		switch (*cursor)
		{
		case '\\':
			fputs("\\\\", file);
			break;
		case '"':
			fputs("\\\"", file);
			break;
		case '\n':
			fputs("\\n", file);
			break;
		case '\r':
			fputs("\\r", file);
			break;
		case '\t':
			fputs("\\t", file);
			break;
		default:
			if (*cursor < 0x20)
			{
				fputc(' ', file);
			}
			else
			{
				fputc(*cursor, file);
			}
			break;
		}
		cursor += 1;
	}
}

//===========================================================================
//
// Write one machine-readable event while retaining the legacy text log.
// The message keeps the existing key=value representation until structured
// event-specific fields are added by the P0 telemetry work.
//
//===========================================================================
static void CoopBotDiag_WriteJSONEvent(const char *event,
	int severity,
	const char *message)
{
	if (coopbot_diag.event_file == NULL ||
		coopbot_diag.jsonl_enabled == NULL ||
		coopbot_diag.jsonl_enabled->value == 0.0f)
	{
		return;
	}

	fputs("{\"time\":", coopbot_diag.event_file);
	fprintf(coopbot_diag.event_file, "%.3f", level.time);
	fputs(",\"episode_id\":\"", coopbot_diag.event_file);
	CoopBotDiag_JSONEscape(coopbot_diag.event_file, coopbot_diag.episode_id);
	fputs("\",\"map\":\"", coopbot_diag.event_file);
	CoopBotDiag_JSONEscape(coopbot_diag.event_file, coopbot_diag.map_name);
	fputs("\",\"event\":\"", coopbot_diag.event_file);
	CoopBotDiag_JSONEscape(coopbot_diag.event_file, event);
	fputs("\",\"severity\":", coopbot_diag.event_file);
	fprintf(coopbot_diag.event_file, "%d", severity);
	fputs(",\"message\":\"", coopbot_diag.event_file);
	CoopBotDiag_JSONEscape(coopbot_diag.event_file, message);
	fputs("\"}\n", coopbot_diag.event_file);
	fflush(coopbot_diag.event_file);
}

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

static void CoopBotDiag_WriteLine(const char *line)
{
	if (coopbot_diag.log_file == NULL || line == NULL)
	{
		return;
	}

	fprintf(coopbot_diag.log_file, "[coopbot] time=%.3f %s\n",
		level.time, line);
	fflush(coopbot_diag.log_file);
}

static void CoopBotDiag_Log(int level, const char *fmt, ...)
{
	char message[1024];
	va_list args;

	if (fmt == NULL)
	{
		return;
	}

	va_start(args, fmt);
	vsnprintf(message, sizeof(message), fmt, args);
	va_end(args);
	message[sizeof(message) - 1] = '\0';
	if (CoopBotDiag_LogLevel() >= level)
	{
		CoopBotDiag_WriteLine(message);
	}
	CoopBotDiag_WriteJSONEvent(CoopBotDiag_EventName(message), level, message);
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

static int CoopBotDiag_EntityNumber(const edict_t *ent)
{
	if (ent == NULL || g_edicts == NULL)
	{
		return -1;
	}

	return (int)(ent - g_edicts);
}

static const char *CoopBotDiag_Classname(const edict_t *ent)
{
	if (ent != NULL && ent->classname != NULL && ent->classname[0] != '\0')
	{
		return ent->classname;
	}

	return "<none>";
}

//===========================================================================
//
// Select the human player used as the companion's primary reference. Co-op
// can have more than one human, so prefer the first active non-bot client and
// fall back to another active client when a diagnostic run is bot-only.
//
//===========================================================================
static edict_t *CoopBotDiag_Player(edict_t *bot)
{
	edict_t *fallback = NULL;
	int client;

	for (client = 0; client < game.maxclients; ++client)
	{
		edict_t *candidate = DF_CLIENTENT(client);

		if (candidate == bot || !candidate->inuse)
		{
			continue;
		}
		if (fallback == NULL)
		{
			fallback = candidate;
		}
		if ((candidate->flags & FL_BOT) == 0)
		{
			return candidate;
		}
	}

	return fallback;
}

//===========================================================================
//
// Return the three-dimensional distance between two active entities.
//
//===========================================================================
static float CoopBotDiag_Distance(const edict_t *first, const edict_t *second)
{
	vec3_t delta;

	if (first == NULL || second == NULL)
	{
		return -1.0f;
	}

	VectorSubtract(first->s.origin, second->s.origin, delta);
	return sqrtf(DotProduct(delta, delta));
}

//===========================================================================
//
// Count active monsters visible from the bot's current position. This is a
// game-side observation for telemetry; botlib remains the owner of AI state.
//
//===========================================================================
static int CoopBotDiag_VisibleMonsters(edict_t *bot)
{
	int visible_count = 0;
	int entity_number;

	if (bot == NULL)
	{
		return 0;
	}

	for (entity_number = game.maxclients;
		entity_number < globals.num_edicts; ++entity_number)
	{
		edict_t *candidate = &g_edicts[entity_number];

		if (!candidate->inuse || (candidate->svflags & SVF_MONSTER) == 0 ||
			candidate->deadflag != DEAD_NO)
		{
			continue;
		}
		if (visible(bot, candidate))
		{
			visible_count += 1;
		}
	}

	return visible_count;
}

//===========================================================================
//
// Record the observable companion state at the input boundary. The state is
// intentionally an inferred diagnostic label, not a new gameplay FSM.
//
//===========================================================================
void CoopBotDiag_RecordBotSnapshot(edict_t *bot, const bot_input_t *input)
{
	edict_t *player;
	edict_t *target;
	float distance_to_player;
	int target_visible = 0;
	int visible_enemies;
	int target_entity;
	const char *state = "idle";
	int client;

	if (bot == NULL || input == NULL)
	{
		return;
	}

	player = CoopBotDiag_Player(bot);
	target = bot->enemy;
	target_entity = (target != NULL && target->inuse &&
		target->takedamage && target->health > 0)
		? CoopBotDiag_EntityNumber(target) : -1;
	distance_to_player = CoopBotDiag_Distance(bot, player);
	if (bot->deadflag != DEAD_NO)
	{
		state = "dead";
	}
	else if (target != NULL && target->inuse && target->takedamage &&
		target->health > 0)
	{
		state = "engaged";
		target_visible = visible(bot, target);
	}
	else if (distance_to_player >= 0.0f && distance_to_player > 768.0f)
	{
		state = "far_from_player";
	}
	else if (distance_to_player >= 0.0f && distance_to_player > 384.0f)
	{
		state = "following";
	}

	client = CoopBotDiag_ClientNumber(bot);
	visible_enemies = CoopBotDiag_VisibleMonsters(bot);
	if (client >= 0 && client < COOPBOT_DIAG_MAX_CLIENTS)
	{
		if (coopbot_diag.clients[client].last_target_entity != target_entity)
		{
			if (target_entity >= 0)
			{
				CoopBotDiag_Log(2,
					"target_acquired client=%d target=%d class=\"%s\"",
					client, target_entity, CoopBotDiag_Classname(target));
			}
			else if (coopbot_diag.clients[client].last_target_entity >= 0)
			{
				CoopBotDiag_Log(2,
					"target_lost client=%d target=%d",
					client, coopbot_diag.clients[client].last_target_entity);
			}
			coopbot_diag.clients[client].last_target_entity = target_entity;
		}
		if (coopbot_diag.clients[client].last_visible_enemies == 0 &&
			visible_enemies > 0)
		{
			CoopBotDiag_Log(2,
				"encounter_observed client=%d visible_enemies=%d",
				client, visible_enemies);
		}
		coopbot_diag.clients[client].last_visible_enemies = visible_enemies;
	}
	CoopBotDiag_Log(2,
		"bot_snapshot client=%d name=\"%s\" state=%s player=%d "
		"player_origin=(%.1f %.1f %.1f) bot_origin=(%.1f %.1f %.1f) "
		"distance_to_player=%.1f target=%d target_class=\"%s\" "
		"target_health=%d target_visible=%d visible_enemies=%d "
		"input_flags=0x%x speed=%.1f",
		client, CoopBotDiag_Name(bot), state,
		CoopBotDiag_EntityNumber(player),
		player != NULL ? player->s.origin[0] : 0.0f,
		player != NULL ? player->s.origin[1] : 0.0f,
		player != NULL ? player->s.origin[2] : 0.0f,
		bot->s.origin[0], bot->s.origin[1], bot->s.origin[2],
		distance_to_player,
		CoopBotDiag_EntityNumber(target), CoopBotDiag_Classname(target),
		target != NULL ? target->health : 0, target_visible,
		visible_enemies, input->actionflags, input->speed);
}

//===========================================================================
//
// Record an actual physics contact between a bot and a human player. Contact
// is deliberately kept separate from "blocked": a later reducer can require
// repeated contact plus movement intent before calling it a path violation.
//
//===========================================================================
void CoopBotDiag_RecordPlayerContact(edict_t *first,
	edict_t *second,
	const trace_t *trace)
{
	edict_t *bot;
	edict_t *player;
	edict_t *moving;
	vec3_t velocity;
	const char *moving_role;
	float moving_speed;

	if (first == NULL || second == NULL || trace == NULL)
	{
		return;
	}

	if ((first->flags & FL_BOT) != 0 && second->client != NULL)
	{
		bot = first;
		player = second;
		moving = first;
		moving_role = "bot";
	}
	else if ((second->flags & FL_BOT) != 0 && first->client != NULL)
	{
		bot = second;
		player = first;
		moving = first;
		moving_role = "player";
	}
	else
	{
		return;
	}

	VectorCopy(bot->velocity, velocity);
	moving_speed = sqrtf(DotProduct(moving->velocity, moving->velocity));
	CoopBotDiag_Log(2,
		"player_contact bot=%d player=%d fraction=%.5f normal=(%.2f %.2f %.2f) "
		"moving_role=%s moving_speed=%.1f bot_speed=%.1f "
		"bot_origin=(%.1f %.1f %.1f) "
		"player_origin=(%.1f %.1f %.1f)",
		CoopBotDiag_EntityNumber(bot), CoopBotDiag_EntityNumber(player),
		trace->fraction, trace->plane.normal[0], trace->plane.normal[1],
		trace->plane.normal[2], moving_role, moving_speed,
		sqrtf(DotProduct(velocity, velocity)),
		bot->s.origin[0], bot->s.origin[1], bot->s.origin[2],
		player->s.origin[0], player->s.origin[1], player->s.origin[2]);
	if (moving == player && moving_speed > 1.0f && trace->fraction < 1.0f)
	{
		CoopBotDiag_Log(2,
			"player_block_candidate bot=%d player=%d moving_speed=%.1f "
			"fraction=%.5f normal=(%.2f %.2f %.2f)",
			CoopBotDiag_EntityNumber(bot), CoopBotDiag_EntityNumber(player),
			moving_speed, trace->fraction, trace->plane.normal[0],
			trace->plane.normal[1], trace->plane.normal[2]);
	}
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
		coopbot_diag.clients[index].last_target_entity = -1;
		coopbot_diag.clients[index].last_visible_enemies = 0;
		coopbot_diag.clients[index].ai_start = zero;
	}
}

void CoopBotDiag_Init(void)
{
	cvar_t *game_dir;

	memset(&coopbot_diag, 0, sizeof(coopbot_diag));
	coopbot_diag.log_level = gi.cvar("coopbot_log", "1", 0);
	coopbot_diag.metrics_enabled = gi.cvar("coopbot_metrics", "1", 0);
	coopbot_diag.metrics_interval = gi.cvar("coopbot_metrics_interval", "10", 0);
	coopbot_diag.slow_ai_ms = gi.cvar("coopbot_slow_ai_ms", "100", 0);
	coopbot_diag.map_dump = gi.cvar("coopbot_map_dump", "1", 0);
	coopbot_diag.jsonl_enabled = gi.cvar("coopbot_jsonl", "1", 0);
	coopbot_diag.episode_id_override = gi.cvar("coopbot_episode_id", "", 0);
	coopbot_diag.seed = gi.cvar("coopbot_seed", "0", 0);
	if (coopbot_diag.seed != NULL && coopbot_diag.seed->string != NULL &&
		coopbot_diag.seed->value > 0.0f)
	{
		srand((unsigned int)strtoul(coopbot_diag.seed->string, NULL, 10));
	}
	game_dir = gi.cvar("game", "", 0);
	if (game_dir != NULL && game_dir->string != NULL && game_dir->string[0] != '\0')
	{
		snprintf(coopbot_diag.log_path, sizeof(coopbot_diag.log_path),
			"%s/coopbot_debug.log", game_dir->string);
	}
	else
	{
		snprintf(coopbot_diag.log_path, sizeof(coopbot_diag.log_path),
			"coopbot_debug.log");
	}
	snprintf(coopbot_diag.event_path, sizeof(coopbot_diag.event_path),
		"%s", coopbot_diag.log_path);
	{
		char *extension = strrchr(coopbot_diag.event_path, '.');
		if (extension != NULL)
		{
			snprintf(extension,
				(size_t)(coopbot_diag.event_path + sizeof(coopbot_diag.event_path) - extension),
				"_events.jsonl");
		}
		else
		{
			snprintf(coopbot_diag.event_path, sizeof(coopbot_diag.event_path),
				"coopbot_events.jsonl");
		}
	}
	coopbot_diag.log_file = fopen(coopbot_diag.log_path, "a");
	coopbot_diag.event_file = fopen(coopbot_diag.event_path, "a");
	if (coopbot_diag.episode_id_override != NULL &&
		coopbot_diag.episode_id_override->string != NULL &&
		coopbot_diag.episode_id_override->string[0] != '\0')
	{
		strncpy(coopbot_diag.episode_id,
			coopbot_diag.episode_id_override->string,
			sizeof(coopbot_diag.episode_id) - 1);
		coopbot_diag.episode_id[sizeof(coopbot_diag.episode_id) - 1] = '\0';
	}
	else
	{
		strncpy(coopbot_diag.episode_id, "pending",
			sizeof(coopbot_diag.episode_id) - 1);
	}
	CoopBotDiag_ResetCounters();
	CoopBotDiag_Log(1,
		"diagnostics initialized file=\"%s\" events=\"%s\" log=%d metrics=%d interval=%.1f slow_ai_ms=%.1f map_dump=%d jsonl=%d seed=\"%s\"",
		coopbot_diag.log_path,
		coopbot_diag.event_path,
		CoopBotDiag_LogLevel(),
		(int)coopbot_diag.metrics_enabled->value,
		coopbot_diag.metrics_interval->value,
		coopbot_diag.slow_ai_ms->value,
		(int)coopbot_diag.map_dump->value,
		(int)coopbot_diag.jsonl_enabled->value,
		coopbot_diag.seed != NULL ? coopbot_diag.seed->string : "0");
}

void CoopBotDiag_Shutdown(void)
{
	CoopBotDiag_Log(1, "diagnostics shutdown");
	if (coopbot_diag.log_file != NULL)
	{
		fclose(coopbot_diag.log_file);
		coopbot_diag.log_file = NULL;
	}
	if (coopbot_diag.event_file != NULL)
	{
		fclose(coopbot_diag.event_file);
		coopbot_diag.event_file = NULL;
	}
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

//===========================================================================
//
// Classify an active Quake II entity for the first level-model inventory.
//
//===========================================================================
static const char *CoopBotDiag_MapEntityKind(const edict_t *ent)
{
	const char *classname;

	if (ent == NULL || ent->classname == NULL)
	{
		return "unknown";
	}

	classname = ent->classname;
	if (strncmp(classname, "monster_", 8) == 0)
	{
		return "monster";
	}
	if (strncmp(classname, "trigger_", 8) == 0)
	{
		return "trigger";
	}
	if (strncmp(classname, "target_", 7) == 0)
	{
		return "target";
	}
	if (strncmp(classname, "func_", 5) == 0)
	{
		return "mover";
	}
	if (strncmp(classname, "item_", 5) == 0 ||
		strncmp(classname, "weapon_", 7) == 0 ||
		strncmp(classname, "ammo_", 5) == 0 ||
		strncmp(classname, "key_", 4) == 0)
	{
		return "pickup";
	}
	if (strncmp(classname, "info_player_", 12) == 0)
	{
		return "spawn";
	}
	if (strcmp(classname, "worldspawn") == 0)
	{
		return "world";
	}
	return "other";
}

//===========================================================================
//
// Return a stable entity field for key/value diagnostics.
//
//===========================================================================
static const char *CoopBotDiag_MapField(const char *value)
{
	return value != NULL && value[0] != '\0' ? value : "<none>";
}

//===========================================================================
//
// Emit the runtime map entity graph that is available to the game module.
// This deliberately records post-spawn entities, so skill/co-op filtering and
// the actual trigger/mover setup are reflected instead of just raw BSP text.
//
//===========================================================================
static void CoopBotDiag_DumpMapEntities(void)
{
	int entity_count = 0;
	int monster_count = 0;
	int trigger_count = 0;
	int mover_count = 0;
	int target_count = 0;
	int link_count = 0;
	int unresolved_link_count = 0;
	int entity_number;

	for (entity_number = 0; entity_number < globals.num_edicts; ++entity_number)
	{
		edict_t *ent = &g_edicts[entity_number];
		const char *kind;

		if (!ent->inuse)
		{
			continue;
		}

		kind = CoopBotDiag_MapEntityKind(ent);
		entity_count += 1;
		monster_count += strcmp(kind, "monster") == 0;
		trigger_count += strcmp(kind, "trigger") == 0;
		mover_count += strcmp(kind, "mover") == 0;
		target_count += strcmp(kind, "target") == 0;
	}

	CoopBotDiag_Log(1,
		"map_inventory map=\"%s\" entities=%d monsters=%d triggers=%d movers=%d targets=%d",
		coopbot_diag.map_name[0] != '\0' ? coopbot_diag.map_name : "<none>",
		entity_count, monster_count, trigger_count, mover_count, target_count);

	for (entity_number = 0; entity_number < globals.num_edicts; ++entity_number)
	{
		edict_t *ent = &g_edicts[entity_number];
		const char *kind;
		edict_t *target;
		int matched_target = 0;

		if (!ent->inuse)
		{
			continue;
		}

		kind = CoopBotDiag_MapEntityKind(ent);
		CoopBotDiag_Log(1,
			"map_entity ent=%d kind=%s class=\"%s\" model=\"%s\" "
			"origin=(%.1f %.1f %.1f) mins=(%.1f %.1f %.1f) maxs=(%.1f %.1f %.1f) "
			"solid=%d movetype=%d spawnflags=0x%x health=%d target=\"%s\" "
			"targetname=\"%s\" killtarget=\"%s\" map=\"%s\" "
			"move_state=%d target_ent=%d has_use=%d has_touch=%d",
			entity_number, kind, CoopBotDiag_MapField(ent->classname),
			CoopBotDiag_MapField(ent->model), ent->s.origin[0], ent->s.origin[1],
			ent->s.origin[2], ent->mins[0], ent->mins[1], ent->mins[2],
			ent->maxs[0], ent->maxs[1], ent->maxs[2], ent->solid, ent->movetype,
			ent->spawnflags, ent->health, CoopBotDiag_MapField(ent->target),
			CoopBotDiag_MapField(ent->targetname),
			CoopBotDiag_MapField(ent->killtarget), CoopBotDiag_MapField(ent->map),
			ent->moveinfo.state, CoopBotDiag_EntityNumber(ent->target_ent),
			ent->use != NULL, ent->touch != NULL);

		if (ent->target == NULL || ent->target[0] == '\0')
		{
			continue;
		}

		for (target = g_edicts; target < &g_edicts[globals.num_edicts]; ++target)
		{
			if (!target->inuse || target->targetname == NULL ||
				target->targetname[0] == '\0' ||
				strcmp(ent->target, target->targetname) != 0)
			{
				continue;
			}

			link_count += 1;
			matched_target = 1;
			CoopBotDiag_Log(1,
				"map_link from=%d target=\"%s\" to=%d to_class=\"%s\"",
				entity_number, ent->target, (int)(target - g_edicts),
				CoopBotDiag_MapField(target->classname));
		}

		if (!matched_target)
		{
			unresolved_link_count += 1;
			CoopBotDiag_Log(1,
				"map_link from=%d target=\"%s\" to=-1 to_class=\"<unresolved>\"",
				entity_number, ent->target);
		}
	}

	CoopBotDiag_Log(1,
		"map_inventory_links map=\"%s\" links=%d unresolved=%d",
		coopbot_diag.map_name[0] != '\0' ? coopbot_diag.map_name : "<none>",
		link_count, unresolved_link_count);
}

//===========================================================================
//
// Record the post-spawn level model when map diagnostics are enabled.
//
//===========================================================================
void CoopBotDiag_RecordMapEntities(const char *mapname)
{
	if (mapname != NULL && mapname[0] != '\0')
	{
		strncpy(coopbot_diag.map_name, mapname,
			sizeof(coopbot_diag.map_name) - 1);
		coopbot_diag.map_name[sizeof(coopbot_diag.map_name) - 1] = '\0';
	}

	if (coopbot_diag.episode_id_override == NULL ||
		coopbot_diag.episode_id_override->string == NULL ||
		coopbot_diag.episode_id_override->string[0] == '\0')
	{
		coopbot_diag.episode_sequence += 1;
		snprintf(coopbot_diag.episode_id, sizeof(coopbot_diag.episode_id),
			"%s-%llu",
			coopbot_diag.map_name[0] != '\0' ? coopbot_diag.map_name : "unknown",
			coopbot_diag.episode_sequence);
	}
	CoopBotDiag_Log(1,
		"episode_start episode_id=\"%s\" map=\"%s\" seed=\"%s\"",
		coopbot_diag.episode_id,
		coopbot_diag.map_name[0] != '\0' ? coopbot_diag.map_name : "<none>",
		coopbot_diag.seed != NULL ? coopbot_diag.seed->string : "0");

	if (coopbot_diag.map_dump == NULL || coopbot_diag.map_dump->value == 0.0f)
	{
		return;
	}

	CoopBotDiag_DumpMapEntities();
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

void CoopBotDiag_RecordBotState(const char *phase, edict_t *bot)
{
	client_persistant_t *pers;
	const gitem_t *weapon;
	const gitem_t *bullets;
	const gitem_t *shells;
	const gitem_t *rockets;
	const gitem_t *grenades;
	const gitem_t *cells;
	const gitem_t *slugs;
	const gitem_t *jacket;
	const gitem_t *combat;
	const gitem_t *body;
	const gitem_t *power_shield;
	int client;

	if (bot == NULL || bot->client == NULL ||
		!(bot->flags & FL_BOT))
	{
		return;
	}

	pers = &bot->client->pers;
	weapon = pers->weapon;
	bullets = FindItem("Bullets");
	shells = FindItem("Shells");
	rockets = FindItem("Rockets");
	grenades = FindItem("Grenades");
	cells = FindItem("Cells");
	slugs = FindItem("Slugs");
	jacket = FindItem("Jacket Armor");
	combat = FindItem("Combat Armor");
	body = FindItem("Body Armor");
	power_shield = FindItem("Power Shield");
	client = CoopBotDiag_ClientNumber(bot);

	CoopBotDiag_Log(1,
		"bot_state phase=\"%s\" client=%d hp=%d max_hp=%d weapon=\"%s\" "
		"selected=%d bullets=%d shells=%d rockets=%d grenades=%d cells=%d slugs=%d "
		"jacket=%d combat=%d body=%d power_shield=%d",
		phase != NULL ? phase : "unknown",
		client,
		pers->health,
		pers->max_health,
		weapon != NULL && weapon->pickup_name != NULL
			? weapon->pickup_name : "<none>",
		pers->selected_item,
		bullets != NULL ? pers->inventory[ITEM_INDEX(bullets)] : 0,
		shells != NULL ? pers->inventory[ITEM_INDEX(shells)] : 0,
		rockets != NULL ? pers->inventory[ITEM_INDEX(rockets)] : 0,
		grenades != NULL ? pers->inventory[ITEM_INDEX(grenades)] : 0,
		cells != NULL ? pers->inventory[ITEM_INDEX(cells)] : 0,
		slugs != NULL ? pers->inventory[ITEM_INDEX(slugs)] : 0,
		jacket != NULL ? pers->inventory[ITEM_INDEX(jacket)] : 0,
		combat != NULL ? pers->inventory[ITEM_INDEX(combat)] : 0,
		body != NULL ? pers->inventory[ITEM_INDEX(body)] : 0,
		power_shield != NULL ? pers->inventory[ITEM_INDEX(power_shield)] : 0);
}

void CoopBotDiag_RecordShot(const char *kind,
	edict_t *attacker,
	const trace_t *trace,
	int damage,
	int mod)
{
	edict_t *target;

	if (CoopBotDiag_LogLevel() < 2 || trace == NULL)
	{
		return;
	}

	target = trace->ent;
	CoopBotDiag_Log(2,
		"shot kind=%s attacker=%d attacker_class=\"%s\" damage=%d mod=%d "
		"fraction=%.5f target=%d target_class=\"%s\" inuse=%d solid=%d "
		"takedamage=%d svflags=0x%x health=%d max_health=%d origin=(%.1f %.1f %.1f)",
		kind != NULL ? kind : "<unknown>",
		CoopBotDiag_EntityNumber(attacker), CoopBotDiag_Classname(attacker),
		damage, mod, trace->fraction, CoopBotDiag_EntityNumber(target),
		CoopBotDiag_Classname(target), target != NULL ? target->inuse : 0,
		target != NULL ? target->solid : 0,
		target != NULL ? target->takedamage : 0,
		target != NULL ? target->svflags : 0,
		target != NULL ? target->health : 0,
		target != NULL ? target->max_health : 0,
		target != NULL ? target->s.origin[0] : 0.0f,
		target != NULL ? target->s.origin[1] : 0.0f,
		target != NULL ? target->s.origin[2] : 0.0f);
}

void CoopBotDiag_RecordProjectileLaunch(edict_t *projectile,
	edict_t *owner,
	int damage,
	int mod)
{
	if (CoopBotDiag_LogLevel() < 2 || projectile == NULL)
	{
		return;
	}

	CoopBotDiag_Log(2,
		"projectile_launch projectile=%d owner=%d damage=%d mod=%d "
		"origin=(%.1f %.1f %.1f) velocity=(%.1f %.1f %.1f) "
		"clipmask=0x%x solid=%d mins=(%.1f %.1f %.1f) maxs=(%.1f %.1f %.1f)",
		CoopBotDiag_EntityNumber(projectile), CoopBotDiag_EntityNumber(owner),
		damage, mod,
		projectile->s.origin[0], projectile->s.origin[1], projectile->s.origin[2],
		projectile->velocity[0], projectile->velocity[1], projectile->velocity[2],
		projectile->clipmask, projectile->solid,
		projectile->mins[0], projectile->mins[1], projectile->mins[2],
		projectile->maxs[0], projectile->maxs[1], projectile->maxs[2]);
}

void CoopBotDiag_RecordProjectileTouch(edict_t *projectile,
	edict_t *other,
	int damage,
	int mod)
{
	if (CoopBotDiag_LogLevel() < 2 || projectile == NULL)
	{
		return;
	}

	CoopBotDiag_Log(2,
		"projectile_touch projectile=%d owner=%d target=%d class=\"%s\" "
		"damage=%d mod=%d origin=(%.1f %.1f %.1f) target_origin=(%.1f %.1f %.1f) "
		"target_mins=(%.1f %.1f %.1f) target_maxs=(%.1f %.1f %.1f) "
		"target_absmin=(%.1f %.1f %.1f) target_absmax=(%.1f %.1f %.1f) "
		"inuse=%d takedamage=%d solid=%d svflags=0x%x",
		CoopBotDiag_EntityNumber(projectile),
		CoopBotDiag_EntityNumber(projectile->owner),
		CoopBotDiag_EntityNumber(other), CoopBotDiag_Classname(other),
		damage, mod,
		projectile->s.origin[0], projectile->s.origin[1], projectile->s.origin[2],
		other != NULL ? other->s.origin[0] : 0.0f,
		other != NULL ? other->s.origin[1] : 0.0f,
		other != NULL ? other->s.origin[2] : 0.0f,
		other != NULL ? other->mins[0] : 0.0f,
		other != NULL ? other->mins[1] : 0.0f,
		other != NULL ? other->mins[2] : 0.0f,
		other != NULL ? other->maxs[0] : 0.0f,
		other != NULL ? other->maxs[1] : 0.0f,
		other != NULL ? other->maxs[2] : 0.0f,
		other != NULL ? other->absmin[0] : 0.0f,
		other != NULL ? other->absmin[1] : 0.0f,
		other != NULL ? other->absmin[2] : 0.0f,
		other != NULL ? other->absmax[0] : 0.0f,
		other != NULL ? other->absmax[1] : 0.0f,
		other != NULL ? other->absmax[2] : 0.0f,
		other != NULL ? other->inuse : 0,
		other != NULL ? other->takedamage : 0,
		other != NULL ? other->solid : 0,
		other != NULL ? other->svflags : 0);
}

void CoopBotDiag_RecordDamageAttempt(edict_t *targ,
	edict_t *inflictor,
	edict_t *attacker,
	int damage,
	int dflags,
	int mod)
{
	if (CoopBotDiag_LogLevel() < 2)
	{
		return;
	}

	CoopBotDiag_Log(2,
		"damage_attempt target=%d class=\"%s\" inflictor=%d attacker=%d "
		"requested=%d dflags=0x%x mod=%d takedamage=%d health=%d deadflag=%d "
		"solid=%d svflags=0x%x",
		CoopBotDiag_EntityNumber(targ), CoopBotDiag_Classname(targ),
		CoopBotDiag_EntityNumber(inflictor), CoopBotDiag_EntityNumber(attacker),
		damage, dflags, mod, targ != NULL ? targ->takedamage : 0,
		targ != NULL ? targ->health : 0, targ != NULL ? targ->deadflag : 0,
		targ != NULL ? targ->solid : 0, targ != NULL ? targ->svflags : 0);
}

void CoopBotDiag_RecordDamageApplied(edict_t *targ,
	edict_t *attacker,
	int requested,
	int applied,
	int health_before,
	int mod)
{
	if (CoopBotDiag_LogLevel() < 2)
	{
		return;
	}

	CoopBotDiag_Log(2,
		"damage_applied target=%d class=\"%s\" attacker=%d requested=%d "
		"applied=%d health=%d->%d mod=%d deadflag=%d solid=%d takedamage=%d",
		CoopBotDiag_EntityNumber(targ), CoopBotDiag_Classname(targ),
		CoopBotDiag_EntityNumber(attacker), requested, applied, health_before,
		targ != NULL ? targ->health : 0, mod,
		targ != NULL ? targ->deadflag : 0,
		targ != NULL ? targ->solid : 0,
		targ != NULL ? targ->takedamage : 0);
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
	{
		char line[2048];
		snprintf(line, sizeof(line),
			"metrics report=%llu map=\"%s\" time=%.1f frames=%llu bots=%d "
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
		line[sizeof(line) - 1] = '\0';
		CoopBotDiag_WriteLine(line);
	}

	for (client = 0; client < game.maxclients; ++client)
	{
		coopbot_diag_client_t *stats = &coopbot_diag.clients[client];
		if (!stats->active)
		{
			continue;
		}

		{
			char line[1024];
			snprintf(line, sizeof(line),
				"metrics bot client=%d name=\"%s\" ai=%llu errors=%llu "
			"inputs=%llu attack=%llu use=%llu jump=%llu crouch=%llu\n",
			client, stats->name, stats->ai_calls, stats->ai_errors,
			stats->input_calls, stats->attack_actions, stats->use_actions,
			stats->jump_actions, stats->crouch_actions);
			line[sizeof(line) - 1] = '\0';
			CoopBotDiag_WriteLine(line);
		}
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

	if (cmd != NULL && Q_stricmp(cmd, "coopbot_map") == 0)
	{
		CoopBotDiag_DumpMapEntities();
		return true;
	}

	return false;
}
