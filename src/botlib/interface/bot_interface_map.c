#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "botlib/common/l_libvar.h"
#include "botlib/common/l_log.h"
#include "botlib/aas/aas_local.h"
#include "bot_interface_map.h"

static bool g_botInterfaceMapModelDumped = false;
static bool g_botInterfaceMapModelWaitLogged = false;

void BotInterface_ResetMapModel(void)
{
	g_botInterfaceMapModelDumped = false;
	g_botInterfaceMapModelWaitLogged = false;
}

/*
=============
BotInterface_DumpCoopMapRegions

Emit a coarse room/region segmentation from AAS clusters.  Clusters are the
stable map-level partition available before any game-specific objective is
known; portal areas are intentionally left out rather than assigned to both
sides of a boundary.
=============
*/
static void BotInterface_DumpCoopMapRegions(void)
{
	int region_count = 0;
	if (aasworld.areas == NULL || aasworld.areasettings == NULL)
	{
		BotLib_LogWrite(
			"coopbot_map_regions count=0 reason=areas_unavailable");
		return;
	}

	for (int cluster = 1; cluster < aasworld.numClusters; ++cluster)
	{
		int area_count = 0;
		vec3_t mins = {FLT_MAX, FLT_MAX, FLT_MAX};
		vec3_t maxs = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
		vec3_t center_sum = {0.0f, 0.0f, 0.0f};

		for (int area = 1; area < aasworld.numAreas; ++area)
		{
			if (area >= aasworld.numAreaSettings ||
				aasworld.areasettings[area].cluster != cluster)
			{
				continue;
			}

			const aas_area_t *area_data = &aasworld.areas[area];
			++area_count;
			for (int axis = 0; axis < 3; ++axis)
			{
				if (area_data->mins[axis] < mins[axis])
				{
					mins[axis] = area_data->mins[axis];
				}
				if (area_data->maxs[axis] > maxs[axis])
				{
					maxs[axis] = area_data->maxs[axis];
				}
				center_sum[axis] += area_data->center[axis];
			}
		}

		if (area_count == 0)
		{
			continue;
		}
		VectorScale(center_sum, 1.0f / (float)area_count, center_sum);
		BotLib_LogWrite(
			"coopbot_map_region region=%d cluster=%d areas=%d "
			"mins=(%.1f %.1f %.1f) maxs=(%.1f %.1f %.1f) "
			"center=(%.1f %.1f %.1f)",
			cluster,
			cluster,
			area_count,
			mins[0], mins[1], mins[2],
			maxs[0], maxs[1], maxs[2],
			center_sum[0], center_sum[1], center_sum[2]);
		++region_count;
	}
	BotLib_LogWrite("coopbot_map_regions count=%d", region_count);
}

/*
=============
BotInterface_DumpCoopMapModel

Emit the AAS area/reachability model and the map controls that can change it.
This is an opt-in P3 diagnostic: the normal botlib log stays compact, while a
coop episode can prove whether a vertical route and its mover entities exist.
=============
*/
void BotInterface_DumpCoopMapModel(const aas_bspentity_t *entities)
{
	int area;
	int reachability;
	int elevator_count = 0;
	int control_count = 0;

	if (g_botInterfaceMapModelDumped ||
		LibVarGetValue("coopbot_map_model") == 0.0f)
	{
		return;
	}
	if (!AAS_Initialized())
	{
		if (!g_botInterfaceMapModelWaitLogged)
		{
			g_botInterfaceMapModelWaitLogged = true;
			BotLib_LogWrite("coopbot_map_model_wait aas_initialized=0");
		}
		return;
	}

	g_botInterfaceMapModelDumped = true;
	for (reachability = 1; reachability < aasworld.numReachability;
		reachability += 1)
	{
		if (aasworld.reachability[reachability].traveltype == TRAVEL_ELEVATOR)
		{
			elevator_count += 1;
		}
	}
	BotLib_LogWrite(
		"coopbot_map_model map=\"%s\" areas=%d clusters=%d reachabilities=%d elevators=%d",
		aasworld.mapName,
		aasworld.numAreas,
		aasworld.numClusters,
		aasworld.numReachability,
		elevator_count);
	BotInterface_DumpCoopMapRegions();

	for (area = 1; area < aasworld.numAreas; area += 1)
	{
		const aas_area_t *area_data = &aasworld.areas[area];
		const aas_areasettings_t *settings =
			area < aasworld.numAreaSettings ? &aasworld.areasettings[area] : NULL;

		BotLib_LogWrite(
			"coopbot_map_area area=%d cluster=%d contents=0x%x flags=0x%x "
			"presence=0x%x reachable=%d mins=(%.1f %.1f %.1f) "
			"maxs=(%.1f %.1f %.1f) center=(%.1f %.1f %.1f)",
			area,
			settings != NULL ? settings->cluster : -1,
			settings != NULL ? settings->contents : 0,
			settings != NULL ? settings->areaflags : 0,
			settings != NULL ? settings->presencetype : 0,
			settings != NULL ? settings->numreachableareas : 0,
			area_data->mins[0], area_data->mins[1], area_data->mins[2],
			area_data->maxs[0], area_data->maxs[1], area_data->maxs[2],
			area_data->center[0], area_data->center[1], area_data->center[2]);
	}

	for (reachability = 1; reachability < aasworld.numReachability;
		reachability += 1)
	{
		const aas_reachability_t *reach = &aasworld.reachability[reachability];
		int source_area = aasworld.reachabilityFromArea != NULL
			? aasworld.reachabilityFromArea[reachability] : 0;

		if (source_area <= 0)
		{
			continue;
		}
		BotLib_LogWrite(
			"coopbot_map_edge reach=%d from=%d to=%d traveltype=%d "
			"traveltime=%d facenum=%d edgenum=%d start=(%.1f %.1f %.1f) "
			"end=(%.1f %.1f %.1f)",
			reachability, source_area, reach->areanum, reach->traveltype,
			reach->traveltime, reach->facenum, reach->edgenum,
			reach->start[0], reach->start[1], reach->start[2],
			reach->end[0], reach->end[1], reach->end[2]);
	}

	if (entities == NULL)
	{
		BotLib_LogWrite("coopbot_map_controls count=0 reason=entities_unavailable");
		return;
	}
	for (const aas_bspentity_t *entity = entities; entity != NULL;
		entity = entity->next)
	{
		const char *classname = AAS_ValueForBSPEpairKey(entity, "classname");
		const char *model = AAS_ValueForBSPEpairKey(entity, "model");
		vec3_t entity_origin;
		vec3_t model_mins;
		vec3_t model_maxs;
		vec3_t model_origin;
		vec3_t angles = {0.0f, 0.0f, 0.0f};
		int modelnum = 0;
		VectorClear(entity_origin);
		VectorClear(model_mins);
		VectorClear(model_maxs);
		VectorClear(model_origin);
		(void)AAS_VectorForBSPEpairKey(entity, "origin", entity_origin);
		if (model != NULL && model[0] == '*')
		{
			modelnum = (int)strtol(model + 1, NULL, 10);
			if (modelnum > 0 && modelnum < aasworld.numBspModels)
			{
				AAS_BSPModelMinsMaxsOrigin(modelnum, angles,
					model_mins, model_maxs, model_origin);
			}
		}
		if (classname == NULL ||
			(strncmp(classname, "func_", 5) != 0 &&
				strncmp(classname, "trigger_", 8) != 0 &&
				strncmp(classname, "target_", 7) != 0))
		{
			continue;
		}

		control_count += 1;
		BotLib_LogWrite(
			"coopbot_map_control class=\"%s\" model=\"%s\" "
			"target=\"%s\" targetname=\"%s\" speed=%.1f height=%.1f "
			"lip=%.1f spawnflags=%d map=\"%s\" message=\"%s\"",
			classname,
			model != NULL ? model : "",
			AAS_ValueForBSPEpairKey(entity, "target") != NULL
				? AAS_ValueForBSPEpairKey(entity, "target") : "",
			AAS_ValueForBSPEpairKey(entity, "targetname") != NULL
				? AAS_ValueForBSPEpairKey(entity, "targetname") : "",
			AAS_FloatForBSPEpairKey(entity, "speed"),
			AAS_FloatForBSPEpairKey(entity, "height"),
			AAS_FloatForBSPEpairKey(entity, "lip"),
			AAS_IntForBSPEpairKey(entity, "spawnflags"),
			AAS_ValueForBSPEpairKey(entity, "map") != NULL
				? AAS_ValueForBSPEpairKey(entity, "map") : "",
			AAS_ValueForBSPEpairKey(entity, "message") != NULL
				? AAS_ValueForBSPEpairKey(entity, "message") : "");
		if (strcmp(classname, "trigger_changelevel") == 0 ||
			strcmp(classname, "target_changelevel") == 0)
		{
			BotLib_LogWrite(
				"coopbot_map_transition class=\"%s\" map=\"%s\" "
				"message=\"%s\" target=\"%s\"",
				classname,
				AAS_ValueForBSPEpairKey(entity, "map") != NULL
					? AAS_ValueForBSPEpairKey(entity, "map") : "",
				AAS_ValueForBSPEpairKey(entity, "message") != NULL
					? AAS_ValueForBSPEpairKey(entity, "message") : "",
				AAS_ValueForBSPEpairKey(entity, "target") != NULL
					? AAS_ValueForBSPEpairKey(entity, "target") : "");
		}
		BotLib_LogWrite(
			"coopbot_map_geometry class=\"%s\" model=\"%s\" "
			"origin=(%.1f %.1f %.1f) model_origin=(%.1f %.1f %.1f) "
			"mins=(%.1f %.1f %.1f) maxs=(%.1f %.1f %.1f)",
			classname,
			model != NULL ? model : "",
			entity_origin[0], entity_origin[1], entity_origin[2],
			model_origin[0], model_origin[1], model_origin[2],
			model_mins[0], model_mins[1], model_mins[2],
			model_maxs[0], model_maxs[1], model_maxs[2]);
	}
	for (const aas_bspentity_t *source = entities; source != NULL;
		source = source->next)
	{
		const char *source_class = AAS_ValueForBSPEpairKey(source, "classname");
		const char *target = AAS_ValueForBSPEpairKey(source, "target");
		bool source_is_control = source_class != NULL &&
			(strncmp(source_class, "func_", 5) == 0 ||
				strncmp(source_class, "trigger_", 8) == 0 ||
				strncmp(source_class, "target_", 7) == 0);

		if (!source_is_control || target == NULL || target[0] == '\0')
		{
			continue;
		}
		bool resolved = false;
		for (const aas_bspentity_t *destination = entities;
			destination != NULL;
			destination = destination->next)
		{
			const char *targetname = AAS_ValueForBSPEpairKey(destination,
				"targetname");
			const char *destination_class = AAS_ValueForBSPEpairKey(
				destination, "classname");

			if (targetname == NULL || destination_class == NULL ||
				strcmp(targetname, target) != 0)
			{
				continue;
			}
			resolved = true;
			BotLib_LogWrite(
				"coopbot_map_control_link source=\"%s\" target=\"%s\" "
				"destination=\"%s\"",
				source_class, target, destination_class);
		}
		if (!resolved)
		{
			BotLib_LogWrite(
				"coopbot_map_control_unresolved source=\"%s\" target=\"%s\"",
				source_class, target);
		}
	}
	BotLib_LogWrite("coopbot_map_controls count=%d", control_count);
}

static bool BotInterface_MapEntityLeadsToChangelevel(
	const aas_bspentity_t *entities,
	const aas_bspentity_t *entity,
	int depth)
{
	const char *classname;
	const char *target;

	if (entities == NULL || entity == NULL || depth >= 8)
	{
		return false;
	}
	classname = AAS_ValueForBSPEpairKey(entity, "classname");
	if (classname == NULL)
	{
		return false;
	}
	if (strcmp(classname, "target_changelevel") == 0 ||
		strcmp(classname, "trigger_changelevel") == 0)
	{
		return true;
	}
	target = AAS_ValueForBSPEpairKey(entity, "target");
	if (target == NULL || target[0] == '\0')
	{
		return false;
	}
	for (const aas_bspentity_t *candidate = entities;
		candidate != NULL;
		candidate = candidate->next)
	{
		const char *targetname = AAS_ValueForBSPEpairKey(candidate,
			"targetname");
		if (candidate != entity && targetname != NULL &&
			strcmp(targetname, target) == 0 &&
			BotInterface_MapEntityLeadsToChangelevel(entities,
				candidate,
				depth + 1))
		{
			return true;
		}
	}
	return false;
}

bool BotInterface_MapPointInsideChangelevelTrigger(
	const aas_bspentity_t *entities,
	const aas_bspentity_t *entity,
	const vec3_t point,
	int *model_number)
{
	const char *classname;
	const char *model;
	vec3_t mins;
	vec3_t maxs;
	vec3_t model_origin;
	vec3_t zero_angles = {0.0f, 0.0f, 0.0f};
	int modelnum;

	if (entities == NULL || entity == NULL || point == NULL)
	{
		return false;
	}
	classname = AAS_ValueForBSPEpairKey(entity, "classname");
	if (classname == NULL ||
		(strcmp(classname, "trigger_once") != 0 &&
			strcmp(classname, "trigger_multiple") != 0 &&
			strcmp(classname, "trigger_changelevel") != 0))
	{
		return false;
	}
	model = AAS_ValueForBSPEpairKey(entity, "model");
	if (model == NULL || model[0] != '*')
	{
		return false;
	}
	modelnum = (int)strtol(model + 1, NULL, 10);
	if (modelnum <= 0 || modelnum > aasworld.numBspModels)
	{
		return false;
	}
	if (!BotInterface_MapEntityLeadsToChangelevel(entities, entity, 0))
	{
		return false;
	}
	AAS_BSPModelMinsMaxsOrigin(modelnum - 1,
		zero_angles, mins, maxs, model_origin);
	for (int axis = 0; axis < 3; ++axis)
	{
		if (point[axis] < model_origin[axis] + mins[axis] - 16.0f ||
			point[axis] > model_origin[axis] + maxs[axis] + 16.0f)
		{
			return false;
		}
	}
	if (model_number != NULL)
	{
		*model_number = modelnum;
	}
	return true;
}

const aas_bspentity_t *BotInterface_MapFindChangelevelTrigger(
	const aas_bspentity_t *entities,
	const vec3_t bot_origin,
	int *model_number)
{
	if (entities == NULL || bot_origin == NULL)
	{
		return NULL;
	}
	for (const aas_bspentity_t *entity = entities;
		entity != NULL;
		entity = entity->next)
	{
		if (BotInterface_MapPointInsideChangelevelTrigger(entities,
			entity,
			bot_origin,
			model_number))
		{
			return entity;
		}
	}
	return NULL;
}
