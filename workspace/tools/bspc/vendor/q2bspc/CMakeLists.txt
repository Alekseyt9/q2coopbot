# Quake II reachability-capable BSPC fork (GPL-2.0-or-later).
# Upstream revision and local compiler fixes are recorded in UPSTREAM.md.
set(BSPC_SOURCES
    _files.c aas_areamerging.c aas_cfg.c aas_create.c aas_edgemelting.c
    aas_facemerging.c aas_file.c aas_gsubdiv.c aas_map.c aas_prunenodes.c
    aas_store.c be_aas_bspc.c
    deps/botlib/be_aas_bspq3.c deps/botlib/be_aas_cluster.c
    deps/botlib/be_aas_move.c deps/botlib/be_aas_optimize.c
    deps/botlib/be_aas_reach.c deps/botlib/be_aas_sample.c
    brushbsp.c bspc.c
    deps/qcommon/cm_common.c deps/qcommon/cm_load.c
    deps/qcommon/cm_load_q2.c deps/qcommon/cm_patch.c
    deps/qcommon/cm_test.c deps/qcommon/cm_trace.c
    csg.c glfile.c l_bsp_ent.c l_bsp_hl.c l_bsp_q1.c l_bsp_q2.c
    l_bsp_q3.c l_bsp_sin.c l_cmd.c deps/botlib/l_libvar.c
    l_log.c l_math.c l_mem.c l_poly.c deps/botlib/l_precomp.c
    l_qfiles.c deps/botlib/l_script.c deps/botlib/l_struct.c
    l_threads.c l_utils.c leakfile.c map.c map_hl.c map_q1.c
    map_q2.c map_q3.c map_sin.c deps/qcommon/md4.c
    nodraw.c portals.c textures.c tree.c deps/qcommon/unzip.c
)

add_executable(bspc ${BSPC_SOURCES})
set_target_properties(bspc PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/workspace/tools/bspc"
)
target_include_directories(bspc PRIVATE . deps)
target_compile_definitions(bspc PRIVATE
    stricmp=strcasecmp Com_Memcpy=memcpy Com_Memset=memset
    MAC_STATIC= QDECL= BSPC _FORTIFY_SOURCE=2
)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bspc PRIVATE -fno-common -ffast-math)
endif()
if(WIN32)
    target_link_libraries(bspc PRIVATE m pthread)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(bspc PRIVATE m Threads::Threads)
endif()
