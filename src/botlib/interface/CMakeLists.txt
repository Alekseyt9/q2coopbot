add_library(botlib_interface STATIC
    bot_interface.c
    bot_interface_area.c
    bot_interface_assets.c
    bot_interface_elevator.c
    bot_interface_map.c
    bot_interface_objective.c
    bot_state.c
    botlib_interface.c
)

register_botlib_sources(
    TARGET botlib_interface
    SOURCES
        bot_interface.c
        bot_interface_area.c
        bot_interface_assets.c
        bot_interface_elevator.c
        bot_interface_map.c
        bot_interface_objective.c
        bot_state.c
        botlib_interface.c
)

target_link_libraries(botlib_interface
    PUBLIC
        botlib_common
        botlib_precomp
        botlib_aas
        botlib_ai
        botlib_ea
        q2bridge
        gladiator_shared_headers
)
