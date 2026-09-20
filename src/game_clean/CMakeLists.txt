# Clean Yamagi baseq2 gameplay plus the optional Gladiator bot bridge.
# This target intentionally has a different name until the port is proven.

file(GLOB_RECURSE CLEAN_GAME_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
list(FILTER CLEAN_GAME_SOURCES EXCLUDE REGEX "/CMakeFiles/")

if(NOT CLEAN_GAME_SOURCES)
    message(FATAL_ERROR "No clean game sources found")
endif()

add_library(game_clean SHARED ${CLEAN_GAME_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/game.def")

target_include_directories(game_clean PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/common")

target_compile_definitions(game_clean PRIVATE C_ONLY BOT BOT_IMPORT)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(game_clean PRIVATE
        "-include${CMAKE_CURRENT_SOURCE_DIR}/coopbot_local.h")
endif()

set_target_properties(game_clean PROPERTIES
    OUTPUT_NAME "gamex86_clean"
    PREFIX ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    C_STANDARD 99
    C_EXTENSIONS ON)

if(WIN32)
    if(MINGW)
        target_sources(game_clean PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/game.def")
    elseif(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_link_options(game_clean PRIVATE
            "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/game.def")
    else()
        target_link_options(game_clean PRIVATE
            "-Wl,/DEF:${CMAKE_CURRENT_SOURCE_DIR}/game.def")
    endif()
    target_compile_definitions(game_clean PRIVATE WIN32)
elseif(UNIX AND NOT APPLE)
    target_link_options(game_clean PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/game.exports")
    set_property(TARGET game_clean APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/game.exports")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(game_clean PRIVATE m ${CMAKE_DL_LIBS})
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(game_clean PRIVATE
        -Wno-unused-but-set-variable
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-parentheses
        -Wno-misleading-indentation
        -Wno-implicit-fallthrough
        -Wno-unknown-pragmas
        -Wno-format-truncation
        -Wno-implicit-function-declaration)
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(game_clean PRIVATE
        -Wno-stringop-truncation
        -Wno-stringop-overflow)
endif()

if(MSVC)
    target_compile_definitions(game_clean PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

install(TARGETS game_clean
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
