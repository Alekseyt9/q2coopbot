"""Resolve fixture release in network clock; never assume a game-frame offset."""
import re

RELEASE=re.compile(r'^sv_test_combat spawncount=(\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')


def release_boundary(stream,manifest,expected_seed):
    assert manifest['synchronous'] and manifest['fixed_world_hold'] and manifest['release_game_frame']>0
    found=[]
    for line in stream:
        if 'g_test_combat_start game_frame=' not in line:continue
        match=RELEASE.fullmatch(line.strip());assert match,'Malformed native release'
        found.append(tuple(map(int,match.groups())))
    assert len(found)==1,'Exactly one native release required'
    generation,network_frame,game_frame,seed=found[0]
    assert game_frame==manifest['release_game_frame'] and seed==expected_seed
    return generation,network_frame,game_frame
