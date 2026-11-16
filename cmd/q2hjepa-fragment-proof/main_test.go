package main

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestNativeCommandProofRejectsMissingDuplicateAndWrongApplication(t *testing.T) {
	r := row{Frame: 3, Generation: 7, Sequence: 11, Actor: 1, Command: quake.UserCmd{Forward: 23, Msec: 100}}
	log := []string{
		"sv_test_step version=1 phase=begin spawncount=7 frame=3 seq=11 actor=1",
		"sv_test_applied_cmd spawncount=7 frame=3 seq=11 kind=new pitch=0 yaw=0 roll=0 forward=23 side=0 up=0 buttons=0 impulse=0 msec=100 light=0",
		"sv_test_step version=1 phase=end spawncount=7 frame=4 seq=11 actor=1",
	}
	if err := applied(log, r); err != nil {
		t.Fatal(err)
	}
	if applied(log[:2], r) == nil {
		t.Fatal("missing world step admitted")
	}
	if applied(append(append([]string{}, log...), log[1]), r) == nil {
		t.Fatal("duplicate native cmd admitted")
	}
	r.Command.Forward++
	if applied(log, r) == nil {
		t.Fatal("different native cmd admitted")
	}
}

func TestEffectWindowUsesCausalCommandsRatherThanNextFrameEvents(t *testing.T) {
	log := []string{
		"sv_test_step version=1 phase=begin spawncount=7 frame=3 seq=11 actor=1",
		"sv_test_damage spawncount=7 server_frame=4 g_test_damage version=1 frame=120 take=3",
		"sv_test_step version=1 phase=end spawncount=7 frame=4 seq=11 actor=1",
		"sv_test_step version=1 phase=begin spawncount=7 frame=4 seq=12 actor=1",
		"g_test_projectile version=1 event=spawn frame=120 shot=9",
	}
	got := effects(log, 7, 3, 1)
	if len(got) != 1 || got[0] != "0:g_test_damage version=1 frame=120 take=3" {
		t.Fatal(got)
	}
}
