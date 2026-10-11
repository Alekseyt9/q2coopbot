package policy

import (
	"encoding/json"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"
)

func TestEquippedWeaponFeaturesPrefixAliasesAndOwnRequest(t *testing.T) {
	for i, name := range weaponNames[1:] {
		for _, alias := range []string{name, "models/weapons/" + equippedWeaponModels[i] + "/tris.md2", strings.ToUpper(name)} {
			o := Observation{Weapon: alias}
			base, err := FeaturesForVersion(o, ThreatFeatureVersion)
			if err != nil {
				t.Fatal(err)
			}
			v, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
			if err != nil || len(v) != EquippedWeaponFeatureWidth || !reflect.DeepEqual(v[:ThreatFeatureWidth], base) || v[ThreatFeatureWidth+i+1] != 1 {
				t.Fatal("prefix/alias mismatch", alias, err)
			}
		}
	}
	name, age := "Shotgun", 150
	o := Observation{Weapon: "Machinegun", PreviousWeaponRequest: &name, PreviousWeaponRequestAgeFrames: &age}
	tail, err := equippedWeaponFeatures(o)
	if err != nil || tail[4] != 1 || tail[14] != 1 || tail[24] != 1 || tail[25] != 4 {
		t.Fatal("request is not independent of equipped weapon", tail, err)
	}
	for _, o := range []Observation{{PreviousWeaponRequest: &name}, {PreviousWeaponRequestAgeFrames: &age}} {
		if _, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion); err == nil {
			t.Fatal("incomplete request accepted")
		}
	}
}

func TestEquippedWeaponFeaturesPythonParity(t *testing.T) {
	python := os.Getenv("Q2_PARITY_PYTHON")
	if python == "" {
		t.Skip("set Q2_PARITY_PYTHON for feature-only host parity")
	}
	cases := []Observation{{}, {Weapon: "modded"}}
	for i, name := range weaponNames[1:] {
		age := i * 40
		request := name
		cases = append(cases, Observation{Weapon: "models/weapons/" + equippedWeaponModels[i] + "/tris.md2", PreviousWeaponRequest: &request, PreviousWeaponRequestAgeFrames: &age})
	}
	data, _ := json.Marshal(cases)
	cmd := exec.Command(python, "-c", "import json,sys;sys.path.insert(0,'../../scripts');from combat_equipped_weapon_features import equipped_weapon_tail,weapon_request_tail;print(json.dumps([equipped_weapon_tail(o)+weapon_request_tail(o) for o in json.loads(sys.argv[1])]))", string(data))
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("oracle: %v %s", err, output)
	}
	var expected [][]float64
	if err = json.Unmarshal(output, &expected); err != nil || len(expected) != len(cases) {
		t.Fatal("invalid oracle", err)
	}
	for i, o := range cases {
		got, err := equippedWeaponFeatures(o)
		if err != nil || !reflect.DeepEqual(got, expected[i]) {
			t.Fatal("Go/Python mismatch", i, got, expected[i], err)
		}
	}
}
