package quake

import (
	"encoding/binary"
	"strings"
	"testing"
)

func inventorySnapshotPacket(index uint16, value string) []byte {
	p := binary.LittleEndian.AppendUint16([]byte{13}, index)
	return append(append(p, value...), 0)
}

func TestInventorySnapshotReplacementAndFrameBinding(t *testing.T) {
	d := NewDecoder()
	d.Spawncount = 7
	d.latestFrame = 128
	d.Inventory[18] = 30
	d.Config[1056+7], d.Config[1056+19] = "Blaster", "Bullets"
	// Exact native encoding: byte slot, signed little-endian short count.
	value := "q2inv1:7:129:070100131100ffffff"
	if _, err := d.Parse(inventorySnapshotPacket(inventorySnapshotConfig, value)); err != nil {
		t.Fatal(err)
	}
	if !d.InventoryKnown || d.InventoryFrame != 129 || d.Inventory[7] != 1 || d.Inventory[19] != 17 || d.Inventory[255] != -1 || d.Inventory[18] != 0 {
		t.Fatalf("snapshot replacement: frame=%d slots=%v", d.InventoryFrame, d.Inventory)
	}
	s := d.Snapshot(Frame{Number: 129})
	if s.InventoryAgeFrames != 0 || len(s.Inventory) != 3 {
		t.Fatalf("tagged frame was replaced by previous packet frame: %+v", s.Inventory)
	}
	if _, err := d.Parse(inventorySnapshotPacket(inventorySnapshotConfig, "q2inv1:7:130:")); err != nil {
		t.Fatal(err)
	}
	if d.Inventory != ([256]int16{}) || !d.InventoryKnown || d.InventoryFrame != 130 || s.Inventory[1].Count != 17 {
		t.Fatal("empty replacement or earlier snapshot isolation broken")
	}
}

func TestInventorySnapshotRejectsMalformedWithoutPartialUpdate(t *testing.T) {
	values := []string{
		"q2inv1:8:130:070100", "q2inv1:7:-1:070100", "q2inv1:7:128:070100",
		"q2inv1:7:2147483648:070100", "q2inv1:7:130:07", "q2inv1:7:130:07zz00",
		"q2inv1:7:130:070100070200", "q2inv1:7:130:130100070100", "q2inv1:7:130:070000",
		"q2inv1:7:130:" + strings.Repeat("070100", 257),
	}
	for _, value := range values {
		t.Run(value[:min(48, len(value))], func(t *testing.T) {
			d := NewDecoder()
			d.Spawncount, d.InventoryFrame, d.InventoryKnown = 7, 129, true
			d.Inventory[19] = 18
			before := d.Inventory
			if _, err := d.Parse(inventorySnapshotPacket(inventorySnapshotConfig, value)); err == nil {
				t.Fatal("invalid snapshot accepted")
			}
			if d.Inventory != before || d.InventoryFrame != 129 || !d.InventoryKnown {
				t.Fatal("invalid snapshot partially changed inventory")
			}
		})
	}
}

func TestInventorySnapshotDoesNotInterpretOtherConfigstrings(t *testing.T) {
	d := NewDecoder()
	if _, err := d.Parse(inventorySnapshotPacket(inventorySnapshotConfig-1, "q2inv1:7:129:070100")); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Parse(inventorySnapshotPacket(inventorySnapshotConfig, "ordinary general string")); err != nil {
		t.Fatal(err)
	}
	if d.InventoryKnown {
		t.Fatal("unrelated configstring changed inventory")
	}
}
