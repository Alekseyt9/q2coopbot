package quake

import (
	"encoding/binary"
	"testing"
)

func TestInventoryUpdatesAtomicallyAndKeepsUnknownSlots(t *testing.T) {
	d := NewDecoder()
	d.Config[1056+7] = "Shotgun"
	d.Config[1056+18] = "Shells"
	packet := make([]byte, 513)
	packet[0] = 5
	binary.LittleEndian.PutUint16(packet[1+7*2:], 1)
	binary.LittleEndian.PutUint16(packet[1+18*2:], 3)
	binary.LittleEndian.PutUint16(packet[1+255*2:], 2)
	if _, err := d.Parse(packet); err != nil {
		t.Fatal(err)
	}
	s := d.Snapshot(Frame{Number: 10})
	if !s.InventoryKnown || len(s.Inventory) != 3 || s.Inventory[0].Name != "Shotgun" || s.Inventory[1].Count != 3 || s.Inventory[2].ID != 255 || s.Inventory[2].Name != "" {
		t.Fatalf("inventory: %+v", s.Inventory)
	}
	if _, err := d.Parse(packet[:512]); err == nil {
		t.Fatal("truncated inventory accepted")
	}
	if d.Inventory[18] != 3 {
		t.Fatal("partial update changed inventory")
	}
	binary.LittleEndian.PutUint16(packet[1+18*2:], 0)
	if _, err := d.Parse(packet); err != nil {
		t.Fatal(err)
	}
	if d.Inventory[18] != 0 || s.Inventory[1].Count != 3 {
		t.Fatal("zero update or snapshot isolation broken")
	}
	serverdata := []byte{12}
	serverdata = binary.LittleEndian.AppendUint32(serverdata, 34)
	serverdata = binary.LittleEndian.AppendUint32(serverdata, 2)
	serverdata = append(serverdata, 0, 0, 0, 0, 0) // attract, game, player short, level.
	if _, err := d.Parse(serverdata); err != nil {
		t.Fatal(err)
	}
	if d.InventoryKnown || d.Inventory[7] != 0 {
		t.Fatal("inventory leaked across serverdata")
	}
}
