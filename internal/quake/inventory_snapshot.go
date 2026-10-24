package quake

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
)

// The isolated native harness reserves the last general configstring. Stock
// clients can ignore it; the packet remains a protocol-34 configstring message.
const inventorySnapshotConfig = 2079
const inventorySnapshotPrefix = "q2inv1:"

// A snapshot replaces all slots, including clearing absent ones. The compact
// body contains sorted (slot byte, little-endian signed count short) triplets.
func (d *Decoder) applyInventorySnapshot(value string) error {
	parts := strings.SplitN(value, ":", 4)
	if len(parts) != 4 || parts[0] != "q2inv1" {
		return fmt.Errorf("invalid inventory snapshot header")
	}
	generation, err := strconv.ParseInt(parts[1], 10, 32)
	if err != nil || int(generation) != d.Spawncount {
		return fmt.Errorf("inventory snapshot generation mismatch")
	}
	frame, err := strconv.ParseInt(parts[2], 10, 32)
	if err != nil || frame < 0 || d.InventoryKnown && int(frame) < d.InventoryFrame {
		return fmt.Errorf("invalid inventory snapshot frame")
	}
	if len(parts[3]) > 256*6 || len(parts[3])%6 != 0 {
		return fmt.Errorf("invalid inventory snapshot length")
	}
	data, err := hex.DecodeString(parts[3])
	if err != nil {
		return fmt.Errorf("invalid inventory snapshot payload: %w", err)
	}
	var inventory [256]int16
	previous := -1
	for i := 0; i < len(data); i += 3 {
		slot := int(data[i])
		count := int16(binary.LittleEndian.Uint16(data[i+1 : i+3]))
		if slot <= previous || count == 0 {
			return fmt.Errorf("invalid inventory snapshot slot order or zero count")
		}
		inventory[slot] = count
		previous = slot
	}
	d.Inventory, d.InventoryKnown, d.InventoryFrame = inventory, true, int(frame)
	return nil
}
