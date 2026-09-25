package quake

import (
	"embed"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strings"
)

//go:embed chktbl.hex
var tableFile embed.FS
var checkTable [1024]byte

func init() {
	raw, _ := tableFile.ReadFile("chktbl.hex")
	v, e := hex.DecodeString(strings.TrimSpace(string(raw)))
	if e != nil || len(v) != 960 {
		panic("invalid command checksum table")
	}
	copy(checkTable[:], v)
}

type UserCmd struct {
	Pitch, Yaw, Roll, Forward, Side, Up int16
	Buttons, Impulse, Msec, Light       byte
}

func deltaCmd(a, b UserCmd) []byte {
	out := []byte{0}
	fields := [][2]int16{{a.Pitch, b.Pitch}, {a.Yaw, b.Yaw}, {a.Roll, b.Roll}, {a.Forward, b.Forward}, {a.Side, b.Side}, {a.Up, b.Up}}
	for i, p := range fields {
		if p[0] != p[1] {
			out[0] |= 1 << i
			out = binary.LittleEndian.AppendUint16(out, uint16(p[1]))
		}
	}
	if a.Buttons != b.Buttons {
		out[0] |= 64
		out = append(out, b.Buttons)
	}
	if a.Impulse != b.Impulse {
		out[0] |= 128
		out = append(out, b.Impulse)
	}
	return append(out, b.Msec, b.Light)
}
func crcMove(data []byte, seq uint32) byte {
	n := min(len(data), 60)
	input := append([]byte{}, data[:n]...)
	offset := int(seq % 1020)
	input = append(input, checkTable[offset:offset+4]...)
	crc := uint16(0xffff)
	sum := 0
	for _, b := range input {
		sum += int(b)
		crc ^= uint16(b) << 8
		for i := 0; i < 8; i++ {
			if crc&0x8000 != 0 {
				crc = crc<<1 ^ 0x1021
			} else {
				crc <<= 1
			}
		}
	}
	return byte(int(crc) ^ sum)
}
func MovePacket(cmd, prev UserCmd, seq uint32) []byte {
	out := []byte{2, 0}
	out = binary.LittleEndian.AppendUint32(out, ^uint32(0))
	out = append(out, deltaCmd(UserCmd{}, prev)...)
	out = append(out, deltaCmd(prev, prev)...)
	out = append(out, deltaCmd(prev, cmd)...)
	out[1] = crcMove(out[2:], seq)
	return out
}

func ConnectRequest(qport uint16, challenge int, name string) string {
	userinfo := fmt.Sprintf(`\name\%s\skin\male/grunt\rate\25000\msg\1\hand\2\fov\90`, name)
	return fmt.Sprintf("connect 34 %d %d \"%s\"\n", qport, challenge, userinfo)
}
