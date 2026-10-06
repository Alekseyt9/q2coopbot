package quake

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// DemoRecorder writes protocol-34 client demos from the bot's signon and
// ordered server messages. UDP/netchannel headers never enter the file.
type DemoRecorder struct {
	Path       string
	file       *os.File
	signon     [][]byte
	generation int
	active     bool
}

func (d *DemoRecorder) Reset() error {
	err := d.Close()
	d.signon = nil
	d.active = false
	return err
}

func (d *DemoRecorder) Packet(decoder *Decoder, frames []Frame) error {
	if decoder.ServerdataSeen {
		if err := d.Reset(); err != nil {
			return err
		}
		d.active = true
	}
	if !d.active {
		return nil
	}
	payload := decoder.DemoPayload
	if d.file == nil {
		if len(frames) == 0 {
			if len(payload) > 0 {
				d.signon = append(d.signon, append([]byte(nil), payload...))
			}
			return nil
		}
		if frames[0].DeltaFrame > 0 {
			return nil
		}
		if err := d.open(); err != nil {
			return err
		}
		for _, message := range d.signon {
			if err := d.write(message); err != nil {
				return err
			}
		}
		d.signon = nil
		if err := d.write(append([]byte{11}, []byte("precache\n\x00")...)); err != nil {
			return err
		}
	}
	if len(payload) == 0 {
		return nil
	}
	return d.write(payload)
}

func (d *DemoRecorder) open() error {
	if err := os.MkdirAll(filepath.Dir(d.Path), 0755); err != nil {
		return err
	}
	ext := filepath.Ext(d.Path)
	stem := strings.TrimSuffix(d.Path, ext)
	for {
		d.generation++
		path := d.Path
		if d.generation > 1 {
			path = fmt.Sprintf("%s-%03d%s", stem, d.generation, ext)
		}
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
		if errors.Is(err, os.ErrExist) {
			continue
		}
		if err != nil {
			return err
		}
		d.file = f
		log.Printf("recording demo: %s", path)
		return nil
	}
}

func (d *DemoRecorder) write(payload []byte) error {
	var header [4]byte
	binary.LittleEndian.PutUint32(header[:], uint32(len(payload)))
	if _, err := d.file.Write(header[:]); err != nil {
		return err
	}
	_, err := d.file.Write(payload)
	return err
}

func (d *DemoRecorder) Close() error {
	if d.file == nil {
		return nil
	}
	_, writeErr := d.file.Write([]byte{255, 255, 255, 255})
	closeErr := d.file.Close()
	d.file = nil
	return errors.Join(writeErr, closeErr)
}
