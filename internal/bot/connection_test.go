package bot

import (
	"net"
	"testing"
)

func TestConnectionCountOnlyChangesOnNewAcceptance(t *testing.T) {
	sink, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()
	source, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	c := Client{conn: source, address: sink.LocalAddr().(*net.UDPAddr), seq: 1}
	packet := append([]byte{255, 255, 255, 255}, []byte("client_connect")...)
	c.handle(packet)
	c.handle(packet)
	if c.connection != 1 {
		t.Fatal(c.connection)
	}
	c.connected = false
	c.handle(packet)
	if c.connection != 2 {
		t.Fatal(c.connection)
	}
}
