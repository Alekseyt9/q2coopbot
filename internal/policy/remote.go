package policy

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"time"
)

const BridgeVersion = "combat_bridge_v1"
const RemoteKind = "combat_remote_v1"
const MaxBridgeMessage = 256 * 1024

type RemoteConfig struct {
	Kind          string `json:"kind"`
	Address       string `json:"address"`
	TimeoutMS     int    `json:"timeout_ms"`
	PolicyVersion string `json:"policy_version"`
	Episode       string `json:"episode"`
	Seed          int    `json:"seed"`
}

type BridgeRequest struct {
	Kind          string          `json:"kind,omitempty"`
	Feedback      json.RawMessage `json:"feedback,omitempty"`
	Version       string          `json:"version"`
	Session       string          `json:"session"`
	Request       uint64          `json:"request"`
	Episode       string          `json:"episode"`
	Seed          int             `json:"seed"`
	PolicyVersion string          `json:"policy_version"`
	Observation   Observation     `json:"observation"`
}

type BridgeResponse struct {
	FeedbackAck   bool   `json:"feedback_ack,omitempty"`
	Version       string `json:"version"`
	Session       string `json:"session"`
	Request       uint64 `json:"request"`
	PolicyVersion string `json:"policy_version"`
	Action        Action `json:"action"`
}

// ReadBridgeMessage bounds each newline-delimited message and rejects trailing
// JSON or unknown fields. Keep one reader per connection (it may buffer ahead).
func ReadBridgeMessage(r *bufio.Reader, dst any) error {
	var data []byte
	for {
		part, err := r.ReadSlice('\n')
		if len(data)+len(part) > MaxBridgeMessage {
			return fmt.Errorf("bridge message too large")
		}
		data = append(data, part...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err != nil {
			return err
		}
		break
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing bridge data")
	}
	return nil
}

func WriteBridgeMessage(w io.Writer, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(data)+1 > MaxBridgeMessage {
		return fmt.Errorf("bridge message too large")
	}
	data = append(data, '\n')
	for len(data) > 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}

func ValidateBridgeAddress(address string) error {
	host, port, err := net.SplitHostPort(address)
	if err != nil || host != "127.0.0.1" {
		return fmt.Errorf("bridge requires literal IPv4 loopback and port")
	}
	p, err := strconv.Atoi(port)
	if err != nil || p < 1024 || p > 65535 {
		return fmt.Errorf("invalid bridge port")
	}
	return nil
}

func (c RemoteConfig) Validate() error {
	if err := ValidateBridgeAddress(c.Address); err != nil {
		return err
	}
	if c.Kind != RemoteKind || c.TimeoutMS < 10 || c.TimeoutMS > 2000 || c.PolicyVersion == "" || len(c.PolicyVersion) > 128 || c.Episode == "" || len(c.Episode) > 128 || c.Seed < 0 || int64(c.Seed) > 2147483647 {
		return fmt.Errorf("invalid remote policy configuration")
	}
	return nil
}

// Remote is only permitted with the native synchronous test barrier. It has
// no access to native damage logs; observation is the entire policy input.
type Remote struct {
	config  RemoteConfig
	version string
	session string
	request uint64
	conn    net.Conn
	reader  *bufio.Reader
}

func LoadProvider(path string, synchronous bool) (Provider, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) > MaxModelBytes {
		return nil, fmt.Errorf("provider config too large")
	}
	var header struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}
	if header.Kind == MLPKind {
		return LoadMLP(path)
	}
	if len(data) > 65536 {
		return nil, fmt.Errorf("provider config too large")
	}
	if header.Kind != RemoteKind {
		return LoadProbe(path)
	}
	if !synchronous {
		return nil, fmt.Errorf("remote provider requires synchronous native pilot")
	}
	var config RemoteConfig
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&config); err != nil {
		return nil, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("trailing remote config data")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	hash := sha256.Sum256(data)
	return &Remote{config: config, session: hex.EncodeToString(nonce[:]), version: "remote:" + config.PolicyVersion + ":" + hex.EncodeToString(hash[:])}, nil
}

func (p *Remote) Version() string { return p.version }
func (p *Remote) DecisionBudget() time.Duration {
	return time.Duration(p.config.TimeoutMS) * time.Millisecond
}
func (p *Remote) Close() error {
	if p.conn == nil {
		return nil
	}
	err := p.conn.Close()
	p.conn = nil
	p.reader = nil
	return err
}

func (p *Remote) Decide(o Observation) (action Action, err error) {
	if o.Version != ObservationVersion || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		return action, fmt.Errorf("invalid remote observation")
	}
	r := BridgeRequest{Observation: o}
	reply, err := p.exchange(r)
	if err != nil {
		return action, err
	}
	if reply.FeedbackAck || reply.Action.Version != ActionVersion || reply.Action.Identity != o.Identity {
		p.Close()
		return action, fmt.Errorf("remote reply identity/version mismatch")
	}
	return reply.Action, nil
}

// Notify is a separate learning channel. No feedback is passed to Decide.
func (p *Remote) Notify(value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	reply, err := p.exchange(BridgeRequest{Kind: "feedback", Feedback: data})
	if err != nil {
		return err
	}
	if !reply.FeedbackAck {
		p.Close()
		return fmt.Errorf("feedback acknowledgement missing")
	}
	return nil
}

func (p *Remote) exchange(r BridgeRequest) (reply BridgeResponse, err error) {
	deadline := time.Now().Add(p.DecisionBudget())
	defer func() {
		if err != nil {
			p.Close()
		}
	}()
	if p.conn == nil {
		p.conn, err = net.DialTimeout("tcp4", p.config.Address, time.Until(deadline))
		if err != nil {
			return reply, err
		}
		p.reader = bufio.NewReader(p.conn)
	}
	if err = p.conn.SetDeadline(deadline); err != nil {
		return reply, err
	}
	p.request++
	r.Version, r.Session, r.Request, r.Episode, r.Seed, r.PolicyVersion = BridgeVersion, p.session, p.request, p.config.Episode, p.config.Seed, p.config.PolicyVersion
	if err = WriteBridgeMessage(p.conn, r); err != nil {
		return reply, err
	}
	if err = ReadBridgeMessage(p.reader, &reply); err != nil {
		return reply, err
	}
	if reply.Version != BridgeVersion || reply.Session != p.session || reply.Request != p.request || reply.PolicyVersion != p.config.PolicyVersion {
		return reply, fmt.Errorf("remote reply identity/version mismatch")
	}
	return reply, nil
}
