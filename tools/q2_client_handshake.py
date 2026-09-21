#!/usr/bin/env python3
"""Drive a Quake II server with a small real-time non-bot client.

This intentionally stops below the rendering/UI layer.  It proves the server
accepts a real network client slot and can process string commands and checked
clc_move usercmds when a graphical Yamagi client cannot be started.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import struct
import sys
import time
from pathlib import Path


OOB = struct.pack("<I", 0xFFFFFFFF)
CHALLENGE_RE = re.compile(r"(?:^|\s)challenge\s+(-?\d+)(?:\s|$)")
CM_ANGLE1 = 1 << 0
CM_ANGLE2 = 1 << 1
CM_ANGLE3 = 1 << 2
CM_FORWARD = 1 << 3
CM_SIDE = 1 << 4
CM_UP = 1 << 5
CM_BUTTONS = 1 << 6
CM_IMPULSE = 1 << 7
BUTTON_ATTACK = 1


def _packet_text(packet: bytes) -> str:
    return packet[4:].decode("latin-1", errors="replace")


def _send_netchan_payload(
    sock: socket.socket,
    address: tuple[str, int],
    sequence: int,
    qport: int,
    payload: bytes,
    acknowledged_sequence: int,
    acknowledged_reliable: int,
    reliable: bool,
) -> None:
    # Quake II's client-to-server netchan header is two little-endian sequence
    # longs followed by the client's qport.
    header = struct.pack(
        "<IIH",
        sequence | (0x80000000 if reliable else 0),
        acknowledged_sequence | (acknowledged_reliable << 31),
        qport,
    )
    sock.sendto(header + payload, address)


def _send_netchan_command(
    sock: socket.socket,
    address: tuple[str, int],
    sequence: int,
    qport: int,
    command: str,
    acknowledged_sequence: int,
    acknowledged_reliable: int,
) -> None:
    # clc_stringcmd is opcode 4 and is carried reliably by the stock client.
    _send_netchan_payload(
        sock,
        address,
        sequence,
        qport,
        bytes((4,)) + command.encode("latin-1") + b"\x00",
        acknowledged_sequence,
        acknowledged_reliable,
        reliable=True,
    )


def _write_delta_usercmd(
    source: tuple[int, int, int, int, int, int, int, int, int, int],
    target: tuple[int, int, int, int, int, int, int, int, int, int],
) -> bytes:
    """Encode MSG_WriteDeltaUsercmd for the fields used by Quake II."""
    bits = 0
    if target[0] != source[0]:
        bits |= CM_ANGLE1
    if target[1] != source[1]:
        bits |= CM_ANGLE2
    if target[2] != source[2]:
        bits |= CM_ANGLE3
    if target[3] != source[3]:
        bits |= CM_FORWARD
    if target[4] != source[4]:
        bits |= CM_SIDE
    if target[5] != source[5]:
        bits |= CM_UP
    if target[6] != source[6]:
        bits |= CM_BUTTONS
    if target[7] != source[7]:
        bits |= CM_IMPULSE

    payload = bytearray((bits,))
    if bits & CM_ANGLE1:
        payload.extend(struct.pack("<h", target[0]))
    if bits & CM_ANGLE2:
        payload.extend(struct.pack("<h", target[1]))
    if bits & CM_ANGLE3:
        payload.extend(struct.pack("<h", target[2]))
    if bits & CM_FORWARD:
        payload.extend(struct.pack("<h", target[3]))
    if bits & CM_SIDE:
        payload.extend(struct.pack("<h", target[4]))
    if bits & CM_UP:
        payload.extend(struct.pack("<h", target[5]))
    if bits & CM_BUTTONS:
        payload.append(target[6] & 0xFF)
    if bits & CM_IMPULSE:
        payload.append(target[7] & 0xFF)
    payload.extend((target[8] & 0xFF, target[9] & 0xFF))
    return bytes(payload)


def _load_check_table() -> bytes:
    """Load id Software's fixed 1024-byte command-check table."""
    candidates = (
        Path(__file__).resolve().parents[2] / "yquake2-upstream/src/common/crc.c",
        Path(__file__).resolve().parents[2] / "yquake2/src/common/crc.c",
    )
    for candidate in candidates:
        try:
            source = candidate.read_text(encoding="latin-1")
        except OSError:
            continue
        start = source.find("static byte chktbl[1024]")
        if start < 0:
            continue
        end = source.find("};", start)
        if end < 0:
            continue
        table_source = source[start:end]
        values = bytes(
            int(value, 16)
            for value in re.findall(r"0x([0-9a-fA-F]{2})", table_source)
        )
        if 0 < len(values) <= 1024:
            # The legacy source declares 1024 bytes but explicitly lists 960;
            # the remaining static-storage bytes are zero-initialized by C.
            return values.ljust(1024, b"\x00")
    raise RuntimeError(
        "could not find Quake II chktbl[1024]; keep yquake2-upstream or "
        "yquake2 next to q2coopbot-release"
    )


def _sequence_crc(payload: bytes, sequence: int, check_table: bytes) -> int:
    check = check_table[sequence % (len(check_table) - 4) :][:4]
    data = payload[:60] + check
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return (crc ^ sum(data)) & 0xFF


def _move_payload(
    command: tuple[int, int, int, int, int, int, int, int, int, int],
    previous: tuple[int, int, int, int, int, int, int, int, int, int],
    sequence: int,
    check_table: bytes,
) -> bytes:
    # clc_move, checksum placeholder, no-delta lastframe, then the same three
    # delta usercmds the stock client sends for loss recovery.
    body = bytearray((2, 0))
    body.extend(struct.pack("<i", -1))
    body.extend(_write_delta_usercmd((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), previous))
    body.extend(_write_delta_usercmd(previous, previous))
    body.extend(_write_delta_usercmd(previous, command))
    body[1] = _sequence_crc(bytes(body[2:]), sequence, check_table)
    return bytes(body)


def _read_server_packet(packet: bytes, reliable_state: int) -> tuple[int, int, int | None]:
    """Return sequence, updated reliable parity, and serverdata spawncount."""
    # Server-to-client packets contain only the two sequence longs.  The
    # client qport is present in the opposite direction (client -> server).
    if len(packet) < 8:
        return 0, reliable_state, None
    first, _ack = struct.unpack_from("<II", packet, 0)
    sequence = first & 0x7FFFFFFF
    if first & 0x80000000:
        reliable_state ^= 1
    payload = packet[8:]
    # svc_serverdata is opcode 12 in the Quake II wire protocol.  Its first
    # fields are opcode, protocol (long), and spawncount (long).
    if payload and payload[0] == 12 and len(payload) >= 9:
        spawncount = struct.unpack_from("<i", payload, 5)[0]
        return sequence, reliable_state, spawncount
    return sequence, reliable_state, None


def _human_marker_seen(path: Path, offset: int, episode_id: str | None) -> tuple[bool, int]:
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            for raw_line in stream:
                # Do not consume a line that is still being appended by the
                # game-side logger; otherwise a transient partial JSON record
                # could hide the human marker for the rest of the run.
                if not raw_line.endswith(b"\n"):
                    return False, offset
                offset += len(raw_line)
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if episode_id and record.get("episode_id") != episode_id:
                    continue
                message = str(record.get("message", ""))
                if (
                    record.get("event") == "bot_snapshot"
                    and "player_is_human=1" in message
                ):
                    return True, offset
    except OSError:
        pass
    return False, offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27910)
    parser.add_argument("--protocol", type=int, default=34)
    parser.add_argument("--name", default="CoopHarness")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--qport", type=int, help="client qport; random by default")
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument(
        "--require-human-marker",
        action="store_true",
        help="succeed only after telemetry records player_is_human=1",
    )
    parser.add_argument(
        "--server-command",
        action="append",
        default=[],
        help=(
            "send a player console command after begin; may be repeated "
            "(for example: --server-command 'say hello')"
        ),
    )
    parser.add_argument(
        "--move-forward",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="send forward movement usercmds for this many seconds after begin",
    )
    parser.add_argument(
        "--forward-speed",
        type=int,
        default=400,
        help="forward usercmd value; use a negative value to retreat",
    )
    parser.add_argument(
        "--side-speed",
        type=int,
        default=0,
        help="side usercmd value; positive/negative values strafe",
    )
    parser.add_argument(
        "--yaw-rate",
        type=float,
        default=0.0,
        metavar="DEG_PER_SEC",
        help="continuous yaw rate while moving",
    )
    parser.add_argument(
        "--attack",
        action="store_true",
        help="hold the primary attack button in --move-forward usercmds",
    )
    parser.add_argument(
        "--jump",
        action="store_true",
        help="hold the jump/up input in --move-forward usercmds",
    )
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.move_forward < 0:
        parser.error("--move-forward must not be negative")

    address = (args.host, args.port)
    qport = args.qport if args.qport is not None else random.randrange(1, 65536)
    if qport < 1 or qport > 65535:
        parser.error("--qport must be between 1 and 65535")

    log_offset = 0
    if args.event_log:
        try:
            log_offset = args.event_log.stat().st_size
        except OSError:
            pass

    userinfo = (
        f"\\name\\{args.name}\\skin\\male/grunt\\rate\\25000"
        "\\msg\\1\\hand\\2\\fov\\90"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    challenge = None
    responses: list[str] = []
    client_connected = False
    human_marker = False
    netchan_packets = 0
    server_sequence = 0
    server_reliable = 0
    spawncount: int | None = None
    new_sent = False
    begin_sent = False
    commands_sent = False
    move_packets = 0
    move_started: float | None = None
    next_move_at = 0.0
    check_table: bytes | None = None
    socket_error: str | None = None
    zero_cmd = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    previous_cmd = zero_cmd
    move_cmd = (
        0,
        0,
        0,
        args.forward_speed,
        args.side_speed,
        200 if args.jump else 0,
        BUTTON_ATTACK if args.attack else 0,
        0,
        50,
        0,
    )
    if args.move_forward > 0:
        try:
            check_table = _load_check_table()
        except RuntimeError as exc:
            parser.error(str(exc))
    next_sequence = 1
    started = time.monotonic()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("0.0.0.0", 0))
        sock.settimeout(0.05 if args.move_forward > 0 else 0.25)
        sock.sendto(OOB + b"getchallenge\n", address)

        while time.monotonic() - started < args.duration:
            try:
                packet, source = sock.recvfrom(65535)
            except socket.timeout:
                packet = b""
                source = None
            except ConnectionResetError as exc:
                # Windows reports an ICMP/peer close on a connected UDP socket
                # as WSAECONNRESET.  Treat it as an observable transport end,
                # not as a harness traceback; the JSON result still exposes
                # whether the client had completed the game handshake.
                socket_error = str(exc)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) == 10054:
                    socket_error = str(exc)
                    break
                raise

            if packet:
                if packet[:4] == OOB:
                    text = _packet_text(packet)
                    responses.append(
                        text.strip()[:240].encode("ascii", "backslashreplace")
                        .decode("ascii")
                    )
                    match = CHALLENGE_RE.search(text)
                    if match and challenge is None:
                        challenge = int(match.group(1))
                        connect = (
                            f'connect {args.protocol} {qport} {challenge} '
                            f'"{userinfo}"\n'
                        ).encode("latin-1")
                        sock.sendto(OOB + connect, source or address)
                else:
                    netchan_packets += 1
                    server_sequence, server_reliable, packet_spawncount = (
                        _read_server_packet(packet, server_reliable)
                    )
                    if packet_spawncount is not None:
                        spawncount = packet_spawncount

            if challenge is not None and not client_connected:
                if any("client_connect" in response for response in responses):
                    client_connected = True
                    if not new_sent:
                        _send_netchan_command(
                            sock,
                            address,
                            next_sequence,
                            qport,
                            "new",
                            server_sequence,
                            server_reliable,
                        )
                        next_sequence += 1
                        new_sent = True

            if new_sent and spawncount is not None and not begin_sent:
                _send_netchan_command(
                    sock,
                    address,
                    next_sequence,
                    qport,
                    f"begin {spawncount}",
                    server_sequence,
                    server_reliable,
                )
                next_sequence += 1
                begin_sent = True

            if begin_sent and not commands_sent:
                for command in args.server_command:
                    _send_netchan_command(
                        sock,
                        address,
                        next_sequence,
                        qport,
                        command,
                        server_sequence,
                        server_reliable,
                    )
                    next_sequence += 1
                commands_sent = True

            if begin_sent and args.move_forward > 0:
                now = time.monotonic()
                if move_started is None:
                    move_started = now
                    next_move_at = now
                if now - move_started < args.move_forward and check_table is not None:
                    while now >= next_move_at:
                        elapsed = now - (move_started or now)
                        yaw = int(elapsed * args.yaw_rate * 65536.0 / 360.0)
                        yaw = ((yaw + 32768) % 65536) - 32768
                        current_cmd = (
                            0,
                            yaw,
                            0,
                            move_cmd[3],
                            move_cmd[4],
                            move_cmd[5],
                            move_cmd[6],
                            0,
                            50,
                            0,
                        )
                        payload = _move_payload(
                            current_cmd,
                            previous_cmd,
                            next_sequence,
                            check_table,
                        )
                        _send_netchan_payload(
                            sock,
                            address,
                            next_sequence,
                            qport,
                            payload,
                            server_sequence,
                            server_reliable,
                            reliable=False,
                        )
                        next_sequence += 1
                        move_packets += 1
                        previous_cmd = current_cmd
                        next_move_at += 0.05

            if args.event_log:
                human_marker, log_offset = _human_marker_seen(
                    args.event_log, log_offset, args.episode_id
                )
            movement_complete = args.move_forward <= 0 or (
                move_started is not None
                and time.monotonic() - move_started >= args.move_forward
            )
            if human_marker and args.require_human_marker and movement_complete:
                break
            if client_connected and not args.require_human_marker and movement_complete:
                break

    # The game logger can flush a final buffered JSONL record just after the
    # socket loop exits.  Re-scan once so the strict gate reflects the complete
    # episode rather than a race with the logger's last write.
    if args.event_log and not human_marker:
        human_marker, _ = _human_marker_seen(args.event_log, 0, args.episode_id)

    result = {
        "host": args.host,
        "port": args.port,
        "qport": qport,
        "challenge_received": challenge is not None,
        "client_connect_received": client_connected,
        "netchan_packets": netchan_packets,
        "spawncount": spawncount,
        "begin_sent": begin_sent,
        "commands_sent": commands_sent,
        "move_packets": move_packets,
        "human_marker_seen": human_marker,
        "socket_error": socket_error,
        "responses": responses,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    if args.require_human_marker:
        return 0 if human_marker else 2
    return 0 if client_connected else 2


if __name__ == "__main__":
    raise SystemExit(main())
