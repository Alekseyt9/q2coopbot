#!/usr/bin/env python3
"""Drive a Quake II server with a small real-time non-bot client.

This intentionally stops below the rendering/UI layer.  It proves the server
accepts a real network client slot and can process string commands and checked
clc_move usercmds when a graphical Yamagi client cannot be started.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import socket
import struct
import sys
import time
from typing import NamedTuple
from pathlib import Path
from urllib.parse import urlsplit

from openjev_controller import MODEL as DEFAULT_OPENJEV_MODEL, OpenJevController
from q2_packet_snapshot import PacketError, PacketSnapshots


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
BUTTON_USE = 2


class MovePhase(NamedTuple):
    duration: float
    forward_speed: int
    side_speed: int
    yaw_rate: float
    attack: bool
    jump: bool


class Waypoint(NamedTuple):
    x: float
    y: float
    z: float
    jump: bool


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
                    (record.get("event") in ("bot_snapshot", "openjev_snapshot")
                     and "player_is_human=1" in message)
                    or record.get("event") == "udp_snapshot"
                ):
                    return True, offset
    except OSError:
        pass
    return False, offset


def _human_origin(
    path: Path,
    offset: int,
    episode_id: str | None,
    previous: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float] | None, int]:
    """Read the newest live human origin without consuming a partial JSON line."""
    origin = previous
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            for raw_line in stream:
                if not raw_line.endswith(b"\n"):
                    break
                offset += len(raw_line)
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if episode_id and record.get("episode_id") != episode_id:
                    continue
                if record.get("event") == "udp_snapshot":
                    value = record.get("snapshot", {}).get("player_origin")
                    if isinstance(value, list) and len(value) == 3:
                        origin = tuple(float(part) for part in value)
                    continue
                if record.get("event") not in ("bot_snapshot", "openjev_snapshot"):
                    continue
                message = str(record.get("message", ""))
                if "player_is_human=1" not in message:
                    continue
                match = re.search(
                    r"player_origin=\((-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\)",
                    message,
                )
                if match:
                    origin = tuple(float(value) for value in match.groups())
    except OSError:
        pass
    return origin, offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27910)
    parser.add_argument("--protocol", type=int, default=34)
    parser.add_argument("--name", default="CoopHarness")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--qport", type=int, help="client qport; random by default")
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--udp-snapshot-log", type=Path,
                        help="write world snapshots decoded from standard server UDP packets")
    parser.add_argument("--episode-id")
    parser.add_argument("--openjev", action="store_true",
                        help="let local Ollama OpenJev steer this UDP test player")
    parser.add_argument("--openjev-model", default=DEFAULT_OPENJEV_MODEL)
    parser.add_argument("--openjev-url", default="http://127.0.0.1:11434")
    parser.add_argument("--openjev-interval", type=float, default=1.0)
    parser.add_argument("--openjev-timeout", type=float, default=15.0)
    parser.add_argument("--openjev-trace", type=Path)
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
        "--server-command-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="wait this long after begin before sending --server-command values",
    )
    parser.add_argument(
        "--server-command-at",
        action="append",
        default=[],
        metavar="SECONDS:COMMAND",
        help="send one command at a relative time after begin; may be repeated",
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
    parser.add_argument(
        "--use",
        action="store_true",
        help="hold the use button in movement usercmds",
    )
    parser.add_argument(
        "--phase",
        action="append",
        metavar="SECONDS:FORWARD:SIDE:YAW:ATTACK:JUMP",
        help=(
            "append a real-time movement phase; ATTACK and JUMP are 0 or 1; "
            "may be repeated to model advance/retreat/hold"
        ),
    )
    parser.add_argument(
        "--waypoint",
        action="append",
        metavar="X:Y:Z[:JUMP]",
        help=(
            "closed-loop live route waypoint read from bot_snapshot player_origin; "
            "JUMP is 0 or 1 and may be repeated"
        ),
    )
    parser.add_argument(
        "--post-move-duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="continue sending live usercmds after a route or phase completes",
    )
    parser.add_argument(
        "--waypoint-horizontal-tolerance",
        type=float,
        default=56.0,
        metavar="UNITS",
        help="horizontal distance required to mark a waypoint reached",
    )
    parser.add_argument(
        "--waypoint-vertical-tolerance",
        type=float,
        default=96.0,
        metavar="UNITS",
        help="vertical distance required to mark a waypoint reached",
    )
    args = parser.parse_args()
    if args.server_command_delay < 0.0:
        parser.error("--server-command-delay must be non-negative")
    scheduled_commands: list[tuple[float, str]] = [
        (args.server_command_delay, command) for command in args.server_command
    ]
    for specification in args.server_command_at:
        separator = specification.find(":")
        if separator <= 0 or separator == len(specification) - 1:
            parser.error("--server-command-at must use SECONDS:COMMAND")
        try:
            command_time = float(specification[:separator])
        except ValueError:
            parser.error("--server-command-at seconds must be numeric")
        if command_time < 0.0:
            parser.error("--server-command-at seconds must be non-negative")
        scheduled_commands.append((command_time, specification[separator + 1:]))
    scheduled_commands.sort(key=lambda item: item[0])

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.post_move_duration < 0.0:
        parser.error("--post-move-duration must be non-negative")
    if args.waypoint_horizontal_tolerance <= 0.0:
        parser.error("--waypoint-horizontal-tolerance must be positive")
    if args.waypoint_vertical_tolerance <= 0.0:
        parser.error("--waypoint-vertical-tolerance must be positive")
    if args.move_forward < 0:
        parser.error("--move-forward must not be negative")
    if args.phase and args.move_forward > 0:
        parser.error("use either --move-forward or --phase, not both")
    if args.waypoint and (args.phase or args.move_forward > 0):
        parser.error("use either --waypoint or movement phases, not both")
    if args.waypoint and args.event_log is None:
        parser.error("--waypoint requires --event-log for live player telemetry")
    if args.openjev and ((args.event_log is None and args.udp_snapshot_log is None)
                         or args.phase or args.waypoint or args.move_forward > 0):
        parser.error("--openjev requires a snapshot log and cannot be combined with movement phases or waypoints")
    if args.openjev_interval <= 0 or args.openjev_timeout <= 0:
        parser.error("OpenJev interval and timeout must be positive")
    openjev_endpoint = urlsplit(args.openjev_url)
    if args.openjev and (openjev_endpoint.scheme != "http"
                         or openjev_endpoint.hostname != "127.0.0.1"
                         or not openjev_endpoint.port
                         or openjev_endpoint.path not in ("", "/")):
        parser.error("--openjev-url must use local Ollama on 127.0.0.1")

    phases: list[MovePhase] = []
    for raw_phase in args.phase or []:
        parts = raw_phase.split(":")
        if len(parts) != 6:
            parser.error(
                "--phase must be SECONDS:FORWARD:SIDE:YAW:ATTACK:JUMP"
            )
        try:
            duration = float(parts[0])
            forward_speed = int(parts[1])
            side_speed = int(parts[2])
            yaw_rate = float(parts[3])
            attack = int(parts[4])
            jump = int(parts[5])
        except ValueError:
            parser.error(f"invalid --phase value: {raw_phase}")
        if duration <= 0:
            parser.error("--phase duration must be positive")
        if attack not in (0, 1) or jump not in (0, 1):
            parser.error("--phase ATTACK and JUMP must be 0 or 1")
        phases.append(
            MovePhase(
                duration,
                forward_speed,
                side_speed,
                yaw_rate,
                bool(attack),
                bool(jump),
            )
        )
    waypoints: list[Waypoint] = []
    for raw_waypoint in args.waypoint or []:
        parts = raw_waypoint.split(":")
        if len(parts) not in (3, 4):
            parser.error("--waypoint must be X:Y:Z[:JUMP]")
        try:
            x, y, z = (float(value) for value in parts[:3])
            jump = int(parts[3]) if len(parts) == 4 else 0
        except ValueError:
            parser.error(f"invalid --waypoint value: {raw_waypoint}")
        if jump not in (0, 1):
            parser.error("--waypoint JUMP must be 0 or 1")
        waypoints.append(Waypoint(x, y, z, bool(jump)))
    if not phases and args.move_forward > 0:
        phases.append(
            MovePhase(
                args.move_forward,
                args.forward_speed,
                args.side_speed,
                args.yaw_rate,
                args.attack,
                args.jump,
            )
        )
    phase_duration = sum(phase.duration for phase in phases)
    if phases and args.duration < phase_duration:
        parser.error(
            f"--duration ({args.duration}) must cover all phases ({phase_duration})"
        )
    if waypoints and args.duration <= 0:
        parser.error("--waypoint route requires a positive --duration")

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
    command_start_at: float | None = None
    move_packets = 0
    openjev_attack_packets = 0
    move_started: float | None = None
    next_move_at = 0.0
    phase_index = 0
    phase_yaw_degrees = 0.0
    waypoint_index = 0
    route_completed_at: float | None = None
    latest_human_origin: tuple[float, float, float] | None = None
    origin_log_offset = log_offset
    check_table: bytes | None = None
    socket_error: str | None = None
    zero_cmd = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    previous_cmd = zero_cmd
    if phases or waypoints or args.openjev:
        try:
            check_table = _load_check_table()
        except RuntimeError as exc:
            parser.error(str(exc))
    next_sequence = 1
    started = time.monotonic()
    snapshot_path = args.udp_snapshot_log or args.event_log
    snapshot_offset = snapshot_path.stat().st_size if snapshot_path and snapshot_path.exists() else 0
    decoder = PacketSnapshots() if args.udp_snapshot_log else None
    decoded_frames = 0
    seen_server_commands: set[str] = set()
    openjev = (OpenJevController(snapshot_path, args.episode_id, snapshot_offset,
                                args.openjev_url, args.openjev_model,
                                args.openjev_interval, args.openjev_timeout,
                                args.openjev_trace) if args.openjev else None)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("0.0.0.0", 0))
        sock.settimeout(0.05 if phases or waypoints or openjev else 0.25)
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
                    if decoder is not None:
                        try:
                            frames = decoder.parse(packet[8:])
                        except (PacketError, ValueError) as exc:
                            decoder.errors += 1
                            decoder.last_error = str(exc)
                            frames = []
                        for frame in frames:
                            snapshot = decoder.snapshot(frame)
                            if snapshot is None:
                                continue
                            decoded_frames += 1
                            human_marker = True
                            latest_human_origin = tuple(snapshot["player_origin"])
                            with args.udp_snapshot_log.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps({"event": "udp_snapshot",
                                    "episode_id": args.episode_id,
                                    "time": snapshot["time"], "map": snapshot["map"],
                                    "snapshot": snapshot}, ensure_ascii=False) + "\n")
                        for requested in decoder.server_commands:
                            if requested in seen_server_commands:
                                continue
                            seen_server_commands.add(requested)
                            if requested.startswith("cmd configstrings ") or requested.startswith("cmd baselines "):
                                command = requested[4:]
                            elif requested.startswith("precache "):
                                command = "begin " + requested.split()[1]
                                begin_sent = True
                                command_start_at = time.monotonic()
                            else:
                                continue
                            _send_netchan_command(sock, address, next_sequence, qport,
                                                  command, server_sequence, server_reliable)
                            next_sequence += 1

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

            if decoder is None and new_sent and spawncount is not None and not begin_sent:
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
                command_start_at = time.monotonic()

            if (
                begin_sent
                and not commands_sent
                and command_start_at is not None
            ):
                elapsed = time.monotonic() - command_start_at
                due_count = 0
                for command_time, command in scheduled_commands:
                    if command_time > elapsed:
                        break
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
                    due_count += 1
                if due_count > 0:
                    scheduled_commands = scheduled_commands[due_count:]
                commands_sent = len(scheduled_commands) == 0

            if begin_sent and phases:
                now = time.monotonic()
                if move_started is None:
                    move_started = now
                    next_move_at = now
                movement_elapsed = now - move_started
                if movement_elapsed < phase_duration and check_table is not None:
                    while now >= next_move_at:
                        while (
                            phase_index + 1 < len(phases)
                            and movement_elapsed
                            >= sum(phase.duration for phase in phases[: phase_index + 1])
                        ):
                            phase_index += 1
                        phase = phases[phase_index]
                        phase_elapsed = movement_elapsed - sum(
                            item.duration for item in phases[:phase_index]
                        )
                        phase_yaw_degrees += phase.yaw_rate * 0.05
                        yaw = int(phase_yaw_degrees * 65536.0 / 360.0)
                        yaw = ((yaw + 32768) % 65536) - 32768
                        current_cmd = (
                            0,
                            yaw,
                            0,
                            phase.forward_speed,
                            phase.side_speed,
                            200 if phase.jump else 0,
                            (BUTTON_ATTACK if phase.attack else 0)
                            | (BUTTON_USE if args.use else 0),
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

            if begin_sent and waypoints and check_table is not None:
                now = time.monotonic()
                if move_started is None:
                    move_started = now
                    next_move_at = now
                while now >= next_move_at:
                    if latest_human_origin is not None:
                        while waypoint_index < len(waypoints):
                            waypoint = waypoints[waypoint_index]
                            dx = waypoint.x - latest_human_origin[0]
                            dy = waypoint.y - latest_human_origin[1]
                            dz = waypoint.z - latest_human_origin[2]
                            if (
                                math.hypot(dx, dy)
                                > args.waypoint_horizontal_tolerance
                                or abs(dz) > args.waypoint_vertical_tolerance
                            ):
                                break
                            waypoint_index += 1
                        if waypoint_index >= len(waypoints) and route_completed_at is None:
                            route_completed_at = now
                        if waypoint_index < len(waypoints):
                            waypoint = waypoints[waypoint_index]
                            dx = waypoint.x - latest_human_origin[0]
                            dy = waypoint.y - latest_human_origin[1]
                            dz = waypoint.z - latest_human_origin[2]
                            # The live spawn view points along -X; network yaw
                            # 180 points along +X. Convert a world vector to
                            # that calibrated Quake II angle convention.
                            yaw_degrees = math.degrees(math.atan2(-dy, -dx)) % 360.0
                            yaw = int(yaw_degrees * 65536.0 / 360.0)
                            current_cmd = (
                                0,
                                ((yaw + 32768) % 65536) - 32768,
                                0,
                                400,
                                0,
                                200 if waypoint.jump or dz > 32.0 else 0,
                                BUTTON_USE if args.use else 0,
                                0,
                                50,
                                0,
                            )
                        else:
                            current_cmd = previous_cmd[:3] + (0, 0, 0, 0, 0, 50, 0)
                    else:
                        current_cmd = previous_cmd[:3] + (0, 0, 0, 0, 0, 50, 0)
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

            if begin_sent and openjev is not None and check_table is not None:
                now = time.monotonic()
                if move_started is None:
                    move_started = now
                    next_move_at = now
                openjev.tick()
                while now >= next_move_at:
                    current_cmd = openjev.command(previous_cmd[1], args.use)
                    payload = _move_payload(current_cmd, previous_cmd,
                                            next_sequence, check_table)
                    _send_netchan_payload(sock, address, next_sequence, qport,
                                          payload, server_sequence, server_reliable,
                                          reliable=False)
                    next_sequence += 1
                    move_packets += 1
                    if current_cmd[6] & BUTTON_ATTACK:
                        openjev_attack_packets += 1
                    previous_cmd = current_cmd
                    next_move_at += 0.05

            if args.event_log:
                log_marker, log_offset = _human_marker_seen(
                    args.event_log, log_offset, args.episode_id
                )
                human_marker = human_marker or log_marker
                if waypoints:
                    latest_human_origin, origin_log_offset = _human_origin(
                        args.event_log,
                        origin_log_offset,
                        args.episode_id,
                        latest_human_origin,
                    )
            if openjev or decoder is not None:
                movement_complete = False
            elif waypoints:
                movement_complete = (
                    route_completed_at is not None
                    and time.monotonic() - route_completed_at
                    >= args.post_move_duration
                )
            else:
                movement_complete = not phases or (
                    move_started is not None
                    and time.monotonic() - move_started
                    >= phase_duration + args.post_move_duration
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
    if openjev:
        openjev.close()

    result = {
        "host": args.host,
        "port": args.port,
        "qport": qport,
        "challenge_received": challenge is not None,
        "client_connect_received": client_connected,
        "netchan_packets": netchan_packets,
        "decoded_frames": decoded_frames,
        "udp_decode_errors": decoder.errors if decoder else 0,
        "udp_decode_last_error": decoder.last_error if decoder else None,
        "udp_configstrings": len(decoder.config) if decoder else 0,
        "udp_map": decoder.map_name if decoder else None,
        "udp_player_number": decoder.player_number if decoder else None,
        "udp_parsed_frames": len(decoder.frames) if decoder else 0,
        "spawncount": spawncount,
        "begin_sent": begin_sent,
        "commands_sent": commands_sent,
        "move_packets": move_packets,
        "phase_count": len(phases),
        "phase_duration": phase_duration,
        "waypoint_count": len(waypoints),
        "waypoint_reached": waypoint_index,
        "human_marker_seen": human_marker,
        "openjev_model": args.openjev_model if openjev else None,
        "openjev_decisions": openjev.decisions if openjev else 0,
        "openjev_attack_packets": openjev_attack_packets,
        "openjev_last_action": openjev.action if openjev else None,
        "openjev_errors": openjev.errors if openjev else [],
        "socket_error": socket_error,
        "responses": responses,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    if args.require_human_marker:
        return 0 if human_marker and (not openjev or openjev.decisions > 0) else 2
    return 0 if client_connected and (not openjev or openjev.decisions > 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
