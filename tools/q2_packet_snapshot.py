"""Decode stock Quake II protocol 34 snapshots without a game DLL hook."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import struct

DEATH_FRAMES = {"soldier": ((272, 474),),
                "infantry": ((125, 178),)}


class PacketError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, size: int) -> bytes:
        if self.pos + size > len(self.data):
            raise PacketError("truncated server message")
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def number(self, fmt: str) -> int:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def byte(self) -> int:
        return self.number("<B")

    def short(self) -> int:
        return self.number("<h")

    def ushort(self) -> int:
        return self.number("<H")

    def long(self) -> int:
        return self.number("<i")

    def string(self) -> str:
        end = self.data.find(b"\0", self.pos)
        if end < 0:
            raise PacketError("unterminated server string")
        value = self.data[self.pos:end].decode("latin-1", "replace")
        self.pos = end + 1
        return value


@dataclass
class Entity:
    number: int
    model: int = 0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame: int = 0


@dataclass
class Frame:
    number: int
    origin: tuple[float, float, float]
    stats: list[int]
    gun: int
    delta_angles: tuple[int, int, int] = (0, 0, 0)
    entities: dict[int, Entity] = field(default_factory=dict)


class PacketSnapshots:
    """Maintain configstrings, baselines and the last 16 protocol frames."""

    def __init__(self) -> None:
        self.config: dict[int, str] = {}
        self.baselines: dict[int, Entity] = {}
        self.frames: dict[int, Frame] = {}
        self.player_number = 0
        self.map_name: str | None = None
        self.teammate_origin: tuple[float, float, float] | None = None
        self.errors = 0
        self.last_error: str | None = None
        self.server_commands: list[str] = []
        self.wall_impacts: list[tuple[float, float, float]] = []

    @staticmethod
    def _bits(reader: Reader) -> tuple[int, int]:
        bits = reader.byte()
        if bits & 0x80:
            bits |= reader.byte() << 8
        if bits & 0x8000:
            bits |= reader.byte() << 16
        if bits & 0x800000:
            bits |= reader.byte() << 24
        number = reader.ushort() if bits & 0x100 else reader.byte()
        return bits, number

    @staticmethod
    def _entity(reader: Reader, number: int, bits: int, old: Entity | None) -> Entity:
        model = old.model if old else 0
        frame = old.frame if old else 0
        origin = list(old.origin if old else (0.0, 0.0, 0.0))
        for bit in (0x800, 0x100000, 0x200000, 0x400000):
            if bits & bit:
                value = reader.byte()
                if bit == 0x800:
                    model = value
        if bits & 0x10:
            frame = reader.byte()
        if bits & 0x20000:
            frame = reader.ushort()
        if bits & 0x10000 and bits & 0x2000000:
            reader.take(4)
        elif bits & 0x10000 or bits & 0x2000000:
            reader.take(1 if bits & 0x10000 else 2)
        for low, high in ((0x4000, 0x80000), (0x1000, 0x40000)):
            if bits & low and bits & high:
                reader.take(4)
            elif bits & low or bits & high:
                reader.take(1 if bits & low else 2)
        for axis, bit in enumerate((1, 2, 0x200)):
            if bits & bit:
                origin[axis] = reader.short() / 8.0
        for bit in (0x400, 4, 8):
            if bits & bit:
                reader.byte()
        if bits & 0x1000000:
            reader.take(6)
        if bits & 0x4000000:
            reader.byte()
        if bits & 0x20:
            reader.byte()
        if bits & 0x8000000:
            reader.take(2)
        return Entity(number, model, tuple(origin), frame)

    def _playerstate(self, reader: Reader, old: Frame | None) -> tuple[tuple[float, float, float], list[int], int, tuple[int, int, int]]:
        flags = reader.ushort()
        origin = list(old.origin if old else (0.0, 0.0, 0.0))
        stats = list(old.stats if old else [0] * 32)
        gun = old.gun if old else 0
        delta_angles = old.delta_angles if old else (0, 0, 0)
        if flags & 1:
            reader.byte()
        if flags & 2:
            origin = [reader.short() / 8.0 for _ in range(3)]
        if flags & 4:
            reader.take(6)
        if flags & 8:
            reader.byte()
        if flags & 16:
            reader.byte()
        if flags & 32:
            reader.take(2)
        if flags & 64:
            delta_angles = tuple(reader.short() for _ in range(3))
        if flags & 128:
            reader.take(3)
        if flags & 256:
            reader.take(6)
        if flags & 512:
            reader.take(3)
        if flags & 4096:
            gun = reader.byte()
        if flags & 8192:
            reader.take(7)
        if flags & 1024:
            reader.take(4)
        if flags & 2048:
            reader.byte()
        if flags & 16384:
            reader.byte()
        statbits = reader.number("<I")
        for index in range(32):
            if statbits & (1 << index):
                stats[index] = reader.short()
        return tuple(origin), stats, gun, delta_angles

    def _frame(self, reader: Reader) -> Frame:
        number, delta = reader.long(), reader.long()
        reader.byte()  # suppress count
        old = self.frames.get(delta) if delta > 0 else None
        if delta > 0 and old is None:
            raise PacketError("missing delta frame")
        reader.take(reader.byte())  # areabits
        if reader.byte() != 17:
            raise PacketError("frame has no playerinfo")
        origin, stats, gun, delta_angles = self._playerstate(reader, old)
        if reader.byte() != 18:
            raise PacketError("frame has no packetentities")
        entities = dict(old.entities) if old else {}
        while True:
            bits, entity_number = self._bits(reader)
            if entity_number == 0:
                break
            if bits & 0x40:
                entities.pop(entity_number, None)
            else:
                entities[entity_number] = self._entity(reader, entity_number, bits,
                    entities.get(entity_number) if old and entity_number in old.entities
                    else self.baselines.get(entity_number))
        frame = Frame(number, origin, stats, gun, delta_angles, entities)
        self.frames[number] = frame
        for previous in tuple(self.frames):
            if number - previous > 16:
                del self.frames[previous]
        return frame

    def parse(self, payload: bytes) -> list[Frame]:
        reader = Reader(payload)
        found: list[Frame] = []
        self.server_commands.clear()
        while reader.pos < len(payload):
            opcode = reader.byte()
            if opcode == 6:  # nop
                continue
            if opcode == 12:
                self.config.clear()
                self.baselines.clear()
                self.frames.clear()
                self.map_name = None
                self.teammate_origin = None
                self.wall_impacts.clear()
                reader.long()  # protocol
                reader.long()  # spawncount
                reader.byte()  # attractloop
                reader.string()  # game directory
                self.player_number = reader.short() + 1
                reader.string()  # level title
            elif opcode == 13:
                index, value = reader.ushort(), reader.string()
                self.config[index] = value
                if index == 33:
                    match = re.search(r"(?:^|/)maps/([^/]+)\.bsp$", value)
                    self.map_name = match.group(1) if match else None
            elif opcode == 14:
                bits, number = self._bits(reader)
                self.baselines[number] = self._entity(reader, number, bits, None)
            elif opcode == 20:
                found.append(self._frame(reader))
            elif opcode in (10,):
                reader.byte()
                reader.string()
            elif opcode == 11:
                self.server_commands.append(reader.string().strip())
            elif opcode in (15, 4):
                reader.string()
            elif opcode in (1, 2):
                reader.take(3)
            elif opcode == 3:  # svc_temp_entity
                effect = reader.byte()
                if effect in (0, 1, 2, 4):  # gunshot, blood, blaster, shotgun
                    position = tuple(reader.short() / 8 for _ in range(3))
                    reader.byte()  # compressed normal
                    if effect == 2:
                        self.wall_impacts.append(position)
                else:
                    # Other effects have different layouts. Seek a complete
                    # frame rather than guessing and corrupting entity state.
                    self.errors += 1
                    self.last_error = f"unsupported temp entity {effect}"
                    for start in range(reader.pos, len(payload)):
                        if payload[start] != 20:
                            continue
                        probe = Reader(payload)
                        probe.pos = start + 1
                        try:
                            frame = self._frame(probe)
                        except PacketError:
                            continue
                        found.append(frame)
                        reader.pos = probe.pos
                        break
                    else:
                        break
            elif opcode == 9:
                flags = reader.byte()
                reader.byte()
                reader.take(bool(flags & 1) + bool(flags & 2) + bool(flags & 16))
                if flags & 8:
                    reader.take(2)
                if flags & 4:
                    reader.take(6)
            elif opcode == 5:
                reader.take(256 * 2)
            elif opcode == 16:
                size = reader.short()
                reader.byte()  # percent
                if size > 0:
                    reader.take(size)
            else:
                # Effects can precede a frame. Recover by seeking a structurally
                # valid svc_frame; malformed candidates leave decoder state intact.
                self.errors += 1
                self.last_error = f"unsupported opcode {opcode} at {reader.pos - 1}"
                for start in range(reader.pos, len(payload)):
                    if payload[start] != 20:
                        continue
                    probe = Reader(payload)
                    probe.pos = start + 1
                    try:
                        frame = self._frame(probe)
                    except PacketError:
                        continue
                    found.append(frame)
                    reader.pos = probe.pos
                    break
                else:
                    break
        return found

    def snapshot(self, frame: Frame) -> dict | None:
        if not self.map_name or self.player_number <= 0:
            return None
        maxclients = int(self.config.get(30, "4") or "4")
        teammates = [entity for entity in frame.entities.values()
                     if 0 < entity.number <= maxclients
                     and entity.number != self.player_number and entity.model == 255]
        if teammates:
            teammate = min(teammates, key=lambda item: item.number)
            self.teammate_origin = teammate.origin
        if self.teammate_origin is None:
            return None
        enemies, pickups = [], []
        for entity in frame.entities.values():
            path = self.config.get(32 + entity.model, "").lower()
            distance = sum((a - b) ** 2 for a, b in zip(entity.origin, frame.origin))
            if "/monsters/" in path and distance < 1024 ** 2:
                kind = path.split("/monsters/", 1)[1].split("/", 1)[0]
                # Stock protocol does not expose monster health. The model
                # animation still identifies corpses for these common foes.
                if any(first <= entity.frame <= last
                       for first, last in DEATH_FRAMES.get(kind, ())):
                    continue
                enemies.append({"id": entity.number, "class": f"monster_{kind}",
                                "origin": entity.origin, "frame": entity.frame,
                                "health": None,
                                "velocity": (0.0, 0.0, 0.0), "distance": distance})
            elif "/items/" in path and distance < 384 ** 2:
                kind = "item_health" if "heal" in path else "item_" + path.split("/items/", 1)[1].split("/", 1)[0]
                pickups.append({"id": entity.number, "class": kind,
                                "origin": entity.origin, "distance": distance})
        enemies.sort(key=lambda item: item.pop("distance"))
        pickups.sort(key=lambda item: item.pop("distance"))
        gun_path = self.config.get(32 + frame.gun, "").lower()
        weapon = "Blaster" if "blast" in gun_path else gun_path.split("/")[2] if "/weapons/" in gun_path else "unknown"
        return {"map": self.map_name, "player_origin": frame.origin,
                "bot_origin": self.teammate_origin, "player_health": frame.stats[1],
                "player_armor": frame.stats[5], "player_ammo": frame.stats[3],
                "player_weapon": weapon, "enemies": enemies[:4],
                "pickups": pickups[:3], "time": frame.number / 10.0,
                "delta_angles": frame.delta_angles}
