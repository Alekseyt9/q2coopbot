"""Offline first-life Machinegun ammo efficiency; never supplies bot observations."""
import argparse
import json
import pathlib


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def episode(batch, result):
    run = batch / f"worker-{result['worker']}-episode-{result['episode']}"
    life = result["first_life"]
    frames = {}
    with (run / "bot.jsonl").open(encoding="utf-8-sig") as trace:
        for line in trace:
            row = json.loads(line)
            if life["start_frame"] <= row["frame"] <= life["end_frame"]:
                frames[row["frame"]] = row
    spent = 0
    increases = 0
    previous = None
    for row in sorted(frames.values(), key=lambda r: r["frame"]):
        # Quake clears the ammo HUD on death; that is not bullet consumption.
        if row["health"] <= 0:
            break
        if "v_machn" not in row.get("weapon", ""):
            previous = None
            continue
        ammo = row["ammo"]
        if previous is not None:
            spent += max(0, previous - ammo)
            increases += int(ammo > previous)
        previous = ammo
    if increases:
        raise ValueError("Ammo replenishment makes this fixed-loadout metric invalid")
    damage = sum(t["health_damage"] for t in life["outgoing_by_target_class"])
    return {"seed": result["seed"], "observed_alive_rounds_consumed": spent,
            "monster_health_damage": damage,
            "damage_per_observed_alive_round": damage / spent if spent else None,
            "terminal_ammo_unknown": life["end_reason"] == "first_observed_death",
            "scope": "Alive Machinegun HUD deltas only; first unobserved and death-tick ammo consumption excluded. Includes shots at props/world. Damage ratio is diagnostic, not exact shot accuracy."}


def report(batch):
    captured = read(batch / "report.json")
    if not captured["capture_complete"] or not captured["provenance_valid"]:
        raise ValueError("Incomplete or unverified capture")
    if read(batch / "manifest.json")["loadout"] != "machinegun":
        raise ValueError("Requires fixed Machinegun")
    episodes = [episode(batch, r) for r in captured["results"]]
    rounds = sum(r["observed_alive_rounds_consumed"] for r in episodes)
    damage = sum(r["monster_health_damage"] for r in episodes)
    return {"episodes": episodes, "observed_alive_rounds_consumed": rounds,
            "monster_health_damage": damage,
            "damage_per_observed_alive_round": damage / rounds if rounds else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(report(args.batch), indent=2), encoding="utf-8")
    print(args.out)
