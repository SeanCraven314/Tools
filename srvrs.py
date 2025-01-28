#!/usr/bin/python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import subprocess
import sys
from pathlib import Path


SERVERS = ["roxy", "lroxy", "betty", "lbetty"]
WRITE_PATH = Path.home() / ".srvrs.log"


def main() -> None:
    print("Fetching server util:")
    out = f"|{'Name':^40}|{'Mem':^4}|{'Vol':^4}|\n"
    for srvr in SERVERS:
        try:
            stdout = subprocess.run(
                args=["ssh", srvr, "nvidia-smi"],
                capture_output=True,
                timeout=2,
            ).stdout
        except subprocess.TimeoutExpired as e:
            continue

        lines = stdout.decode().split("\n")
        for i, line in enumerate(lines):
            if "MiB /" in line:
                name_line = lines[max(i - 1, 0)]
                name = name_line.split("|")[1]
                gpu_name = f"{srvr}:{name[:-4].strip()}"
                [memory, util] = line.split("|")[-3:-1]
                memory = format_memory(memory)
                util = format_vol_util(util)
                out += f"|{gpu_name:<40}|{memory:>13}|{util:>13}|\n"

    out = out.rstrip()
    if is_daemon():
        WRITE_PATH.touch()
        with WRITE_PATH.open("w") as f:
            f.write(out)
        sys.exit(0)
    else:
        print(out)
        sys.exit(0)


def is_daemon() -> bool:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", "-d", default=False, action="store_true")
    args = parser.parse_args()
    return args.daemon


def green(text: str):
    green_text = f"\033[92m{text}\033[0m"
    return green_text


def red(text: str):
    green_text = f"\033[91m{text}\033[0m"
    return green_text


def format_float_fraction(f: float) -> str:
    if f > 0.5:
        return red(f"{f * 100:.0f}%")
    else:
        return green(f"{f * 100:.0f}%")


def format_vol_util(util_str: str) -> str:
    [percent, _] = util_str.rsplit("%", maxsplit=1)
    util = float(percent) / 100
    return format_float_fraction(util)


def format_memory(memeory_util_str: str) -> str:
    [util, max] = memeory_util_str.split("/")
    util = util.strip()
    max = max.strip()
    integer_util = int(util[:-3])
    integer_max = int(max[:-3])
    usage = integer_util / integer_max
    return format_float_fraction(usage)


if __name__ == "__main__":
    main()
