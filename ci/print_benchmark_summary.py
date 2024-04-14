#!/usr/bin/env python3

import unittest
import asyncio
import argparse
import json
import re
import sys
import os
import tabulate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs=1)
    parser.add_argument("baseline_results", nargs=1)

    args = parser.parse_args()

    with open(args.results[0], "r") as f:
        results = {entry["name"]: entry["value"] for entry in json.load(f)}

    try:
        with open(args.baseline_results[0], "r") as f:
            baseline_results = {entry["name"]: entry["value"] for entry in json.load(f)}
    except FileNotFoundError:
        baseline_results: dict[str, float] = {}
    
    keys = sorted(list(results.keys()))
    values: list[str] = []
    for key in keys:
        val_str = f"{results[key]:.3f}"
        if key in baseline_results:
            diff = results[key] - baseline_results[key]

            emoji = ""
            sign = ""
            if diff > 0:
                emoji = "🔺 "
                sign = "+"
            elif diff < 0:
                emoji = "🔻 "
                sign = "-"
            
            diff_str = f" ({emoji}{sign}{abs(diff):.3f})"

        values.append(val_str + diff_str)

    table_str = tabulate.tabulate([values], headers=keys, tablefmt="github")
    print(table_str)

if __name__ == "__main__":
    main()