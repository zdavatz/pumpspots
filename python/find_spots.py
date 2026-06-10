#!/usr/bin/env python3
"""Look up pumpspot candidates via Google Places API (New) and emit CSV rows
matching the pumpspots.csv schema:

    Latitude;Longitude;Name of Spot;Name of Spotfinder;Height to water;Freetext;Link to images of the Spot

Usage
-----
    # one-off lookup, prints first hit
    python find_spots.py "Hafen Murten"

    # batch: one query per line on stdin -> CSV rows on stdout
    python find_spots.py --batch < queries.txt

    # batch with location bias around a centre (lake middle), 15 km radius
    python find_spots.py --batch --bias 46.93,7.10 --radius 15000 < queries.txt

The API key is read from `pumpspots/.google-maps-key` (gitignored).
Height/freetext/instagram are left as TODO markers — fill in by hand once
you've confirmed the spot. The README requires personal experience at the
spot, so this tool only fetches coordinates, never invents details.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
KEY_FILE = Path(__file__).resolve().parent.parent / ".google-maps-key"


def load_key() -> str:
    if not KEY_FILE.exists():
        sys.exit(f"missing API key file: {KEY_FILE}")
    return KEY_FILE.read_text().strip()


def search(query: str, key: str, bias: tuple[float, float] | None, radius: int) -> list[dict]:
    payload: dict = {"textQuery": query}
    if bias is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {"latitude": bias[0], "longitude": bias[1]},
                "radius": radius,
            }
        }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": "places.displayName,places.location,places.formattedAddress,places.types",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
    return data.get("places", []) or []


def csv_row(place: dict, spotfinder: str) -> str:
    name = place["displayName"]["text"].replace(";", ",")
    lat = place["location"]["latitude"]
    lon = place["location"]["longitude"]
    # Schema: Lat;Lon;Name;Spotfinder;Height;Freetext;Link
    return f"{lat:.6f};{lon:.6f};{name};{spotfinder};TODO;TODO;TODO"


def parse_bias(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    lat, lon = s.split(",")
    return float(lat), float(lon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="single query (omit with --batch)")
    ap.add_argument("--batch", action="store_true", help="read one query per line from stdin")
    ap.add_argument("--bias", help="lat,lon location bias centre")
    ap.add_argument("--radius", type=int, default=15000, help="bias radius in metres (default 15000)")
    ap.add_argument("--spotfinder", default="TODO", help="value for column 4")
    ap.add_argument("--n", type=int, default=1, help="how many hits per query to emit (default 1)")
    args = ap.parse_args()

    key = load_key()
    bias = parse_bias(args.bias)

    if args.batch:
        queries = [line.strip() for line in sys.stdin if line.strip() and not line.startswith("#")]
    elif args.query:
        queries = [args.query]
    else:
        ap.error("provide a query or use --batch")
        return

    for q in queries:
        places = search(q, key, bias, args.radius)
        if not places:
            print(f"# no hit: {q}", file=sys.stderr)
            continue
        for p in places[: args.n]:
            print(csv_row(p, args.spotfinder))


if __name__ == "__main__":
    main()
