#!/usr/bin/env python3
"""Render a list of spots on an OSM-tile map and save as a PDF where each
marker is a clickable link that opens the spot in Google Maps.

Usage:
    python make_map_pdf.py <out.pdf>

No API key required — uses OpenStreetMap tiles via the `staticmap` package.
PDF: A4 landscape. Each red marker has a transparent clickable rectangle
linking to `https://www.google.com/maps/search/?api=1&query=LAT,LON`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from staticmap import CircleMarker, StaticMap
from staticmap.staticmap import _lon_to_x, _lat_to_y

# (label, lat, lon)
SPOTS: list[tuple[str, float, float]] = [
    ("Hafen Murten",      46.929874, 7.117002),
    ("Schiffsteg Praz",   46.951656, 7.097275),
    ("Schiffsteg Môtier", 46.946850, 7.084061),
    ("Hafen Sugiez",      46.966221, 7.114491),
    ("Hafen Faoug",       46.905310, 7.072986),
]

# Render the map at this pixel size; reportlab scales it onto the page.
MAP_W, MAP_H = 2200, 1500
MARKER_RADIUS_PX = 26  # also defines the clickable area


def gmaps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def render_map_and_get_marker_pixels() -> tuple[Path, list[tuple[float, float]]]:
    m = StaticMap(
        MAP_W,
        MAP_H,
        url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        headers={"User-Agent": "pumpspots-map/1.0"},
    )
    for _, lat, lon in SPOTS:
        m.add_marker(CircleMarker((lon, lat), "#d62728", MARKER_RADIUS_PX))
        m.add_marker(CircleMarker((lon, lat), "#ffffff", MARKER_RADIUS_PX // 3))

    img = m.render()  # after this, m.zoom, m.x_center, m.y_center are set

    pixels: list[tuple[float, float]] = []
    for _, lat, lon in SPOTS:
        tx = _lon_to_x(lon, m.zoom)
        ty = _lat_to_y(lat, m.zoom)
        px = m._x_to_px(tx)
        py = m._y_to_px(ty)
        pixels.append((px, py))

    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    img.save(tmp, "PNG")
    return tmp, pixels


def build_pdf(out_path: Path) -> None:
    map_png, marker_px = render_map_and_get_marker_pixels()

    page_w, page_h = landscape(A4)  # in PDF points (1/72 inch)
    c = canvas.Canvas(str(out_path), pagesize=landscape(A4))
    c.drawImage(str(map_png), 0, 0, width=page_w, height=page_h)

    # Convert image pixel coords → PDF point coords. Y axis flips (PDF origin
    # is bottom-left, image is top-left).
    sx = page_w / MAP_W
    sy = page_h / MAP_H
    link_r = MARKER_RADIUS_PX * sx * 1.4  # slightly larger than the dot

    for (lat_lon, (px, py)) in zip(SPOTS, marker_px):
        _, lat, lon = lat_lon
        x_pdf = px * sx
        y_pdf = page_h - py * sy
        rect = (x_pdf - link_r, y_pdf - link_r, x_pdf + link_r, y_pdf + link_r)
        c.linkURL(gmaps_url(lat, lon), rect, relative=0, thickness=0)

    c.showPage()
    c.save()
    map_png.unlink(missing_ok=True)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("murtensee.pdf")
    print(f"  rendering {len(SPOTS)} spots → OSM tiles…")
    build_pdf(out)
    print(f"  wrote {out} — markers are clickable links to Google Maps")


if __name__ == "__main__":
    main()
