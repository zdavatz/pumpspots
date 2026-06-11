// Pumpspots map PDF generator.
//
// Fetches OpenStreetMap tiles, stitches them into a single image around a list
// of spots, draws red markers at each spot, and writes an A4-landscape PDF
// where each marker is a clickable link to Google Maps.
//
// No API key required (OSM tiles are public; tile usage policy honoured via
// the User-Agent header).

use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::time::Duration;

use ab_glyph::{FontRef, PxScale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_text_mut, text_size};
use printpdf::{
    Actions, BorderArray, ColorArray, HighlightingMode, ImageTransform, ImageXObject,
    LinkAnnotation, Mm, PdfDocument, Px, Rect,
};

// (label, lat, lon, optional URL override for the click target).
// If the URL is None, a `maps.google.com/maps/search?…&query=lat,lon` link is
// generated. Use Some(...) to point at a curated shortened Google Maps URL.
const SPOTS: &[(&str, f64, f64, Option<&str>)] = &[
    ("Hafen Murten",      46.930929, 7.115784, Some("https://maps.app.goo.gl/bAQxsJh1gK4S2XVb7")),
    ("Schiffsteg Praz",   46.951656, 7.097275, None),
    ("Schiffsteg Môtier", 46.946850, 7.084061, None),
    ("Hafen Sugiez",      46.966051, 7.113715, Some("https://maps.app.goo.gl/1psju5aghy4c5hyBA")),
    ("Hafen Faoug",       46.908229, 7.070834, Some("https://maps.app.goo.gl/re92N9BxHH9x4LDX9")),
    ("Bise Noir Pump",    46.934980, 7.121314, Some("https://maps.app.goo.gl/3JPWtXwiR8gFFKb46")),
];

const MAP_W: u32 = 2200;
const MAP_H: u32 = 1500;
const TILE: u32 = 256;
const PAD_FRAC: f64 = 0.15; // padding around bbox

// Web Mercator helpers — output in *tile* units at the given zoom (i.e. the
// whole world is 2^z tiles wide; multiply by 256 to get pixel coords).
fn lon_to_x_tiles(lon: f64, z: u32) -> f64 {
    (lon + 180.0) / 360.0 * (1u32 << z) as f64
}
fn lat_to_y_tiles(lat: f64, z: u32) -> f64 {
    let r = lat.to_radians();
    (1.0 - (r.tan() + 1.0 / r.cos()).ln() / PI) / 2.0 * (1u32 << z) as f64
}

fn pick_zoom_and_centre() -> (u32, f64, f64) {
    let (mut lat_min, mut lat_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut lon_min, mut lon_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, lat, lon, _) in SPOTS {
        lat_min = lat_min.min(lat);
        lat_max = lat_max.max(lat);
        lon_min = lon_min.min(lon);
        lon_max = lon_max.max(lon);
    }
    let dlat = lat_max - lat_min;
    let dlon = lon_max - lon_min;
    let lat_min = lat_min - dlat * PAD_FRAC;
    let lat_max = lat_max + dlat * PAD_FRAC;
    let lon_min = lon_min - dlon * PAD_FRAC;
    let lon_max = lon_max + dlon * PAD_FRAC;

    let mut best = 1;
    for z in 1..=18 {
        let dx_px = (lon_to_x_tiles(lon_max, z) - lon_to_x_tiles(lon_min, z)) * TILE as f64;
        let dy_px = (lat_to_y_tiles(lat_min, z) - lat_to_y_tiles(lat_max, z)) * TILE as f64;
        if dx_px > MAP_W as f64 || dy_px > MAP_H as f64 {
            break;
        }
        best = z;
    }
    let centre_lat = (lat_min + lat_max) / 2.0;
    let centre_lon = (lon_min + lon_max) / 2.0;
    (best, centre_lat, centre_lon)
}

fn fetch_tile(client: &reqwest::blocking::Client, z: u32, x: u32, y: u32) -> Result<RgbaImage, Box<dyn Error>> {
    let url = format!("https://tile.openstreetmap.org/{}/{}/{}.png", z, x, y);
    let bytes = client.get(&url).send()?.error_for_status()?.bytes()?;
    Ok(image::load_from_memory(&bytes)?.to_rgba8())
}

/// Render the map and return `(image, marker_pixel_coords)`.
fn render_map() -> Result<(RgbaImage, Vec<(i32, i32)>), Box<dyn Error>> {
    let (z, c_lat, c_lon) = pick_zoom_and_centre();
    let cx_px = lon_to_x_tiles(c_lon, z) * TILE as f64;
    let cy_px = lat_to_y_tiles(c_lat, z) * TILE as f64;
    let x0 = cx_px - MAP_W as f64 / 2.0;
    let y0 = cy_px - MAP_H as f64 / 2.0;

    let tx_min = (x0 / TILE as f64).floor() as i32;
    let ty_min = (y0 / TILE as f64).floor() as i32;
    let tx_max = ((x0 + MAP_W as f64) / TILE as f64).ceil() as i32;
    let ty_max = ((y0 + MAP_H as f64) / TILE as f64).ceil() as i32;
    let max_tile = 1i32 << z;

    let client = reqwest::blocking::Client::builder()
        .user_agent("pumpspots-map/0.1 (+https://github.com/zdavatz/pumpspots)")
        .timeout(Duration::from_secs(20))
        .build()?;

    let mut canvas = RgbaImage::new(MAP_W, MAP_H);
    println!("  zoom {} → fetching {}×{} tiles…", z, tx_max - tx_min, ty_max - ty_min);
    for tx in tx_min..tx_max {
        for ty in ty_min..ty_max {
            if ty < 0 || ty >= max_tile {
                continue;
            }
            let txw = ((tx % max_tile) + max_tile) % max_tile;
            let tile = fetch_tile(&client, z, txw as u32, ty as u32)?;
            let dx = (tx as f64 * TILE as f64 - x0) as i64;
            let dy = (ty as f64 * TILE as f64 - y0) as i64;
            image::imageops::overlay(&mut canvas, &tile, dx, dy);
        }
    }

    // Marker pixel coords (in canvas space).
    let mut markers = Vec::with_capacity(SPOTS.len());
    for &(_, lat, lon, _) in SPOTS {
        let mx = (lon_to_x_tiles(lon, z) * TILE as f64 - x0) as i32;
        let my = (lat_to_y_tiles(lat, z) * TILE as f64 - y0) as i32;
        markers.push((mx, my));
    }

    // Draw markers: red dot + white core.
    for &(mx, my) in &markers {
        draw_filled_circle_mut(&mut canvas, (mx, my), 26, Rgba([214, 39, 40, 255]));
        draw_filled_circle_mut(&mut canvas, (mx, my), 9, Rgba([255, 255, 255, 255]));
    }

    // Draw labels next to each marker — bold black text with a white halo so
    // it's readable over any map background.
    let font_bytes: &[u8] = include_bytes!("../assets/DejaVuSans-Bold.ttf");
    let font = FontRef::try_from_slice(font_bytes)
        .map_err(|e| format!("font load failed: {:?}", e))?;
    let scale = PxScale::from(46.0);
    for (i, (_, _, _, _)) in SPOTS.iter().enumerate() {
        let (mx, my) = markers[i];
        let name = SPOTS[i].0;
        let (tw, th) = text_size(scale, &font, name);
        // Default: label to the right of the marker. If too close to right
        // edge, place to the left instead.
        let gap = 36_i32;
        let mut tx = mx + gap;
        let ty = my - th as i32 / 2;
        if tx + tw as i32 > MAP_W as i32 - 20 {
            tx = mx - gap - tw as i32;
        }
        // White halo: draw the text repeatedly with small offsets.
        let halo = Rgba([255u8, 255, 255, 255]);
        let fg = Rgba([0u8, 0, 0, 255]);
        for (dx, dy) in [
            (-3, 0), (3, 0), (0, -3), (0, 3),
            (-2, -2), (2, -2), (-2, 2), (2, 2),
        ] {
            draw_text_mut(&mut canvas, halo, tx + dx, ty + dy, scale, &font, name);
        }
        draw_text_mut(&mut canvas, fg, tx, ty, scale, &font, name);
    }

    Ok((canvas, markers))
}

fn build_pdf(out: &str) -> Result<(), Box<dyn Error>> {
    let (canvas, markers) = render_map()?;

    // A4 landscape: 297 × 210 mm.
    let page_w_mm: f32 = 297.0;
    let page_h_mm: f32 = 210.0;
    let (doc, page1, layer1) =
        PdfDocument::new("Pumpspots Murtensee", Mm(page_w_mm), Mm(page_h_mm), "Layer 1");
    let layer = doc.get_page(page1).get_layer(layer1);

    // Embed map as image. printpdf wants raw 8-bit pixel data.
    let (w, h) = canvas.dimensions();
    let raw = canvas.into_raw();
    // Strip alpha → RGB (printpdf RGB image)
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for chunk in raw.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    let img_xobj = ImageXObject {
        width: Px(w as usize),
        height: Px(h as usize),
        color_space: printpdf::ColorSpace::Rgb,
        bits_per_component: printpdf::ColorBits::Bit8,
        interpolate: true,
        image_data: rgb,
        image_filter: None,
        smask: None,
        clipping_bbox: None,
    };
    let pdf_image = printpdf::Image::from(img_xobj);

    // Scale image to fill page via DPI: page_w_mm = (w_px / dpi) * 25.4
    //   → dpi = w_px * 25.4 / page_w_mm
    let dpi = (w as f32) * 25.4 / page_w_mm;
    pdf_image.add_to_layer(
        layer.clone(),
        ImageTransform {
            translate_x: Some(Mm(0.0)),
            translate_y: Some(Mm(0.0)),
            rotate: None,
            scale_x: Some(1.0),
            scale_y: Some(1.0),
            dpi: Some(dpi),
        },
    );

    // Clickable link annotations over each marker. Rect is in Mm.
    let r_mm: f32 = 6.0; // hit area in mm (~17 pt)
    let sx_mm = page_w_mm / w as f32;
    let sy_mm = page_h_mm / h as f32;
    for (&(mx, my), &(_, lat, lon, override_url)) in markers.iter().zip(SPOTS.iter()) {
        let cx_mm = mx as f32 * sx_mm;
        let cy_mm = page_h_mm - my as f32 * sy_mm; // PDF y origin is bottom-left
        let url = match override_url {
            Some(u) => u.to_string(),
            None => format!("https://www.google.com/maps/search/?api=1&query={},{}", lat, lon),
        };
        let rect = Rect::new(
            Mm(cx_mm - r_mm),
            Mm(cy_mm - r_mm),
            Mm(cx_mm + r_mm),
            Mm(cy_mm + r_mm),
        );
        // Border width 0 → invisible rectangle.
        let link = LinkAnnotation::new(
            rect,
            Some(BorderArray::Solid([0.0, 0.0, 0.0])),
            Some(ColorArray::default()),
            Actions::uri(url),
            Some(HighlightingMode::Invert),
        );
        layer.add_link_annotation(link);
    }

    let mut writer = BufWriter::new(File::create(out)?);
    doc.save(&mut writer)?;
    println!("  wrote {}", out);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "murtensee.pdf".to_string());
    build_pdf(&out)
}
