from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from PIL import Image  # optional compression for handout photos
except ImportError:  # pragma: no cover - Pillow is optional
    Image = None  # type: ignore[assignment]

from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import RectangleObject
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# --- Input & output --------------------------------------------------------
PDF_PATH = Path("1 200k original karta.pdf")  # base map (page 1)
GPX_PATH = Path("activity_20435991472.gpx")  # GPX flight track
PHOTO_PATTERN = "IMG_*.jpg"  # onboard photos sourced from camera

PHOTO_ANALYSIS_CSV = Path("photo_analysis.csv")
OUTPUT_OVERLAY = Path("route_overlay.pdf")
OUTPUT_MERGED = Path("route_marked.pdf")
OUTPUT_KEY = Path("photo_overlay_key.csv")
HANDOUT_PDF = Path("photo_handout.pdf")
MAP_IMAGE = Path("map_page.png")  # cached raster of base map
CROPPED_MAP_PDF = Path("route_cropped.pdf")

# --- Map geometry ----------------------------------------------------------
BASE_WIDTH = 1191.0  # px width of preview coordinate system
BASE_HEIGHT = 842.0  # px height of preview coordinate system
METERS_PER_MM = 200.0  # paper scale (1 mm = 200 m)

# --- Feature visibility ----------------------------------------------------
#DRAW_ROUTE = True  # draw main route polyline
DRAW_ROUTE = False  # draw main route polyline
#DRAW_TURNPOINTS = True  # draw turnpoint circles and labels
DRAW_TURNPOINTS = False  # draw turnpoint circles and labels
#DRAW_PHOTO_MARKERS = True  # mark photo positions on the map
DRAW_PHOTO_MARKERS = False  # mark photo positions on the map
#DRAW_CONTROL_PHOTO_MARKERS = True  # include SP/TP/FP photo markers on the map
DRAW_CONTROL_PHOTO_MARKERS = False  # include SP/TP/FP photo markers on the map
#DRAW_PHOTO_DOTS = True  # show exact GPS location of each photo and check for lateral distance
DRAW_PHOTO_DOTS = False  # show exact GPS location of each photo and check for lateral distance
#DRAW_HEADINGS = True  # annotate leg headings
DRAW_HEADINGS = False  # annotate leg headings
#DRAW_MINUTE_MARKERS = True  # place minute ticks along the route
DRAW_MINUTE_MARKERS = False  # place minute ticks along the route

#HANDOUT_INCLUDE_SUMMARY = True  # append executive summary page to handout
HANDOUT_INCLUDE_SUMMARY = False  # append executive summary page to handout

# --- Colour palette --------------------------------------------------------
COLOURS = {
    "red": (1.0, 0.0, 0.0),
    "dark_red": (0.82, 0.0, 0.0),
    "orange": (1.0, 0.55, 0.0),
    "yellow": (1.0, 0.9, 0.0),
    "lime": (0.75, 1.0, 0.0),
    "green": (0.0, 0.6, 0.0),
    "teal": (0.0, 0.55, 0.55),
    "cyan": (0.0, 0.7, 0.9),
    "blue": (0.13, 0.4, 0.85),
    "navy": (0.05, 0.15, 0.4),
    "violet": (0.5, 0.2, 0.7),
    "magenta": (0.85, 0.0, 0.55),
    "brown": (0.55, 0.27, 0.07),
    "grey": (0.5, 0.5, 0.5),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}

ROUTE_COLOR = COLOURS["dark_red"]  # stroke color for route polyline
TP_COLOR = COLOURS["dark_red"]  # color of turnpoint circles
PHOTO_COLOR = COLOURS["violet"]  # marker color for photos
HEADING_COLOR = COLOURS["red"]  # color for leg heading text
TP_LABEL_COLOR = COLOURS["black"]  # TP label text color
PHOTO_LABEL_COLOR = COLOURS["violet"]  # photo label text color
PHOTO_LABEL_HALO = COLOURS["white"]  # halo behind photo labels

# --- Vector styling --------------------------------------------------------
ROUTE_WIDTH_SCALE = 2.5  # route line thickness multiplier
TP_RADIUS_SCALE = 20.0  # turnpoint circle radius multiplier
PHOTO_TICK_HALF_SCALE = 7.0  # half-length of photo marker cross
PHOTO_LINE_WIDTH_SCALE = 4.2  # thickness of photo marker lines
PHOTO_DOT_RADIUS_SCALE = 5.0  # radius multiplier for exact-location dots
TP_FONT_SCALE = 12.0  # base font size for turnpoint labels
PHOTO_FONT_SCALE = 14.0  # font size for photo labels on overlay
HEADING_FONT_SCALE = 18.0  # font size for leg heading text
HEADING_OFFSET_SCALE = 60.0  # distance headings sit from leg
PHOTO_LABEL_OFFSET_MULTIPLIER = 4.0  # label distance from photo marker
LABEL_MIN_DISTANCE = 10.0  # minimum spacing between labels (pt)
LABEL_DISTANCE_STEP = 5.0  # step per adjustment iteration (pt)
MAX_LABEL_ADJUST_STEPS = 12  # max retries when nudging labels
LABEL_COLLISION_MARGIN = 3.0  # padding around label bounding boxes

# --- Route timing & spacing ------------------------------------------------
PHOTO_MIN_DISTANCE_AFTER_TP_NM = 1.0  # minimum photo spacing after TP (NM)
PHOTO_MAX_LATERAL_DISTANCE_M = 1000.0  # maximum lateral offset allowed for photos
TP_TIME_FONT_SCALE = 12.0  # base font size for TP timing annotation
TP_LABEL_OFFSET_FACTOR = 0.35  # TP font multiplier controlling label offset
TP_TIME_SWEEP_FACTOR = 0.85  # multiplier determining time label distance
TAKEOFF_TO_SP_MIN = 4.0  # minutes from takeoff to start point (SP)
DEFAULT_SPEED_VALUE = 75.0  # default groundspeed magnitude
DEFAULT_SPEED_UNIT = "kt"  # accepted: kt or mph

# --- Minute markers --------------------------------------------------------
MINUTE_TICK_HALF_SCALE = 7.0  # half-length of minute marker cross
MINUTE_LINE_WIDTH_SCALE = 1.2  # thickness of minute markers
MINUTE_LABEL_FONT_SCALE = 10.0  # font size for minute labels
MINUTE_LABEL_OFFSET_MULTIPLIER = 4.0  # distance minute labels sit from marker
MINUTE_LABEL_PREFIX = ""  # optional prefix before minute numbers
MINUTE_MARKER_COLOR = COLOURS["navy"]  # color for minute tick marks
MINUTE_LABEL_COLOR = COLOURS["navy"]  # color for minute label text
MINUTE_LABEL_HALO = COLOURS["white"]  # halo color around minute labels

# --- Unit conversions ------------------------------------------------------
KNOT_TO_MPS = 0.514444  # converts knots to metres per second
MPH_TO_MPS = 0.44704  # converts miles per hour to metres per second
NM_TO_METERS = 1852.0  # length of one nautical mile in metres

# --- Style variants --------------------------------------------------------
STYLE_VARIANTS = [
    {
        "id": "",
        "label": "Route overlay",
        "tp_radius_scale": TP_RADIUS_SCALE,
    }
]

# --- Handout behaviour -----------------------------------------------------
HANDOUT_LETTER_SALT = "TVN2025"  # salt used for deterministic photo lettering
HANDOUT_SPLIT_TP = "TP5"  # turnpoint splitting alphabet halves
HANDOUT_LETTER_SCALE = 0.15  # handout letter size vs photo slot
HANDOUT_PHOTO_DPI = 220  # target DPI when compressing handout photos
CROPPED_MAP_MARGIN_MM = 10.0  # desired margin (mm) around route crop
A4_WIDTH, A4_HEIGHT = A4
PORTRAIT_PAGE = (A4_WIDTH, A4_HEIGHT)  # portrait A4 dimensions
LANDSCAPE_PAGE = (A4_HEIGHT, A4_WIDTH)  # landscape A4 dimensions

ROUTE_POINTS = [
    ("SP", 46.60085033500145, 16.18002295367121),
    ("TP1", 46.500800369520974, 16.155215089283285),
    ("TP2", 46.51512847255438, 16.00836867366053),
    ("TP3", 46.616968940165265, 16.069071738369164),
    ("TP4", 46.800823749993334, 16.036800946819525),
    ("TP5", 46.83695523, 16.30844783),
    ("TP6", 46.77518719523106, 16.202892517628687),
    ("TP7", 46.55234631757589, 16.430293184364793),
    ("FP", 46.608093848672596, 16.234769094059345),
]

ROUTE_POINT_INDEX = {name: idx for idx, (name, *_coords) in enumerate(ROUTE_POINTS)}

CONTROL_POINTS = [
    ("TP1", 46.500800369520974, 16.155215089283285, 609.0, 653.0),
    ("TP4", 46.80079362139448, 16.036790112456796, 476.0, 183.0),
    ("TP5", 46.83695523, 16.30844783, 768.0, 118.0),
    ("TP3", 46.616968940165265, 16.069071738369164,  513, 471),
]
LOCAL_TZ = timezone(timedelta(hours=2))
EARTH_RADIUS_M = 6_371_000.0

# DO NOT EDIT AFTER THIS

lat0 = sum(pt[1] for pt in ROUTE_POINTS) / len(ROUTE_POINTS)
lon0 = sum(pt[2] for pt in ROUTE_POINTS) / len(ROUTE_POINTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate precision flying overlays")
    parser.add_argument(
        "--speed",
        help="Groundspeed (e.g. 75kt, 80 kts, 75mph). Defaults to configured value.",
    )
    return parser.parse_args()


def parse_speed_arg(speed_arg: str | None) -> Tuple[float, str, float, str]:
    if speed_arg is None:
        value = DEFAULT_SPEED_VALUE
        unit = DEFAULT_SPEED_UNIT.lower()
    else:
        token = speed_arg.strip().lower().replace(" ", "")
        if token.endswith("kts"):
            unit = "kt"
            token = token[:-3]
        elif token.endswith("kt"):
            unit = "kt"
            token = token[:-2]
        elif token.endswith("mph"):
            unit = "mph"
            token = token[:-3]
        else:
            unit = DEFAULT_SPEED_UNIT.lower()
        try:
            value = float(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid --speed value: {speed_arg}") from exc

    if unit not in {"kt", "mph"}:
        raise SystemExit(f"Unsupported speed unit '{unit}'. Use kt or mph.")
    if value <= 0:
        raise SystemExit("Speed must be positive")

    if unit == "kt":
        mps = value * KNOT_TO_MPS
        label = f"{value:.1f} kt"
    else:
        mps = value * MPH_TO_MPS
        label = f"{value:.1f} mph"
    return value, unit, mps, label


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def latlon_to_xy(lat: float, lon: float) -> Tuple[float, float]:
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_RADIUS_M
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def build_legs() -> List[dict]:
    legs = []
    cumulative = 0.0
    for (name_a, lat_a, lon_a), (name_b, lat_b, lon_b) in zip(ROUTE_POINTS[:-1], ROUTE_POINTS[1:]):
        ax, ay = latlon_to_xy(lat_a, lon_a)
        bx, by = latlon_to_xy(lat_b, lon_b)
        length = haversine(lat_a, lon_a, lat_b, lon_b)
        legs.append(
            {
                "from_name": name_a,
                "to_name": name_b,
                "from_lat": lat_a,
                "from_lon": lon_a,
                "to_lat": lat_b,
                "to_lon": lon_b,
                "A_xy": (ax, ay),
                "B_xy": (bx, by),
                "length": length,
                "cumulative_start": cumulative,
            }
        )
        cumulative += length
    return legs


LEGS = build_legs()
LEG_LENGTH_LOOKUP = {
    f"{leg['from_name']}-{leg['to_name']}": leg["length"]
    for leg in LEGS
}


def ensure_map_image() -> None:
    if MAP_IMAGE.exists():
        return
    print(f"Map raster '{MAP_IMAGE}' missing; rendering from {PDF_PATH}...", flush=True)
    cmd = [
        "sips",
        "-s",
        "format",
        "png",
        str(PDF_PATH),
        "--out",
        str(MAP_IMAGE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"Failed to rasterize map PDF with sips (exit {proc.returncode}):\n{proc.stderr}"
        )
    print("  map raster created.", flush=True)


# ---------------------------------------------------------------------------
# GPX and photo analysis
# ---------------------------------------------------------------------------
def load_track_points(gpx_path: Path) -> List[Tuple[datetime, float, float]]:
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(gpx_path).getroot()
    points: List[Tuple[datetime, float, float]] = []
    for pt in root.findall(".//gpx:trkpt", ns):
        lat = float(pt.attrib["lat"])
        lon = float(pt.attrib["lon"])
        time_txt = pt.find("gpx:time", ns)
        if time_txt is None:
            continue
        time = datetime.fromisoformat(time_txt.text.replace("Z", "+00:00"))
        points.append((time, lat, lon))
    return points


def extract_creation_time(photo_path: Path) -> datetime:
    proc = subprocess.run(["sips", "-g", "creation", str(photo_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sips failed for {photo_path}: {proc.stderr}")
    for line in proc.stdout.splitlines():
        if "creation:" in line:
            _, creation_str = line.split(":", 1)
            local_dt = datetime.strptime(creation_str.strip(), "%Y:%m:%d %H:%M:%S")
            return local_dt.replace(tzinfo=LOCAL_TZ)
    raise RuntimeError(f"No creation time for {photo_path}")


def interpolate_track_time(track: List[Tuple[datetime, float, float]], target: datetime) -> Tuple[float, float]:
    if target <= track[0][0]:
        idx = 0
    elif target >= track[-1][0]:
        idx = len(track) - 2
    else:
        lo, hi = 0, len(track) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if track[mid][0] < target:
                lo = mid + 1
            else:
                hi = mid
        idx = max(0, lo - 1)
    t0, lat0_pt, lon0_pt = track[idx]
    t1, lat1_pt, lon1_pt = track[idx + 1]
    if t1 == t0:
        frac = 0.0
    else:
        frac = (target - t0).total_seconds() / (t1 - t0).total_seconds()
    lat_interp = lat0_pt + (lat1_pt - lat0_pt) * frac
    lon_interp = lon0_pt + (lon1_pt - lon0_pt) * frac
    return lat_interp, lon_interp


def extract_control_hint(photo_name: str) -> str | None:
    """Return claimed control token (SP, FP, TPn) from filename if present."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", photo_name).upper()
    for token in cleaned.split():
        if token in {"SP", "FP"}:
            return token
        if token.startswith("TP") and token[2:].isdigit():
            return token
    return None


def project_to_leg(lat: float, lon: float) -> dict:
    px, py = latlon_to_xy(lat, lon)
    best = None
    for leg in LEGS:
        ax, ay = leg["A_xy"]
        bx, by = leg["B_xy"]
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        v_norm2 = vx * vx + vy * vy
        t_proj = (wx * vx + wy * vy) / v_norm2 if v_norm2 else 0.0
        t_clamped = max(0.0, min(1.0, t_proj))
        proj_x = ax + t_clamped * vx
        proj_y = ay + t_clamped * vy
        dx = px - proj_x
        dy = py - proj_y
        lateral = math.hypot(dx, dy)
        cross = vx * wy - vy * wx
        lateral_signed = lateral if cross >= 0 else -lateral
        if best is None or abs(lateral_signed) < abs(best["lateral_signed"]):
            leg_distance = t_clamped * leg["length"]
            best = {
                "leg": f"{leg['from_name']}-{leg['to_name']}",
                "from_name": leg["from_name"],
                "t_clamped": t_clamped,
                "leg_length": leg["length"],
                "leg_distance_m": leg_distance,
                "cumulative_start": leg["cumulative_start"],
                "lateral": lateral,
                "lateral_signed": lateral_signed,
            }
    if best is None:
        raise RuntimeError("Could not project photo to any leg")
    along = best["cumulative_start"] + best["t_clamped"] * best["leg_length"]
    return {
        "leg": best["leg"],
        "from_point": best["from_name"],
        "fraction_on_leg": best["t_clamped"],
        "leg_distance_m": best["leg_distance_m"],
        "along_m": along,
        "lateral_m": best["lateral"],
        "lateral_signed_m": best["lateral_signed"],
        "map_mm_from_start": best["leg_distance_m"] / METERS_PER_MM,
    }


def analyse_photos() -> List[dict]:
    if not GPX_PATH.exists():
        raise SystemExit(f"Missing GPX file: {GPX_PATH}")
    track = load_track_points(GPX_PATH)
    if len(track) < 2:
        raise SystemExit("GPX track does not contain enough points")

    photos = sorted(Path.cwd().glob(PHOTO_PATTERN))
    if not photos:
        raise SystemExit(f"No photos matching pattern {PHOTO_PATTERN}")

    analysis_rows: List[dict] = []
    for photo in photos:
        local_dt = extract_creation_time(photo)
        utc_dt = local_dt.astimezone(timezone.utc)
        lat_interp, lon_interp = interpolate_track_time(track, utc_dt)
        leg_info = project_to_leg(lat_interp, lon_interp)
        control_hint = extract_control_hint(photo.name)
        analysis_rows.append(
            {
                "photo": photo.name,
                "local_time": local_dt.isoformat(),
                "utc_time": utc_dt.isoformat(),
                "lat": lat_interp,
                "lon": lon_interp,
                "control_hint": control_hint,
                **leg_info,
            }
        )

    # write CSV for downstream tooling
    FIELDNAMES = [
        "photo",
        "local_time",
        "utc_time",
        "lat",
        "lon",
        "leg",
        "control_hint",
        "from_point",
        "fraction_on_leg",
        "leg_distance_m",
        "map_mm_from_start",
        "along_m",
        "lateral_m",
        "lateral_signed_m",
    ]
    with PHOTO_ANALYSIS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in analysis_rows:
            writer.writerow(row)

    print(f"Wrote {PHOTO_ANALYSIS_CSV} with {len(analysis_rows)} rows")
    return analysis_rows


# ---------------------------------------------------------------------------
# Map overlay helpers
# ---------------------------------------------------------------------------
@dataclass
class Affine:
    ax: float
    bx: float
    cx: float
    ay: float
    by: float
    cy: float

    def project(self, lat: float, lon: float) -> Tuple[float, float]:
        x = self.ax * lon + self.bx * lat + self.cx
        y = self.ay * lon + self.by * lat + self.cy
        return x, y


def _det3(m: Sequence[Sequence[float]]) -> float:
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def solve_affine(control_pts_named: Iterable[Tuple[str, float, float, float, float]]) -> Affine:
    lon_lat = [(lon, lat) for (_, lat, lon, _, _) in control_pts_named]
    x_targets = [x for (_, _, _, x, _) in control_pts_named]
    y_targets = [y for (_, _, _, _, y) in control_pts_named]

    A = [
        [lon_lat[0][0], lon_lat[0][1], 1.0],
        [lon_lat[1][0], lon_lat[1][1], 1.0],
        [lon_lat[2][0], lon_lat[2][1], 1.0],
    ]
    detA = _det3(A)
    if abs(detA) < 1e-9:
        raise ValueError("Control points are colinear; cannot solve affine transform.")

    def replace_col(col: int, values: Sequence[float]) -> List[List[float]]:
        M = [row[:] for row in A]
        for i in range(3):
            M[i][col] = values[i]
        return M

    det_x = _det3(replace_col(0, x_targets))
    det_y = _det3(replace_col(1, x_targets))
    det_cx = _det3(replace_col(2, x_targets))

    det_x2 = _det3(replace_col(0, y_targets))
    det_y2 = _det3(replace_col(1, y_targets))
    det_cy = _det3(replace_col(2, y_targets))

    return Affine(
        ax=det_x / detA,
        bx=det_y / detA,
        cx=det_cx / detA,
        ay=det_x2 / detA,
        by=det_y2 / detA,
        cy=det_cy / detA,
    )


def preview_to_pdf(x_base: float, y_base: float, page_width: float, page_height: float) -> Tuple[float, float]:
    x_pdf = x_base / BASE_WIDTH * page_width
    y_pdf = page_height - (y_base / BASE_HEIGHT * page_height)
    return x_pdf, y_pdf


def normalize(dx: float, dy: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def point_distance(px: float, py: float, qx: float, qy: float) -> float:
    return math.hypot(px - qx, py - qy)


def rotated_bbox(
    x: float,
    y: float,
    width: float,
    height: float,
    angle_deg: float,
) -> Tuple[float, float, float, float]:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    world = [
        (
            x + cx * cos_a - cy * sin_a,
            y + cx * sin_a + cy * cos_a,
        )
        for cx, cy in corners
    ]
    xs = [pt[0] for pt in world]
    ys = [pt[1] for pt in world]
    return (
        min(xs) - LABEL_COLLISION_MARGIN,
        min(ys) - LABEL_COLLISION_MARGIN,
        max(xs) + LABEL_COLLISION_MARGIN,
        max(ys) + LABEL_COLLISION_MARGIN,
    )


def boxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def adjust_label_position(
    x: float,
    y: float,
    dir_x: float,
    dir_y: float,
    width: float,
    height: float,
    angle_deg: float,
    placed_boxes: List[Tuple[float, float, float, float]],
    step: float = LABEL_DISTANCE_STEP,
    max_steps: int = MAX_LABEL_ADJUST_STEPS,
    fallback_angle: float | None = None,
) -> Tuple[float, float, Tuple[float, float, float, float]]:
    length = math.hypot(dir_x, dir_y)
    if length == 0:
        primary_dir = (1.0, 0.0)
    else:
        primary_dir = (dir_x / length, dir_y / length)

    def candidate_positions(direction: Tuple[float, float]):
        dx, dy = direction
        yield x, y
        for i in range(1, max_steps + 1):
            yield x + dx * i * step, y + dy * i * step
            yield x - dx * i * step, y - dy * i * step

    def is_clear(cx: float, cy: float) -> Tuple[bool, Tuple[float, float, float, float]]:
        bbox = rotated_bbox(cx, cy, width, height, angle_deg)
        for existing in placed_boxes:
            if boxes_overlap(bbox, existing):
                return False, bbox
        return True, bbox

    for cand_x, cand_y in candidate_positions(primary_dir):
        ok, bbox = is_clear(cand_x, cand_y)
        if ok:
            return cand_x, cand_y, bbox

    if fallback_angle is None:
        fallback_dir = (primary_dir[1], -primary_dir[0])
    else:
        rad = math.radians(fallback_angle)
        fallback_dir = (math.cos(rad), math.sin(rad))

    for cand_x, cand_y in candidate_positions(fallback_dir):
        ok, bbox = is_clear(cand_x, cand_y)
        if ok:
            return cand_x, cand_y, bbox

    return x, y, rotated_bbox(x, y, width, height, angle_deg)


def variant_path(base: Path, suffix: str) -> Path:
    if not suffix:
        return base
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def init_bounds(page_width: float, page_height: float) -> dict[str, float]:
    return {
        "min_x": page_width,
        "min_y": page_height,
        "max_x": 0.0,
        "max_y": 0.0,
    }


def bounds_valid(bounds: dict[str, float]) -> bool:
    return bounds["max_x"] > bounds["min_x"] and bounds["max_y"] > bounds["min_y"]


def add_point_to_bounds(bounds: dict[str, float], x: float, y: float) -> None:
    bounds["min_x"] = min(bounds["min_x"], x)
    bounds["min_y"] = min(bounds["min_y"], y)
    bounds["max_x"] = max(bounds["max_x"], x)
    bounds["max_y"] = max(bounds["max_y"], y)


def add_box_to_bounds(bounds: dict[str, float], box: Tuple[float, float, float, float]) -> None:
    min_x, min_y, max_x, max_y = box
    add_point_to_bounds(bounds, min_x, min_y)
    add_point_to_bounds(bounds, max_x, max_y)


def add_line_to_bounds(
    bounds: dict[str, float], x1: float, y1: float, x2: float, y2: float
) -> None:
    add_point_to_bounds(bounds, x1, y1)
    add_point_to_bounds(bounds, x2, y2)


def alphabetic_label(index: int) -> str:
    if index < 0:
        raise ValueError("Index must be non-negative")
    chars: List[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        chars.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(chars))


def handout_sort_key(photo: dict) -> str:
    token = f"{HANDOUT_LETTER_SALT}:{photo.get('photo', '')}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def ordered_enroute_photos(enroute_photos: List[dict]) -> List[dict]:
    if not enroute_photos:
        return []

    split_idx = ROUTE_POINT_INDEX.get(HANDOUT_SPLIT_TP) if HANDOUT_SPLIT_TP else None
    first_bucket: List[dict] = []
    second_bucket: List[dict] = []

    for photo in enroute_photos:
        bucket = first_bucket
        if split_idx is not None:
            leg = photo.get("leg", "")
            if "-" in leg:
                _, leg_to = leg.split("-", 1)
            else:
                leg_to = leg
            dest_idx = ROUTE_POINT_INDEX.get(leg_to)
            if dest_idx is not None and dest_idx > split_idx:
                bucket = second_bucket
        bucket.append(photo)

    first_sorted = sorted(first_bucket, key=handout_sort_key)
    second_sorted = sorted(second_bucket, key=handout_sort_key)
    return first_sorted + second_sorted


def create_cropped_map_pdf(
    merged_pdf: Path,
    bounds: dict[str, float],
    page_width: float,
    page_height: float,
    suffix: str,
) -> Path | None:
    if not bounds_valid(bounds):
        return None

    raw_width = bounds["max_x"] - bounds["min_x"]
    raw_height = bounds["max_y"] - bounds["min_y"]

    if raw_width <= 0 or raw_height <= 0:
        return None

    aspect = raw_width / raw_height if raw_height else 1.0
    preferred = LANDSCAPE_PAGE if aspect > 1.0 else PORTRAIT_PAGE
    orientations = [preferred, LANDSCAPE_PAGE if preferred == PORTRAIT_PAGE else PORTRAIT_PAGE]
    target_width = target_height = None
    for orient in orientations:
        ow, oh = orient
        if raw_width <= ow and raw_height <= oh:
            target_width, target_height = orient
            break
    if target_width is None:
        target_width, target_height = orientations[0]

    orientation_label = "landscape" if target_width > target_height else "portrait"

    margin_px = CROPPED_MAP_MARGIN_MM * mm
    if raw_width > target_width or raw_height > target_height:
        print(
            "Warning: route exceeds target page; reducing margins to zero",
        )
        effective_margin_x = 0.0
        effective_margin_y = 0.0
    else:
        max_margin_x = max(0.0, (target_width - raw_width) / 2.0)
        max_margin_y = max(0.0, (target_height - raw_height) / 2.0)
        effective_margin_x = min(margin_px, max_margin_x)
        effective_margin_y = min(margin_px, max_margin_y)

    min_x = max(0.0, bounds["min_x"] - effective_margin_x)
    min_y = max(0.0, bounds["min_y"] - effective_margin_y)
    max_x = min(page_width, bounds["max_x"] + effective_margin_x)
    max_y = min(page_height, bounds["max_y"] + effective_margin_y)

    if max_x <= min_x or max_y <= min_y:
        return None

    crop_width = max_x - min_x
    crop_height = max_y - min_y


    try:
        reader = PdfReader(str(merged_pdf))
    except Exception as exc:
        print(f"Warning: unable to read merged map for cropping: {exc}")
        return None

    if not reader.pages:
        return None

    page = copy.deepcopy(reader.pages[0])

    offset_x = (target_width - crop_width) / 2.0
    offset_y = (target_height - crop_height) / 2.0
    offset_x = max(0.0, offset_x)
    offset_y = max(0.0, offset_y)

    def apply_matrix(pg, matrix: Tuple[float, float, float, float, float, float]) -> None:
        try:
            pg.add_transformation(matrix)
        except AttributeError:
            if hasattr(pg, "addTransformation"):
                pg.addTransformation(matrix)
            else:
                raise

    crop_rect = RectangleObject([min_x, min_y, max_x, max_y])
    for attr in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
        setattr(page, attr, RectangleObject(crop_rect))

    apply_matrix(page, (1.0, 0.0, 0.0, 1.0, -min_x, -min_y))

    new_width = crop_width
    new_height = crop_height
    new_rect = RectangleObject([0.0, 0.0, new_width, new_height])
    for attr in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
        setattr(page, attr, RectangleObject(new_rect))

    apply_matrix(page, (1.0, 0.0, 0.0, 1.0, offset_x, offset_y))

    final_rect = RectangleObject([0.0, 0.0, target_width, target_height])
    for attr in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
        setattr(page, attr, RectangleObject(final_rect))

    writer = PdfWriter()
    writer.add_page(page)

    cropped_path = variant_path(CROPPED_MAP_PDF, suffix)
    try:
        with cropped_path.open("wb") as fh:
            writer.write(fh)
    except Exception as exc:
        print(f"Warning: unable to write cropped map PDF {cropped_path}: {exc}")
        return None

    print(
        f"Cropped map saved to {cropped_path} "
        f"({orientation_label}, content {crop_width / mm:.1f} × {crop_height / mm:.1f} mm)")
    return cropped_path


def assign_handout_labels(photo_data: List[dict]) -> None:
    enroute = [p for p in photo_data if not p.get("is_control")]
    ordered = ordered_enroute_photos(enroute)
    for idx, photo in enumerate(ordered):
        photo["handout_label"] = alphabetic_label(idx)

    for photo in photo_data:
        if photo.get("is_control"):
            control_label = control_point_handout_label(photo)
            if control_label:
                photo["handout_label"] = control_label
            else:
                photo.setdefault("handout_label", photo.get("label", ""))


def control_point_handout_label(photo: dict) -> str | None:
    if not photo.get("is_control"):
        return None

    hint = photo.get("control_hint")
    if isinstance(hint, str) and hint:
        return hint

    leg = photo.get("leg")
    if not leg or "-" not in leg:
        return None

    start_name, end_name = leg.split("-", 1)
    fraction = float(photo.get("fraction", 0.0))
    leg_length = LEG_LENGTH_LOOKUP.get(leg)
    if leg_length is None:
        return start_name

    distance_from_start = fraction * leg_length
    distance_from_end = abs(leg_length - distance_from_start)
    if distance_from_start <= distance_from_end:
        return start_name
    return end_name


def generate_summary_page(output_path: Path, info: dict) -> None:
    c = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4
    margin = 16 * mm
    map_box_width = 80 * mm
    map_box_height = 70 * mm
    map_box_left = page_width - margin - map_box_width
    map_box_bottom = margin
    map_box_top = map_box_bottom + map_box_height

    variant_label = info.get("variant_label", "Route")
    speed_label = info.get("speed_label", "")
    total_distance = info.get("total_distance_km", 0.0)
    photo_data = info.get("photo_data", [])
    control_count = sum(1 for p in photo_data if p.get("is_control"))
    enroute_count = max(0, len(photo_data) - control_count)
    minute_markers = info.get("minute_markers", [])
    excluded_photos = info.get("excluded_photos", [])
    offroute_warnings = info.get("offroute_warnings", [])
    control_close = info.get("control_close", [])

    def draw_metric(x: float, y: float, label: str, value: str) -> float:
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawString(x, y, label.upper())
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y - 10, value)
        return y - 18

    y = page_height - margin
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.black)
    c.drawString(margin, y, f"Executive Summary · {variant_label}")
    y -= 9 * mm
    tagline = (
        f"{speed_label}  •  {total_distance:.1f} km course  •  {len(photo_data)} photos "
        f"({enroute_count} enroute / {control_count} control)"
    ).strip()
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.grey)
    c.drawString(margin, y, tagline)
    c.setFillColor(colors.black)
    y -= 4 * mm
    c.setLineWidth(0.6)
    c.line(margin, y, map_box_left - 4 * mm, y)
    y -= 8 * mm

    metric_x_left = margin
    metric_x_right = margin + 70 * mm
    metric_y_left = y
    metric_y_right = y

    if minute_markers:
        minutes = sorted(m.get("minute", 0) for m in minute_markers)
        minute_summary = f"{len(minutes)} markers ({minutes[0]}–{minutes[-1]} min)"
    else:
        minute_summary = "None"

    metric_y_left = draw_metric(metric_x_left, metric_y_left, "Speed", speed_label or "—")
    metric_y_left = draw_metric(metric_x_left, metric_y_left, "Course distance", f"{total_distance:.1f} km")
    metric_y_left = draw_metric(metric_x_left, metric_y_left, "Minute markers", minute_summary)

    metric_y_right = draw_metric(
        metric_x_right,
        metric_y_right,
        "Photo count",
        f"{len(photo_data)} total",
    )
    metric_y_right = draw_metric(
        metric_x_right,
        metric_y_right,
        "Enroute vs control",
        f"{enroute_count} / {control_count}",
    )
    metric_y_right = draw_metric(
        metric_x_right,
        metric_y_right,
        "Warnings",
        f"{len(excluded_photos)} removed, {len(offroute_warnings)} flagged, {len(control_close)} control close",
    )

    y = min(metric_y_left, metric_y_right) - 6 * mm
    section_floor = map_box_top + 12 * mm
    if y < section_floor:
        y = section_floor

    waypoint_times = info.get("waypoint_times", {})
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Waypoint schedule")
    y -= 5 * mm
    c.setFont("Helvetica", 10)
    for name, _lat, _lon in ROUTE_POINTS:
        entry = waypoint_times.get(name) or {}
        time_txt = entry.get("time_str") or "—"
        c.drawString(margin + 6 * mm, y, f"{name:<3} {time_txt}")
        y -= 4.8 * mm
        if y < section_floor:
            break

    if y > section_floor:
        y -= 3 * mm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Leg breakdown")
        y -= 5 * mm
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(margin + 2 * mm, y, "Leg")
        c.drawString(margin + 32 * mm, y, "Distance (km)")
        c.drawString(margin + 74 * mm, y, "Bearing")
        c.setFillColor(colors.black)
        y -= 4 * mm
        for leg in info.get("legs", [])[:8]:
            c.drawString(margin + 2 * mm, y, f"{leg['from']}→{leg['to']}")
            c.drawRightString(margin + 64 * mm, y, f"{leg['length_km']:.2f}")
            c.drawRightString(margin + 92 * mm, y, f"{int(leg['bearing']):03d}°")
            y -= 4.5 * mm
            if y < section_floor:
                break

    notes: List[str] = []
    if excluded_photos:
        notes.append(f"{len(excluded_photos)} photos removed for TP spacing")
    if offroute_warnings:
        notes.append(
            f"{len(offroute_warnings)} photos flagged for >{PHOTO_MAX_LATERAL_DISTANCE_M:.0f} m offset"
        )
    if control_close:
        notes.append(
            f"{len(control_close)} control photos within {PHOTO_MIN_DISTANCE_AFTER_TP_NM:.1f} NM"
        )
    if not notes:
        notes.append("All spacing constraints satisfied")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, map_box_top + 6 * mm, "Notes")
    c.setFont("Helvetica", 10)
    note_y = map_box_top + 2 * mm
    for note in notes:
        c.drawString(margin + 4 * mm, note_y, f"• {note}")
        note_y -= 4.5 * mm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(map_box_left, map_box_top + 6 * mm, "Route footprint (schematic)")
    c.setLineWidth(0.6)
    c.setFillColor(colors.whitesmoke)
    c.rect(map_box_left, map_box_bottom, map_box_width, map_box_height, fill=1, stroke=0)
    c.setStrokeColor(colors.black)
    c.rect(map_box_left, map_box_bottom, map_box_width, map_box_height, fill=0, stroke=1)

    bounds = info.get("route_bounds")
    route_path = info.get("route_path", [])
    if bounds and route_path:
        route_width = max(1.0, bounds["max_x"] - bounds["min_x"])
        route_height = max(1.0, bounds["max_y"] - bounds["min_y"])
        pad = 4 * mm
        avail_w = max(1.0, map_box_width - pad * 2)
        avail_h = max(1.0, map_box_height - pad * 2)
        scale = min(avail_w / route_width, avail_h / route_height)
        draw_width = route_width * scale
        draw_height = route_height * scale
        origin_x = map_box_left + (map_box_width - draw_width) / 2.0
        origin_y = map_box_bottom + (map_box_height - draw_height) / 2.0

        # Optional background map
        map_image = info.get("map_image_path")
        if map_image and map_image.exists():
            try:
                img_reader = ImageReader(str(map_image))
                img_w, img_h = img_reader.getSize()
                img_scale = min(
                    1.0,
                    (map_box_width - pad) / img_w,
                    (map_box_height - pad) / img_h,
                )
                img_draw_w = img_w * img_scale
                img_draw_h = img_h * img_scale
                img_x = map_box_left + (map_box_width - img_draw_w) / 2.0
                img_y = map_box_bottom + (map_box_height - img_draw_h) / 2.0
                c.saveState()
                c.setFillColor(colors.white)
                c.rect(img_x, img_y, img_draw_w, img_draw_h, stroke=0, fill=1)
                c.drawImage(
                    img_reader,
                    img_x,
                    img_y,
                    width=img_draw_w,
                    height=img_draw_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                c.restoreState()
            except Exception as exc:
                print(f"Warning: summary map background failed: {exc}")

        start_color = colors.HexColor("#ffca28")
        finish_color = colors.HexColor("#66bb6a")
        route_color = colors.HexColor("#c62828")

        # Draw route polyline
        c.setStrokeColor(route_color)
        c.setLineWidth(1.3)
        prev = None
        for name, _lat, _lon, px, py in route_path:
            sx = origin_x + (px - bounds["min_x"]) * scale
            sy = origin_y + (py - bounds["min_y"]) * scale
            if prev is not None:
                c.line(prev[0], prev[1], sx, sy)
            prev = (sx, sy)

        # Draw markers
        c.setLineWidth(0.7)
        label_offset = 6.5
        for idx, (name, _lat, _lon, px, py) in enumerate(route_path):
            sx = origin_x + (px - bounds["min_x"]) * scale
            sy = origin_y + (py - bounds["min_y"]) * scale
            radius = 3.2
            if name == "SP":
                c.setFillColor(start_color)
                caption = "Start"
            elif name == "FP":
                c.setFillColor(finish_color)
                caption = "Finish"
            else:
                c.setFillColor(colors.white)
                caption = name
            c.setStrokeColor(route_color)
            c.circle(sx, sy, radius, stroke=1, fill=1)

            c.setFont("Helvetica", 7)
            c.setFillColor(colors.black)
            dy = label_offset if idx % 2 == 0 else -label_offset
            c.drawCentredString(sx, sy + dy, caption)

        # Legend
        legend_x = map_box_left + 5 * mm
        legend_y = map_box_bottom + map_box_height - 10 * mm
        c.setFont("Helvetica", 8)

        def legend_entry(y_pos: float, text: str, fill_color) -> float:
            c.setFillColor(fill_color)
            c.setStrokeColor(route_color)
            c.circle(legend_x, y_pos, 2.4, stroke=1, fill=1)
            c.setFillColor(colors.black)
            c.drawString(legend_x + 7, y_pos - 2, text)
            return y_pos - 9

        legend_y = legend_entry(legend_y, "Start", start_color)
        legend_y = legend_entry(legend_y, "Finish", finish_color)
        # Legend removed for cleaner layout

        c.setStrokeColor(colors.red)
        c.setLineWidth(0.8)
        c.rect(origin_x, origin_y, max(2.0, draw_width), max(2.0, draw_height), fill=0, stroke=1)

        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawRightString(
            map_box_left + map_box_width,
            map_box_bottom - 4 * mm,
            f"Content span: {route_width / mm:.1f} × {route_height / mm:.1f} mm",
        )

    c.showPage()
    c.save()


def generate_overlay(
    analysis_rows: List[dict],
    speed_mps: float,
    speed_label: str,
    variant: dict,
) -> None:
    variant_id = variant.get("id", "")
    variant_label = variant.get("label", variant_id or "default")
    if not PDF_PATH.exists():
        raise SystemExit(f"Missing base map PDF: {PDF_PATH}")

    # Determine map image size via PDF page box
    ensure_map_image()
    reader = PdfReader(str(PDF_PATH))
    page = reader.pages[0]
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)

    width_scale = page_width / BASE_WIDTH
    height_scale = page_height / BASE_HEIGHT
    style_scale = (width_scale + height_scale) / 2

    tp_radius_scale = variant.get("tp_radius_scale", TP_RADIUS_SCALE)
    route_width_scale = variant.get("route_width_scale", ROUTE_WIDTH_SCALE)
    photo_tick_scale = variant.get("photo_tick_half_scale", PHOTO_TICK_HALF_SCALE)
    photo_line_scale = variant.get("photo_line_width_scale", PHOTO_LINE_WIDTH_SCALE)
    heading_font_scale = variant.get("heading_font_scale", HEADING_FONT_SCALE)
    heading_offset_scale = variant.get("heading_offset_scale", HEADING_OFFSET_SCALE)
    minute_tick_scale = variant.get("minute_tick_half_scale", MINUTE_TICK_HALF_SCALE)
    minute_line_scale = variant.get("minute_line_width_scale", MINUTE_LINE_WIDTH_SCALE)
    minute_font_scale = variant.get("minute_label_font_scale", MINUTE_LABEL_FONT_SCALE)
    tp_time_font_scale = variant.get("tp_time_font_scale", TP_TIME_FONT_SCALE)

    route_line_width = max(0.4 * style_scale, route_width_scale * style_scale)
    tp_radius = tp_radius_scale * width_scale
    photo_cross_half = photo_tick_scale * style_scale
    photo_line_width = max(0.4 * style_scale, photo_line_scale * style_scale)
    tp_font = max(4.0, TP_FONT_SCALE * style_scale)
    photo_font = max(3.0, PHOTO_FONT_SCALE * style_scale)
    heading_font = max(4.0, heading_font_scale * style_scale)
    heading_offset = heading_offset_scale * style_scale
    minute_cross_half = minute_tick_scale * style_scale
    minute_line_width = max(0.4 * style_scale, minute_line_scale * style_scale)
    minute_font = max(3.0, minute_font_scale * style_scale)
    tp_time_font = max(minute_font, tp_time_font_scale * style_scale)

    affine = solve_affine(CONTROL_POINTS)

    route_xy = []
    coord_lookup = {}
    for name, lat, lon in ROUTE_POINTS:
        xb, yb = affine.project(lat, lon)
        xp, yp = preview_to_pdf(xb, yb, page_width, page_height)
        route_xy.append((name, lat, lon, xp, yp))
        coord_lookup[name] = (xp, yp)

    content_bounds = init_bounds(page_width, page_height)
    for _, _, _, xp, yp in route_xy:
        add_point_to_bounds(content_bounds, xp, yp)
        if DRAW_TURNPOINTS:
            add_box_to_bounds(content_bounds, (xp - tp_radius, yp - tp_radius, xp + tp_radius, yp + tp_radius))

    photo_data: List[dict] = []
    excluded_photos: List[Tuple[str, str, str, float]] = []
    offroute_warnings: List[Tuple[str, str, str, float]] = []
    control_close: List[Tuple[str, str, str, float]] = []
    min_distance_m = (PHOTO_MIN_DISTANCE_AFTER_TP_NM * NM_TO_METERS
                      if PHOTO_MIN_DISTANCE_AFTER_TP_NM else 0.0)

    for idx, row in enumerate(analysis_rows, start=1):
        lat = float(row["lat"])
        lon = float(row["lon"])
        xb, yb = affine.project(lat, lon)
        xp, yp = preview_to_pdf(xb, yb, page_width, page_height)
        leg_distance_m = float(row["leg_distance_m"])
        label = f"P{idx:02d}"
        photo_file = row["photo"]
        control_hint = row.get("control_hint")
        is_control = control_hint is not None
        lateral_m = float(row.get("lateral_m", 0.0))
        too_close = (
            PHOTO_MIN_DISTANCE_AFTER_TP_NM > 0
            and leg_distance_m < min_distance_m
        )
        too_far = (
            DRAW_PHOTO_DOTS
            and PHOTO_MAX_LATERAL_DISTANCE_M > 0
            and lateral_m > PHOTO_MAX_LATERAL_DISTANCE_M
        )

        if too_close and not is_control:
            excluded_photos.append((label, photo_file, row["leg"], leg_distance_m))
            continue
        if too_close and is_control:
            control_close.append((label, photo_file, row["leg"], leg_distance_m))
        if too_far:
            offroute_warnings.append((label, photo_file, row["leg"], lateral_m))

        photo_data.append(
            {
                "label": label,
                "photo": photo_file,
                "local_time": row["local_time"],
                "leg": row["leg"],
                "map_mm": float(row["map_mm_from_start"]),
                "fraction": float(row["fraction_on_leg"]),
                "leg_distance_m": leg_distance_m,
                "lat": lat,
                "lon": lon,
                "base_x": xb,
                "base_y": yb,
                "pdf_x": xp,
                "pdf_y": yp,
                "lateral_m": lateral_m,
                "is_control": is_control,
                "control_hint": control_hint,
            }
        )

    assign_handout_labels(photo_data)

    minute_markers: List[dict] = []
    placed_label_boxes: List[Tuple[float, float, float, float]] = []
    cropped_map_path: Path | None = None

    def register_label_box(box: Tuple[float, float, float, float]) -> None:
        placed_label_boxes.append(box)
        add_box_to_bounds(content_bounds, box)

    waypoint_times: dict[str, dict[str, float | str]] = {}

    suffix = variant.get("id", "")
    overlay_path = variant_path(OUTPUT_OVERLAY, suffix)
    merged_path = variant_path(OUTPUT_MERGED, suffix)
    legend_path = variant_path(OUTPUT_KEY, suffix)

    c = canvas.Canvas(str(overlay_path), pagesize=(page_width, page_height))

    c.saveState()
    c.setStrokeColorRGB(*ROUTE_COLOR)
    c.setLineWidth(route_line_width)
    leg_segments = []
    segment_lookup = {}
    min_draw_margin = route_line_width * 0.5
    for (name_a, lat_a, lon_a, x_a, y_a), (name_b, lat_b, lon_b, x_b, y_b) in zip(route_xy[:-1], route_xy[1:]):
        dx = x_b - x_a
        dy = y_b - y_a
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy = dx / dist, dy / dist
        trim = min(tp_radius, max(0.0, dist / 2.0 - min_draw_margin))
        start_x = x_a + ux * trim
        start_y = y_a + uy * trim
        end_x = x_b - ux * trim
        end_y = y_b - uy * trim
        seg_dx = end_x - start_x
        seg_dy = end_y - start_y
        if DRAW_ROUTE and math.hypot(seg_dx, seg_dy) > 0.2:
            c.line(start_x, start_y, end_x, end_y)
        add_line_to_bounds(content_bounds, start_x, start_y, end_x, end_y)
        heading = bearing_degrees(lat_a, lon_a, lat_b, lon_b)
        leg_key = f"{name_a}-{name_b}"
        segment_lookup[leg_key] = {
            "start": (start_x, start_y),
            "end": (end_x, end_y),
            "vector": (seg_dx, seg_dy),
            "raw_start": (x_a, y_a),
            "raw_end": (x_b, y_b),
        }
        leg_segments.append(
            {
                "from": name_a,
                "to": name_b,
                "start": (start_x, start_y),
                "end": (end_x, end_y),
                "vector": (seg_dx, seg_dy),
                "heading": heading,
                "raw_start": (x_a, y_a),
                "raw_end": (x_b, y_b),
            }
        )
    c.restoreState()

    meters_per_minute = speed_mps * 60.0
    total_distance = LEGS[-1]["cumulative_start"] + LEGS[-1]["length"]

    if DRAW_MINUTE_MARKERS:
        max_minute = int(math.floor(TAKEOFF_TO_SP_MIN + total_distance / meters_per_minute))
        for minute in range(0, max_minute + 1):
            dist_from_sp = (minute - TAKEOFF_TO_SP_MIN) * meters_per_minute
            if dist_from_sp < -1e-6 or dist_from_sp > total_distance + 1e-6:
                continue
            remaining = dist_from_sp
            for leg in LEGS:
                leg_length = leg["length"]
                if remaining <= leg_length + 1e-6:
                    ratio = 0.0 if leg_length == 0 else remaining / leg_length
                    lat = leg["from_lat"] + ratio * (leg["to_lat"] - leg["from_lat"])
                    lon = leg["from_lon"] + ratio * (leg["to_lon"] - leg["from_lon"])
                    xb, yb = affine.project(lat, lon)
                    xp, yp = preview_to_pdf(xb, yb, page_width, page_height)
                    minute_markers.append(
                        {
                            "minute": minute,
                            "leg": f"{leg['from_name']}-{leg['to_name']}",
                            "px": xp,
                            "py": yp,
                            "fraction": ratio,
                            "distance": dist_from_sp,
                        }
                    )
                    break
                remaining -= leg_length

    # waypoint crossing times (minutes from takeoff)
    for idx, (name, lat, lon) in enumerate(ROUTE_POINTS):
        if idx == 0:
            dist_from_sp = 0.0
        else:
            leg = LEGS[idx - 1]
            dist_from_sp = leg["cumulative_start"] + leg["length"]
        total_minutes = TAKEOFF_TO_SP_MIN + dist_from_sp / meters_per_minute
        total_seconds = int(round(total_minutes * 60))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        if seconds == 60:
            minutes += 1
            seconds = 0
        time_str = f"{minutes:02d}:{seconds:02d}"
        waypoint_times[name] = {
            "time_str": time_str,
            "minutes": total_minutes,
        }

    for idx, (name, lat, lon, x, y) in enumerate(route_xy):
        if DRAW_TURNPOINTS:
            c.saveState()
            if hasattr(c, "setFillAlpha"):
                c.setFillAlpha(0.0)
            c.setStrokeColorRGB(*TP_COLOR)
            c.setLineWidth(route_line_width)
            c.circle(x, y, tp_radius, stroke=1, fill=0)
            c.restoreState()

        if DRAW_TURNPOINTS:
            label_text = name
            time_text = waypoint_times[name]['time_str'] if name in waypoint_times else None
            if len(route_xy) == 1:
                vout = (1.0, 0.0)
                vin = (-1.0, 0.0)
            else:
                if idx < len(route_xy) - 1:
                    vout = (route_xy[idx + 1][3] - x, route_xy[idx + 1][4] - y)
                else:
                    vout = (x - route_xy[idx - 1][3], y - route_xy[idx - 1][4])
                if idx > 0:
                    vin = (x - route_xy[idx - 1][3], y - route_xy[idx - 1][4])
                else:
                    vin = (-vout[0], -vout[1])

            if vout == (0.0, 0.0):
                vout = (1.0, 0.0)
            vout_u = normalize(*vout)
            vin_u = normalize(*vin) if vin != (0.0, 0.0) else (-vout_u[0], -vout_u[1])

            sum_vec = (vin_u[0] + vout_u[0], vin_u[1] + vout_u[1])
            sum_len = math.hypot(sum_vec[0], sum_vec[1])
            if sum_len > 1e-6:
                exterior = (-sum_vec[0] / sum_len, -sum_vec[1] / sum_len)
            else:
                exterior = normalize(-vout_u[1], vout_u[0])
            if math.hypot(*exterior) < 1e-6:
                exterior = normalize(vout_u[1], -vout_u[0])

            label_margin = max(
                tp_font * TP_LABEL_OFFSET_FACTOR,
                route_line_width * 2.5,
                photo_cross_half * 2.5,
                minute_cross_half * 2.2,
            )
            base_radius = tp_radius + label_margin
            heading_angle = math.degrees(math.atan2(vin_u[1], vin_u[0])) - 90.0
            if idx == 0 and len(route_xy) > 1:
                first_vec = (route_xy[1][3] - x, route_xy[1][4] - y)
                heading_angle = math.degrees(math.atan2(first_vec[1], first_vec[0])) - 90.0

            radial_dir = exterior
            if math.hypot(*radial_dir) < 1e-6:
                radial_dir = (1.0, 0.0)
            radial_dir = normalize(*radial_dir)
            tangential_dir = normalize(-radial_dir[1], radial_dir[0])

            name_width = c.stringWidth(label_text, "Helvetica-Bold", tp_font)
            name_height = tp_font

            candidate_dirs: List[tuple[tuple[float, float], float]] = []
            for radial_weight, tangential_weight in (
                (1.0, 0.0),
                (0.95, 0.25),
                (0.95, -0.25),
                (0.85, 0.45),
                (0.85, -0.45),
            ):
                vec = (
                    radial_dir[0] * radial_weight + tangential_dir[0] * tangential_weight,
                    radial_dir[1] * radial_weight + tangential_dir[1] * tangential_weight,
                )
                if math.hypot(*vec) < 1e-6:
                    continue
                vec = normalize(*vec)
                if vec[0] * radial_dir[0] + vec[1] * radial_dir[1] <= 0.25:
                    continue
                candidate_dirs.append((vec, math.degrees(math.atan2(vec[1], vec[0]))))

            best_label = None
            for direction_vec, direction_angle in candidate_dirs:
                anchor_x = x + direction_vec[0] * base_radius
                anchor_y = y + direction_vec[1] * base_radius
                cand_x, cand_y, cand_box = adjust_label_position(
                    anchor_x,
                    anchor_y,
                    direction_vec[0],
                    direction_vec[1],
                    name_width,
                    name_height,
                    heading_angle,
                    placed_label_boxes,
                    fallback_angle=direction_angle,
                )
                distance = math.hypot(cand_x - x, cand_y - y)
                if best_label is None or distance < best_label[0]:
                    best_label = (distance, cand_x, cand_y, cand_box, direction_vec)

            if best_label is None:
                name_x, name_y, name_box = adjust_label_position(
                    x + radial_dir[0] * base_radius,
                    y + radial_dir[1] * base_radius,
                    radial_dir[0],
                    radial_dir[1],
                    name_width,
                    name_height,
                    heading_angle,
                    placed_label_boxes,
                )
                chosen_dir = radial_dir
            else:
                _, name_x, name_y, name_box, chosen_dir = best_label
            register_label_box(name_box)

            time_x = time_y = None
            time_box = None
            if time_text:
                time_width = c.stringWidth(time_text, "Helvetica-Bold", tp_time_font)
                time_height = tp_time_font
                time_margin = max(
                    tp_time_font * 0.6,
                    route_line_width * 1.2,
                    photo_cross_half,
                    minute_cross_half,
                )
                time_radius = tp_radius + label_margin + time_margin
                base_tangent = normalize(-chosen_dir[1], chosen_dir[0])

                time_candidates: List[tuple[float, float, tuple[float, float, float, float]]] = []
                for radial_w, tangential_w in (
                    (0.35, 0.95),
                    (0.35, -0.95),
                    (0.55, 0.75),
                    (0.55, -0.75),
                    (0.2, 1.0),
                    (0.2, -1.0),
                ):
                    vec = (
                        chosen_dir[0] * radial_w + tangential_dir[0] * tangential_w,
                        chosen_dir[1] * radial_w + tangential_dir[1] * tangential_w,
                    )
                    if math.hypot(*vec) < 1e-6:
                        continue
                    vec = normalize(*vec)
                    if vec[0] * chosen_dir[0] + vec[1] * chosen_dir[1] <= 0.2:
                        continue
                    anchor_x = x + vec[0] * time_radius
                    anchor_y = y + vec[1] * time_radius
                    cand_x, cand_y, cand_box = adjust_label_position(
                        anchor_x,
                        anchor_y,
                        vec[0],
                        vec[1],
                        time_width,
                        time_height,
                        heading_angle,
                        placed_label_boxes,
                        fallback_angle=math.degrees(math.atan2(vec[1], vec[0])),
                    )
                    dist = math.hypot(cand_x - x, cand_y - y)
                    if dist < tp_radius + route_line_width * 0.6:
                        continue
                    separation = math.hypot(cand_x - name_x, cand_y - name_y)
                    time_candidates.append((dist, separation, (cand_x, cand_y, cand_box)))

                if time_candidates:
                    time_candidates.sort()
                    _, _, best_box = time_candidates[0]
                    time_x, time_y, time_box = best_box
                    register_label_box(time_box)
                else:
                    time_anchor_x = name_x + base_tangent[0] * (tp_time_font * TP_TIME_SWEEP_FACTOR)
                    time_anchor_y = name_y + base_tangent[1] * (tp_time_font * TP_TIME_SWEEP_FACTOR)
                    time_x, time_y, time_box = adjust_label_position(
                        time_anchor_x,
                        time_anchor_y,
                        base_tangent[0],
                        base_tangent[1],
                        time_width,
                        time_height,
                        heading_angle,
                        placed_label_boxes,
                    )
                    register_label_box(time_box)

            c.saveState()
            c.translate(name_x, name_y)
            c.rotate(heading_angle)
            c.setFont("Helvetica-Bold", tp_font)
            c.setFillColorRGB(*TP_LABEL_COLOR)
            c.drawString(0, 0, label_text)
            c.restoreState()

            if DRAW_MINUTE_MARKERS and time_text and time_x is not None and time_y is not None:
                c.saveState()
                c.translate(time_x, time_y)
                c.rotate(heading_angle)
                c.setFont("Helvetica-Bold", tp_time_font)
                c.setFillColorRGB(*MINUTE_LABEL_COLOR)
                c.drawString(0, 0, time_text)
                c.restoreState()

    if DRAW_PHOTO_DOTS:
        dot_radius = max(1.2, PHOTO_DOT_RADIUS_SCALE * style_scale)
        for photo in photo_data:
            if photo.get("is_control") and not DRAW_CONTROL_PHOTO_MARKERS:
                continue
            c.saveState()
            c.setFillColorRGB(*PHOTO_COLOR)
            c.circle(photo["pdf_x"], photo["pdf_y"], dot_radius, stroke=0, fill=1)
            c.restoreState()
            add_point_to_bounds(content_bounds, photo["pdf_x"], photo["pdf_y"])

    if DRAW_PHOTO_MARKERS:
        for photo in photo_data:
            if photo.get("is_control") and not DRAW_CONTROL_PHOTO_MARKERS:
                continue
            leg = photo["leg"]
            start_name, end_name = leg.split("-")
            start_coord = coord_lookup.get(start_name)
            end_coord = coord_lookup.get(end_name)
            if not start_coord or not end_coord:
                continue

            segment = segment_lookup.get(leg)
            if segment is not None:
                raw_start = segment["raw_start"]
                raw_end = segment["raw_end"]
                seg_start = segment["start"]
                seg_end = segment["end"]
            else:
                raw_start = start_coord
                raw_end = end_coord
                seg_start = start_coord
                seg_end = end_coord

            fraction = max(0.0, min(1.0, photo["fraction"]))
            raw_dx = raw_end[0] - raw_start[0]
            raw_dy = raw_end[1] - raw_start[1]
            x_center = raw_start[0] + raw_dx * fraction
            y_center = raw_start[1] + raw_dy * fraction

            inside_tp = False
            for _, _, _, tp_x, tp_y in route_xy:
                if point_distance(x_center, y_center, tp_x, tp_y) <= tp_radius:
                    inside_tp = True
                    break
            if inside_tp:
                continue

            dx = seg_end[0] - seg_start[0]
            dy = seg_end[1] - seg_start[1]
            if dx == 0 and dy == 0:
                continue
            ux, uy = normalize(dx, dy)
            px_vec, py_vec = -uy, ux

            line_x1 = x_center - px_vec * photo_cross_half
            line_y1 = y_center - py_vec * photo_cross_half
            line_x2 = x_center + px_vec * photo_cross_half
            line_y2 = y_center + py_vec * photo_cross_half

            c.saveState()
            c.setStrokeColorRGB(*PHOTO_COLOR)
            c.setLineWidth(photo_line_width)
            c.line(line_x1, line_y1, line_x2, line_y2)
            c.restoreState()
            add_line_to_bounds(content_bounds, line_x1, line_y1, line_x2, line_y2)

            label_offset = photo_cross_half * PHOTO_LABEL_OFFSET_MULTIPLIER
            label_x = x_center + px_vec * label_offset
            label_y = y_center + py_vec * label_offset
            handout_text = photo.get("handout_label", photo["label"])
            label_width = c.stringWidth(handout_text, "Helvetica-Bold", photo_font)
            label_height = photo_font
            label_x, label_y, label_box = adjust_label_position(
                label_x,
                label_y,
                px_vec,
                py_vec,
                label_width,
                label_height,
                0.0,
                placed_label_boxes,
                fallback_angle=math.degrees(math.atan2(py_vec, px_vec)) + 90.0,
            )
            c.saveState()
            c.setFont("Helvetica-Bold", photo_font)
            c.setFillColorRGB(*PHOTO_LABEL_HALO)
            c.drawString(label_x, label_y, handout_text)
            c.setFillColorRGB(*PHOTO_LABEL_COLOR)
            c.drawString(label_x, label_y, handout_text)
            c.restoreState()
            register_label_box(label_box)

    if DRAW_MINUTE_MARKERS:
        for marker in minute_markers:
            leg = marker["leg"]
            segment = segment_lookup.get(leg)
            if segment is not None:
                seg_start = segment["start"]
                seg_end = segment["end"]
            else:
                start_name, end_name = leg.split("-")
                seg_start = coord_lookup.get(start_name)
                seg_end = coord_lookup.get(end_name)
            if seg_start is None or seg_end is None:
                continue

            x_center = marker["px"]
            y_center = marker["py"]
            inside_tp = False
            for _, _, _, tp_x, tp_y in route_xy:
                if point_distance(x_center, y_center, tp_x, tp_y) <= tp_radius:
                    inside_tp = True
                    break
            if inside_tp:
                continue

            dx = seg_end[0] - seg_start[0]
            dy = seg_end[1] - seg_start[1]
            if dx == 0 and dy == 0:
                continue
            ux, uy = normalize(dx, dy)
            px_vec, py_vec = -uy, ux

            line_x1 = x_center - px_vec * minute_cross_half
            line_y1 = y_center - py_vec * minute_cross_half
            line_x2 = x_center + px_vec * minute_cross_half
            line_y2 = y_center + py_vec * minute_cross_half

            c.saveState()
            c.setStrokeColorRGB(*MINUTE_MARKER_COLOR)
            c.setLineWidth(minute_line_width)
            c.line(line_x1, line_y1, line_x2, line_y2)
            c.restoreState()
            add_line_to_bounds(content_bounds, line_x1, line_y1, line_x2, line_y2)

            label_text = f"{MINUTE_LABEL_PREFIX}{marker['minute']:02d}" if MINUTE_LABEL_PREFIX else f"{marker['minute']:02d}"
            label_offset = minute_cross_half * MINUTE_LABEL_OFFSET_MULTIPLIER
            label_x = x_center + px_vec * label_offset
            label_y = y_center + py_vec * label_offset
            angle_deg = math.degrees(math.atan2(dy, dx)) - 90.0
            label_width = c.stringWidth(label_text, "Helvetica-Bold", minute_font)
            label_height = minute_font
            label_x, label_y, label_box = adjust_label_position(
                label_x,
                label_y,
                px_vec,
                py_vec,
                label_width,
                label_height,
                angle_deg,
                placed_label_boxes,
                fallback_angle=math.degrees(math.atan2(py_vec, px_vec)) + 90.0,
            )
            c.saveState()
            c.translate(label_x, label_y)
            c.rotate(angle_deg)
            c.setFont("Helvetica-Bold", minute_font)
            c.setFillColorRGB(*MINUTE_LABEL_HALO)
            c.drawString(0, 0, label_text)
            c.setFillColorRGB(*MINUTE_LABEL_COLOR)
            c.drawString(0, 0, label_text)
            c.restoreState()
            register_label_box(label_box)

    if DRAW_HEADINGS:
        for leg in leg_segments:
            raw_start_x, raw_start_y = leg["raw_start"]
            raw_end_x, raw_end_y = leg["raw_end"]
            dx_raw = raw_end_x - raw_start_x
            dy_raw = raw_end_y - raw_start_y
            if dx_raw == 0 and dy_raw == 0:
                continue
            ux, uy = normalize(dx_raw, dy_raw)
            px_vec, py_vec = -uy, ux
            mid_x = (raw_start_x + raw_end_x) / 2.0
            mid_y = (raw_start_y + raw_end_y) / 2.0
            text_x = mid_x + px_vec * heading_offset
            text_y = mid_y + py_vec * heading_offset
            angle_deg = math.degrees(math.atan2(dy_raw, dx_raw)) - 90.0
            heading_deg = int(float(f"{leg['heading']:.0f}"))
            heading_text = f"{heading_deg:03d}°"
            heading_width = c.stringWidth(heading_text, "Helvetica-Bold", heading_font)
            heading_height = heading_font
            text_x, text_y, heading_box = adjust_label_position(
                text_x,
                text_y,
                px_vec,
                py_vec,
                heading_width,
                heading_height,
                angle_deg,
                placed_label_boxes,
                fallback_angle=math.degrees(math.atan2(py_vec, px_vec)) + 90.0,
            )

            c.saveState()
            c.translate(text_x, text_y)
            c.rotate(angle_deg)
            c.setFont("Helvetica-Bold", heading_font)
            c.setFillColorRGB(*HEADING_COLOR)
            c.drawString(0, 0, heading_text)
            c.restoreState()
            register_label_box(heading_box)

    c.showPage()
    c.save()

    overlay_reader = PdfReader(str(overlay_path))
    writer = PdfWriter()
    base_page = reader.pages[0]
    base_page.merge_page(overlay_reader.pages[0])
    writer.add_page(base_page)

    with merged_path.open("wb") as fh:
        writer.write(fh)

    cropped_map_path = create_cropped_map_pdf(
        merged_path,
        content_bounds,
        page_width,
        page_height,
        suffix,
    )

    with legend_path.open("w", newline="") as fh:
        writer_csv = csv.writer(fh)
        writer_csv.writerow(["label", "photo_file", "local_time", "leg", "map_mm_from_leg_start"])
        for photo in photo_data:
            writer_csv.writerow(
                [
                    photo["label"],
                    photo["photo"],
                    photo["local_time"],
                    photo["leg"],
                    f"{photo['map_mm']:.2f}",
                ]
            )

    print(f"Overlay saved to {overlay_path}")
    print(f"Merged map saved to {merged_path}")
    print(f"Photo legend saved to {legend_path}")
    if cropped_map_path:
        print(f"Cropped map saved to {cropped_map_path}")

    print(
        f"\n[{variant_label}] Speed setting: {speed_label} (SP offset {TAKEOFF_TO_SP_MIN:.1f} min)"
    )

    # Summary output
    total_distance_km = sum(leg["length"] for leg in LEGS) / 1000.0
    print(f"\n[{variant_label}] Route Legs:")
    for leg in LEGS:
        bearing = bearing_degrees(leg["from_lat"], leg["from_lon"], leg["to_lat"], leg["to_lon"])
        bearing_deg = int(float(f"{bearing:.0f}"))
        leg_km = leg["length"] / 1000.0
        leg_mm = leg["length"] / METERS_PER_MM
        print(
            f"  {leg['from_name']} -> {leg['to_name']}: "
            f"{leg_km:.2f} km ({leg_mm:.1f} mm), bearing {bearing_deg:03d}°"
        )
    total_mm = total_distance_km * 1000.0 / METERS_PER_MM
    print(f"  Total course distance: {total_distance_km:.2f} km ({total_mm:.1f} mm)")

    if DRAW_MINUTE_MARKERS and minute_markers:
        first_minute = min(m["minute"] for m in minute_markers)
        last_minute = max(m["minute"] for m in minute_markers)
        print(
            f"[{variant_label}] Minute markers: {len(minute_markers)} placed for minutes {first_minute}–{last_minute}"
            f" (interval 1 min)"
        )
    elif DRAW_MINUTE_MARKERS:
        print(f"[{variant_label}] Minute markers: none (route shorter than SP offset)")
    else:
        print(f"[{variant_label}] Minute markers: disabled")

    print(f"\n[{variant_label}] Waypoint times from takeoff:")
    for name, _, _ in ROUTE_POINTS:
        time_info = waypoint_times.get(name)
        time_str = time_info["time_str"] if time_info else "N/A"
        print(f"  {name:<4} {time_str}")

    photo_table = [
        (
            p.get("handout_label", ""),
            p["label"],
            p["photo"],
            p["leg"],
            p["map_mm"],
        )
        for p in photo_data
    ]

    print(f"\n[{variant_label}] Photo Index Table:")
    print("  Letter | Label  | Photo File                              | Leg        | Map mm")
    print("  ------ | ------ | --------------------------------------- | ---------- | ------")
    for letter, label, photo, leg_name, map_mm in photo_table:
        print(f"  {letter:<6}| {label:<6}| {photo:<39} | {leg_name:<10} | {map_mm:6.2f}")

    if excluded_photos:
        print(
            f"\n[{variant_label}] WARNING: Photos closer than {PHOTO_MIN_DISTANCE_AFTER_TP_NM:.1f} NM after TP (removed):"
        )
        for label, fname, leg_name, dist_m in excluded_photos:
            print(
                f"  {label} ({fname}) on {leg_name} at {dist_m/1000:.2f} km (~{dist_m/NM_TO_METERS:.2f} NM)"
            )
    elif PHOTO_MIN_DISTANCE_AFTER_TP_NM > 0:
        print(
            f"\nAll photos satisfy the {PHOTO_MIN_DISTANCE_AFTER_TP_NM:.1f} NM minimum after TP."
        )

    if offroute_warnings:
        print(
            f"[{variant_label}] WARNING: Photos farther than {PHOTO_MAX_LATERAL_DISTANCE_M:.0f} m from route:"
        )
        for label, fname, leg_name, lateral_m in offroute_warnings:
            print(f"  {label} ({fname}) on {leg_name} at {lateral_m:.0f} m lateral offset")

    if control_close:
        print(
            f"[{variant_label}] Note: control photos within {PHOTO_MIN_DISTANCE_AFTER_TP_NM:.1f} NM retained:"
        )
        for label, fname, leg_name, dist_m in control_close:
            print(
                f"  {label} ({fname}) on {leg_name} at {dist_m/1000:.2f} km (~{dist_m/NM_TO_METERS:.2f} NM)"
            )

    return {
        "photo_data": photo_data,
        "cropped_map_path": cropped_map_path,
        "route_bounds": content_bounds,
        "page_size": (page_width, page_height),
        "route_path": route_xy,
        "variant_label": variant_label,
        "waypoint_times": waypoint_times,
        "speed_label": speed_label,
        "total_distance_km": total_distance_km,
        "minute_markers": minute_markers,
        "excluded_photos": excluded_photos,
        "offroute_warnings": offroute_warnings,
        "control_close": control_close,
        "legs": [
            {
                "from": leg["from_name"],
                "to": leg["to_name"],
                "length_km": leg["length"] / 1000.0,
                "bearing": bearing_degrees(
                    leg["from_lat"],
                    leg["from_lon"],
                    leg["to_lat"],
                    leg["to_lon"],
                ),
            }
            for leg in LEGS
        ],
        "map_image_path": MAP_IMAGE if MAP_IMAGE.exists() else None,
    }


def generate_handout_pdf(
    photo_data: List[dict],
    cropped_map_path: Path | None = None,
    summary_info: dict | None = None,
) -> None:
    if not photo_data:
        print("No photos available for handout.")
        return

    assign_handout_labels(photo_data)
    control_photos = [p for p in photo_data if p.get("is_control")]
    enroute_photos = [p for p in photo_data if not p.get("is_control")]
    ordered_enroute = ordered_enroute_photos(enroute_photos)

    def split_by_handout_tp(items: List[dict]) -> tuple[List[dict], List[dict]]:
        before: List[dict] = []
        after: List[dict] = []
        split_idx = ROUTE_POINT_INDEX.get(HANDOUT_SPLIT_TP)
        for photo in items:
            leg = photo.get("leg", "")
            if "-" in leg:
                _, leg_to = leg.split("-", 1)
            else:
                leg_to = leg
            dest_idx = ROUTE_POINT_INDEX.get(leg_to)
            if split_idx is None or dest_idx is None or dest_idx <= split_idx:
                before.append(photo)
            else:
                after.append(photo)
        return before, after

    enroute_before, enroute_after = split_by_handout_tp(ordered_enroute)

    page_width, page_height = A4
    c = canvas.Canvas(str(HANDOUT_PDF), pagesize=A4)
    margin_x = 6 * mm
    margin_y = 6 * mm
    columns = 2
    rows = 2
    inner_margin = 1 * mm
    slot_width = (page_width - margin_x * 2) / columns
    slot_height = (page_height - margin_y * 2) / rows
    per_page = columns * rows

    def draw_section(title: str, photos: List[dict], first_page: bool) -> bool:
        if not photos:
            return first_page
        title_y = page_height - margin_y - 2 * mm
        if not first_page:
            c.showPage()
        def draw_heading() -> None:
            c.saveState()
            c.setFont("Helvetica-Bold", 16)
            heading_x = page_width - margin_x
            heading_y = margin_y + 4 * mm
            c.translate(heading_x, heading_y)
            c.rotate(90)
            c.drawString(0, 0, title)
            c.restoreState()

        draw_heading()
        for idx, photo in enumerate(photos):
            slot = idx % per_page
            if slot == 0:
                if idx != 0:
                    c.showPage()
                    draw_heading()

            col = slot % columns
            row = slot // columns
            origin_x = margin_x + col * slot_width
            origin_y = page_height - margin_y - (row + 1) * slot_height

            photo_path = Path(photo["photo"])
            if not photo_path.is_absolute():
                photo_path = Path.cwd() / photo_path
            try:
                if Image is not None:
                    with Image.open(photo_path) as pil_img:
                        pil_img = pil_img.convert("RGB")
                        img_w, img_h = pil_img.size

                        available_photo_width = slot_width - inner_margin * 2
                        available_photo_height = slot_height - inner_margin * 2
                        photo_scale = min(available_photo_width / img_h, available_photo_height / img_w)
                        photo_width = img_h * photo_scale
                        photo_height = img_w * photo_scale
                        photo_x = origin_x + (slot_width - photo_width) / 2
                        photo_y = origin_y + (slot_height - photo_height) / 2

                        target_px_w = max(1, int(photo_height / 72.0 * HANDOUT_PHOTO_DPI))
                        target_px_h = max(1, int(photo_width / 72.0 * HANDOUT_PHOTO_DPI))
                        pil_copy = pil_img.copy()
                        pil_copy.thumbnail((target_px_w, target_px_h), Image.LANCZOS)
                        buffer = BytesIO()
                        pil_copy.save(buffer, format="JPEG", quality=85, optimize=True)
                        buffer.seek(0)
                        image_reader = ImageReader(buffer)
                else:
                    image_reader = ImageReader(str(photo_path))
                    img_w, img_h = image_reader.getSize()

                    available_photo_width = slot_width - inner_margin * 2
                    available_photo_height = slot_height - inner_margin * 2
                    photo_scale = min(available_photo_width / img_h, available_photo_height / img_w)
                    photo_width = img_h * photo_scale
                    photo_height = img_w * photo_scale
                    photo_x = origin_x + (slot_width - photo_width) / 2
                    photo_y = origin_y + (slot_height - photo_height) / 2
            except Exception as exc:
                print(f"Warning: unable to load {photo_path}: {exc}")
                continue

            c.saveState()
            c.translate(photo_x + photo_width / 2, photo_y + photo_height / 2)
            c.rotate(90)
            c.drawImage(
                image_reader,
                -photo_height / 2,
                -photo_width / 2,
                width=photo_height,
                height=photo_width,
                preserveAspectRatio=True,
                mask="auto",
            )
            c.restoreState()

            letter = photo.get("handout_label", "")
            if letter:
                letter_font = max(36.0, min(photo_width, photo_height) * HANDOUT_LETTER_SCALE)
                c.setFont("Helvetica-Bold", letter_font)
                c.setFillColor(colors.red)
                letter_margin = inner_margin + letter_font * 0.1
                letter_width = c.stringWidth(letter, "Helvetica-Bold", letter_font)
                anchor_x = photo_x + letter_margin + letter_font * 0.6
                anchor_y = photo_y + photo_height - letter_margin - letter_width - letter_font * 0.15
                anchor_y = max(anchor_y, photo_y + letter_margin)
                c.saveState()
                c.translate(anchor_x, anchor_y)
                c.rotate(90)
                c.drawString(0, 0, letter)
                c.restoreState()
        return False

    first_page = True
    if control_photos:
        first_page = draw_section("Control Point Photos", control_photos, first_page)
    if enroute_before:
        first_page = draw_section(f"Photos before {HANDOUT_SPLIT_TP}", enroute_before, first_page)
    if enroute_after:
        first_page = draw_section(f"Photos after {HANDOUT_SPLIT_TP}", enroute_after, first_page)

    def append_pdfs(base_path: Path, append_paths: List[Path]) -> None:
        try:
            base_reader = PdfReader(str(base_path))
            writer = PdfWriter()
            for page in base_reader.pages:
                writer.add_page(page)
            for extra in append_paths:
                extra_reader = PdfReader(str(extra))
                for page in extra_reader.pages:
                    writer.add_page(page)
            with base_path.open("wb") as fh:
                writer.write(fh)
        except Exception as exc:
            print(f"Warning: unable to append supplemental pages to handout: {exc}")

    c.save()
    print(f"Handout saved to {HANDOUT_PDF}")

    extras: List[Path] = []
    cleanup: List[Path] = []

    if HANDOUT_INCLUDE_SUMMARY and summary_info:
        summary_page = HANDOUT_PDF.with_name("photo_handout_summary.pdf")
        try:
            generate_summary_page(summary_page, summary_info)
            extras.append(summary_page)
            cleanup.append(summary_page)
        except Exception as exc:
            print(f"Warning: unable to generate summary page: {exc}")

    if cropped_map_path and cropped_map_path.exists():
        extras.append(cropped_map_path)
        print(f"Queued cropped map from {cropped_map_path} for handout append")

    if extras:
        append_pdfs(HANDOUT_PDF, extras)

    for tmp in cleanup:
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    _, _, speed_mps, speed_label = parse_speed_arg(getattr(args, "speed", None))
    analysis = analyse_photos()

    handout_payload = None
    for variant in STYLE_VARIANTS:
        variant_label = variant.get("label", variant.get("id", "route"))
        print(f"\n=== Generating variant: {variant_label} ===")
        result = generate_overlay(analysis, speed_mps, speed_label, variant)
        if handout_payload is None:
            handout_payload = result

    if handout_payload:
        generate_handout_pdf(
            handout_payload["photo_data"],
            handout_payload.get("cropped_map_path"),
            summary_info=handout_payload,
        )
