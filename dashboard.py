"""The session's results: save them as JSON, and build the HTML dashboard.

save_dashboard_data() writes <video>.dashboard.json at the end of a run
(run_pipeline.py calls it). build_dashboard() turns that JSON into one
self-contained .html file -- map image and data embedded, no external
libraries or network access -- so it opens offline or can be sent as a
single attachment. To rebuild a dashboard from an earlier session:

    python dashboard.py "data/<run folder>/dashboard.json" [map.png] [-o out.html]

It shows: summary numbers, the map with every room and symbol colored by
visited / brushed / missed plus each hand's path (scrubbable over time), a
table of every room and symbol, an enter/exit timeline (visits >= 1 s; plus
rows for the paper itself and for how reliably the crop was tracked), and
the order things were visited in.

Coordinates are the PNG's own pixels. Dashboard JSON from before the
refactor (tracked on the PNG rotated 180 degrees) is rotated back on load.
"""

import argparse
import base64
import json
import os

import cv2

from edge_tracing import load_image
from finger_tracking import PATH_BREAK_JUMP, PATH_MARGIN

GOOD_SOURCES = {"found", "lines", "edges", "edge", "PNG", "recovered", "initial", "manual"}
WEAK_SOURCES = {"flow", "held"}


def save_dashboard_data(path, tracker, timing, ref, video_path, fps, duration_ms):
    """Write the session's results JSON and return it as a dict."""
    timing.finish()
    log = tracker.session_log
    data = {
        "video": os.path.basename(video_path),
        "map": os.path.basename(ref.path),
        "map_path": os.path.abspath(ref.path),
        "coordinate_space": "png",
        "ref_w": ref.w, "ref_h": ref.h,
        "fps": fps, "duration_ms": duration_ms,
        "frames": len(log),
        "frames_with_hand": sum(1 for r in log if r["Left"] or r["Right"]),
        "min_visit_ms": 1000,
        "paper": timing.paper.summary(),
        "boxes": timing.summaries(),
        "sequence": timing.sequence(),
        "page_source": [r["page_source"] for r in log],
        "path": {hand: _path(log, hand, ref) for hand in ("Left", "Right")},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    rooms = [b for b in data["boxes"].values() if b["type"] == "room"]
    paper = data["paper"]
    print(f"Saved {path}")
    print(f"  on paper: {paper['visits']} visit(s), {paper['total_ms'] / 1000:.1f}s total")
    print(f"  rooms visited (>= 1 s): {sum(r['status'] == 'visited' for r in rooms)}/{len(rooms)}; "
          f"missed entirely: {[n for n, b in data['boxes'].items() if b['type'] == 'room' and b['missed']] or 'none'}")
    for name, b in sorted(data["boxes"].items(), key=lambda kv: -kv[1]["total_ms"]):
        print(f"  [{b['type']}] {name}: {b['status']}, {b['visits']} visit(s), "
              f"{b['total_ms'] / 1000:.1f}s total")
    return data


def _path(log, hand, ref):
    """One hand's timestamped path; None = a gap (hand not seen, too far off
    the map, or a jump too big to be real movement -- same rules as the
    trails drawn on finger_paths.png)."""
    out, last = [], None
    mx, my = PATH_MARGIN * ref.w, PATH_MARGIN * ref.h
    for r in log:
        tip = r[hand]
        if tip is None or not (-mx <= tip["map_x"] < ref.w + mx and -my <= tip["map_y"] < ref.h + my):
            out.append(None)
            last = None
            continue
        if last is not None and ((tip["map_x"] - last[0]) ** 2 +
                                 (tip["map_y"] - last[1]) ** 2) ** 0.5 > PATH_BREAK_JUMP:
            out.append(None)
        out.append({"x": tip["map_x"], "y": tip["map_y"], "t_ms": r["t_ms"]})
        last = (tip["map_x"], tip["map_y"])
    return out


def _prepare(data):
    """Compact the data for embedding; bring legacy coordinates into PNG space."""
    w, h = data["ref_w"], data["ref_h"]
    legacy = data.get("coordinate_space", "").startswith("reference PNG rotated")

    def fix(x, y):
        return ((w - 1) - x, (h - 1) - y) if legacy else (x, y)

    def entry(name, b):
        if "polygon" in b:
            poly = [list(fix(x, y)) for x, y in b["polygon"]]
        else:
            x0, y0, x1, y1 = b.get("bbox") or b["box_px"]
            poly = [list(fix(x, y)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        visits = b.get("visits", 0)
        touched = b.get("touched", b.get("entered", visits > 0))
        return {
            "name": name, "type": b.get("type", "symbol"), "room": b.get("room"),
            "status": b.get("status") or ("visited" if visits else "brushed" if touched else "missed"),
            "poly": [[round(x, 1), round(y, 1)] for x, y in poly],
            "visits": visits, "brushes": b.get("brushes", 0),
            "total_ms": b.get("total_ms", 0), "longest_ms": b.get("longest_ms", 0),
            "first_enter_ms": b.get("first_enter_ms"),
            "events": b.get("events", []),
        }

    boxes = [entry(name, b) for name, b in data.get("boxes", {}).items()]
    paper = entry("On paper", data["paper"]) if data.get("paper") else None

    paths = {}
    for hand, points in data.get("path", {}).items():
        out = []
        for p in points:
            if p is None:
                if out and out[-1] is not None:
                    out.append(None)
                continue
            x, y = fix(p["x"], p["y"])
            out.append([p["t_ms"], round(x, 1), round(y, 1)])
        paths[hand] = out

    # page_source per frame -> runs of [start_ms, end_ms, quality]
    runs = []
    fps = data.get("fps") or 30.0
    for i, src in enumerate(data.get("page_source") or []):
        quality = ("good" if src in GOOD_SOURCES else
                   "weak" if src in WEAK_SOURCES else "lost")
        t0 = round(1000.0 * i / fps)
        t1 = round(1000.0 * (i + 1) / fps)
        if runs and runs[-1][2] == quality:
            runs[-1][1] = t1
        else:
            runs.append([t0, t1, quality])

    duration = data.get("duration_ms") or max(
        [pt[0] for pts in paths.values() for pt in pts if pt] or [0])
    return {
        "video": data.get("video"), "map": data.get("map"),
        "w": w, "h": h, "duration_ms": duration,
        "frames": data.get("frames"), "frames_with_hand": data.get("frames_with_hand"),
        "min_visit_ms": data.get("min_visit_ms", 1000),
        "boxes": boxes, "paper": paper, "sequence": data.get("sequence", []),
        "paths": paths, "tracking": runs,
    }


def build_dashboard(data, map_path, out_path):
    """Write the HTML dashboard for `data` (the dict save_dashboard_data
    returns, or a loaded .dashboard.json) to out_path."""
    img = load_image(map_path)
    if img is None:
        raise FileNotFoundError(map_path)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Could not encode {map_path}")
    payload = _prepare(data)
    payload["image"] = "data:image/png;base64," + base64.b64encode(png).decode()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    title = os.path.splitext(payload["video"] or "Session")[0]
    html = (_TEMPLATE.replace("__TITLE__", _escape(title))
                     .replace("__DATA__", blob))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def _escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Session dashboard</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --right: #2a78d6; --left: #eb6834; --both: #1baf7a;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  --good-ink: #006300; --warning-ink: #8a5a00; --critical-ink: #b02a2a;
  --hover: rgba(11,11,11,0.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --right: #3987e5; --left: #d95926; --both: #199e70;
    --good-ink: #0ca30c; --warning-ink: #fab219; --critical-ink: #e66767;
    --hover: rgba(255,255,255,0.06);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255,255,255,0.10);
  --right: #3987e5; --left: #d95926; --both: #199e70;
  --good-ink: #0ca30c; --warning-ink: #fab219; --critical-ink: #e66767;
  --hover: rgba(255,255,255,0.06);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
header { margin-bottom: 20px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 15px; margin: 0 0 12px; font-weight: 600; }
.sub { color: var(--ink-2); margin: 0; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; min-width: 0;
}
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin-bottom: 16px; }
.tile .label { color: var(--ink-2); font-size: 12.5px; }
.tile .value { font-size: 26px; font-weight: 650; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px; margin-bottom: 16px; }
@media (max-width: 980px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px;
  margin-bottom: 10px; color: var(--ink-2); font-size: 13px; }
.controls label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
.swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.scrub { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.scrub input { flex: 1; min-width: 0; }
.scrub output { font-variant-numeric: tabular-nums; color: var(--ink-2);
  min-width: 92px; text-align: right; font-size: 13px; }
button {
  font: inherit; font-size: 13px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--axis); border-radius: 8px; padding: 4px 12px; cursor: pointer;
}
button:hover { background: var(--hover); }
.mapwrap { position: relative; width: 100%; }
.mapwrap svg { display: block; width: 100%; height: auto; border-radius: 8px; }
.boxlabel { fill: #0b0b0b; paint-order: stroke; stroke: #ffffff;
  stroke-linejoin: round; font-weight: 600; pointer-events: none; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 500; font-size: 12px;
  border-bottom: 1px solid var(--grid); padding: 6px 8px; white-space: nowrap; }
td { border-bottom: 1px solid var(--grid); padding: 6px 8px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr { cursor: default; }
tbody tr:hover, tbody tr.hl { background: var(--hover); }
.tablewrap { overflow-x: auto; max-height: 620px; overflow-y: auto; }
.status { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.status .dot { width: 16px; height: 16px; border-radius: 50%; display: inline-grid;
  place-items: center; font-size: 11px; font-weight: 700; color: #fff; }
.status.visited { color: var(--good-ink); } .status.visited .dot { background: var(--good); }
.status.brushed { color: var(--warning-ink); } .status.brushed .dot { background: var(--warning); color: #3a2700; }
.status.missed { color: var(--critical-ink); } .status.missed .dot { background: var(--critical); }
.type { color: var(--ink-2); }
.timeline svg { display: block; width: 100%; }
.timeline .rowlabel { font-size: 12px; fill: var(--ink-2); }
.timeline .tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2);
  font-size: 12.5px; margin-bottom: 10px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.chip { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
ol.seq { margin: 0; padding-left: 28px; columns: 2 280px; column-gap: 32px; }
ol.seq li { padding: 3px 0; break-inside: avoid; color: var(--ink-2); }
ol.seq li b { color: var(--ink); font-weight: 600; }
.empty { color: var(--muted); }
#tip { position: fixed; pointer-events: none; z-index: 10; display: none;
  background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12.5px; max-width: 260px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
#tip .t { font-weight: 600; margin-bottom: 2px; }
#tip .m { color: var(--ink-2); }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title"></h1>
    <p class="sub" id="subtitle"></p>
  </header>

  <section class="tiles" id="tiles" aria-label="Summary"></section>

  <section class="grid">
    <div class="card">
      <h2>Map and path</h2>
      <div class="controls">
        <label><input type="checkbox" id="showLeft" checked>
          <span class="swatch" style="background:var(--left)"></span>Left hand</label>
        <label><input type="checkbox" id="showRight" checked>
          <span class="swatch" style="background:var(--right)"></span>Right hand</label>
        <label><input type="checkbox" id="showBoxes" checked>Boxes</label>
        <label><input type="checkbox" id="showLabels" checked>Box names</label>
      </div>
      <div class="mapwrap"><svg id="map" role="img" aria-label="Map with finger paths"></svg></div>
      <div class="scrub">
        <button id="play" type="button">Play</button>
        <input type="range" id="time" min="0" step="1" aria-label="Session time">
        <output id="timeLabel"></output>
      </div>
    </div>

    <div class="card">
      <h2>Boxes</h2>
      <div class="legend" id="statusLegend"></div>
      <div class="tablewrap">
        <table>
          <thead><tr>
            <th>Name</th><th>Type</th><th>Status</th>
            <th class="num">Visits</th><th class="num">Total time</th>
            <th class="num">Longest</th><th class="num">First entered</th>
          </tr></thead>
          <tbody id="boxRows"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="card timeline" style="margin-bottom:16px">
    <h2>Enter / exit timeline</h2>
    <div class="legend">
      <span><i class="chip" style="background:var(--left)"></i>Left hand</span>
      <span><i class="chip" style="background:var(--right)"></i>Right hand</span>
      <span><i class="chip" style="background:var(--both)"></i>Both hands</span>
      <span style="color:var(--muted)">· Page tracking row:</span>
      <span><i class="chip" style="background:var(--good)"></i>Reliable</span>
      <span><i class="chip" style="background:var(--warning)"></i>Estimated</span>
      <span><i class="chip" style="background:var(--critical)"></i>Lost / map not found yet</span>
    </div>
    <div id="timeline"></div>
  </section>

  <section class="card">
    <h2>Order of visits</h2>
    <div id="sequence"></div>
  </section>
</main>
<div id="tip" role="tooltip"></div>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmt = (ms) => {
    if (ms == null) return "—";
    const s = ms / 1000, m = Math.floor(s / 60);
    return m + ":" + (s - m * 60).toFixed(1).padStart(4, "0");
  };
  const secs = (ms) => (ms / 1000).toFixed(1) + " s";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const STATUS = {
    visited: { label: "Visited", icon: "✓" },
    brushed: { label: "Brushed only", icon: "~" },
    missed:  { label: "Missed", icon: "✕" },
  };
  const statusHTML = (s) =>
    `<span class="status ${s}"><span class="dot" aria-hidden="true">${STATUS[s].icon}</span>${STATUS[s].label}</span>`;
  const handVar = (hands) =>
    hands.length > 1 ? "var(--both)" : hands[0] === "Left" ? "var(--left)" : "var(--right)";
  const handText = (hands) => hands.length > 1 ? "Both hands" : hands[0] + " hand";

  // ---------- header + tiles ----------
  $("title").textContent = D.video ? D.video.replace(/\.[^.]+$/, "") : "Session";
  $("subtitle").textContent = [D.map && "Map: " + D.map,
    "Length " + fmt(D.duration_ms),
    "Visits count only when a finger stays ≥ " + (D.min_visit_ms / 1000) + " s"]
    .filter(Boolean).join(" · ");

  const rooms = D.boxes.filter((b) => b.type === "room");
  const symbols = D.boxes.filter((b) => b.type === "symbol");
  const visited = (list) => list.filter((b) => b.status === "visited").length;
  const missed = D.boxes.filter((b) => b.status === "missed");
  let reliable = null;
  if (D.tracking.length) {
    const good = D.tracking.filter((r) => r[2] === "good")
      .reduce((a, r) => a + (r[1] - r[0]), 0);
    reliable = good / Math.max(1, D.tracking[D.tracking.length - 1][1]);
  }
  const tiles = [
    ["Session length", fmt(D.duration_ms), D.frames ? D.frames + " frames" : ""],
    ["Time on paper", D.paper ? secs(D.paper.total_ms) : "—",
      D.paper ? D.paper.visits + " touch" + (D.paper.visits === 1 ? "" : "es") + " ≥ " + secs(D.min_visit_ms) : ""],
    ["Rooms visited", rooms.length ? visited(rooms) + " / " + rooms.length : "—",
      rooms.length ? "" : "no rooms found"],
    ["Symbols visited", symbols.length ? visited(symbols) + " / " + symbols.length : "—",
      symbols.length ? "" : "no symbols found"],
    ["Missed entirely", D.boxes.length ? String(missed.length) : "—",
      missed.length ? missed.slice(0, 3).map((b) => esc(b.name)).join(", ") +
        (missed.length > 3 ? "…" : "") : ""],
    ["Finger detected", D.frames ? Math.round(100 * D.frames_with_hand / D.frames) + "%" : "—",
      "of frames"],
    ["Page tracked reliably", reliable == null ? "—" : Math.round(100 * reliable) + "%",
      "of session time"],
  ];
  $("tiles").innerHTML = tiles.map(([l, v, n]) =>
    `<div class="card tile"><div class="label">${l}</div><div class="value">${v}</div>` +
    `<div class="note">${n || "&nbsp;"}</div></div>`).join("");

  // ---------- tooltip ----------
  const tip = $("tip");
  function showTip(evt, title, lines) {
    tip.innerHTML = `<div class="t">${esc(title)}</div>` +
      lines.map((l) => `<div class="m">${l}</div>`).join("");
    tip.style.display = "block";
    const pad = 14, r = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.style.display = "none"; };
  const typeText = (b) => b.type === "room" ? "Room" :
    b.type === "symbol" ? "Symbol" + (b.room ? " in " + b.room : "") : b.type;
  const boxLines = (b) => [
    esc(typeText(b)) + " · " + STATUS[b.status].label,
    b.visits + " visit" + (b.visits === 1 ? "" : "s") + " · " + secs(b.total_ms) + " total",
    b.brushes ? b.brushes + " brief pass" + (b.brushes === 1 ? "" : "es") + " (< " + secs(D.min_visit_ms) + ")" : "",
  ].filter(Boolean);

  // ---------- map ----------
  const svg = $("map");
  svg.setAttribute("viewBox", `0 0 ${D.w} ${D.h}`);
  el("image", { href: D.image, x: 0, y: 0, width: D.w, height: D.h }, svg);
  const boxLayer = el("g", {}, svg);
  const pathLayer = el("g", { fill: "none", "stroke-width": 3,
    "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  const dotLayer = el("g", {}, svg);
  const stroke = Math.max(2, D.w / 400);
  // label size in map pixels, so it stays readable whatever the PNG's size
  // (the map image itself is always white paper, so labels use fixed ink)
  const fontPx = Math.max(12, D.w / 50);
  const boxEls = {};
  for (const b of D.boxes) {
    const xs = b.poly.map((p) => p[0]), ys = b.poly.map((p) => p[1]);
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    const g = el("g", { "data-name": b.name }, boxLayer);
    const color = `var(--${b.status === "visited" ? "good" : b.status === "brushed" ? "warning" : "critical"})`;
    const r = el("polygon", { points: b.poly.map((p) => p.join(",")).join(" "),
      fill: color, "fill-opacity": b.type === "room" ? 0.10 : 0.04,
      stroke: color, "stroke-width": b.type === "room" ? stroke : stroke * 0.75 }, g);
    const label = el("text", { x: x0 + fontPx * 0.3, y: y0 + fontPx * 1.05, class: "boxlabel",
      "font-size": b.type === "room" ? fontPx : fontPx * 0.75, "stroke-width": fontPx / 4 }, g);
    label.textContent = b.name;
    // hit target = the box itself (rooms/symbols are large enough)
    r.addEventListener("mousemove", (e) => { showTip(e, b.name, boxLines(b)); highlight(b.name, true); });
    r.addEventListener("mouseleave", () => { hideTip(); highlight(b.name, false); });
    boxEls[b.name] = { g, r, label };
  }
  const handPaths = {
    Left: el("path", { stroke: "var(--left)" }, pathLayer),
    Right: el("path", { stroke: "var(--right)" }, pathLayer),
  };
  const handDots = {
    Left: el("circle", { r: 7, fill: "var(--left)", stroke: "var(--surface)", "stroke-width": 2 }, dotLayer),
    Right: el("circle", { r: 7, fill: "var(--right)", stroke: "var(--surface)", "stroke-width": 2 }, dotLayer),
  };

  function pathUpTo(points, t) {
    let d = "", pen = false, last = null;
    for (const p of points) {
      if (p === null) { pen = false; continue; }
      if (p[0] > t) break;
      d += (pen ? "L" : "M") + p[1] + " " + p[2];
      pen = true; last = p;
    }
    return { d, last };
  }

  // ---------- time / scrubbing ----------
  const slider = $("time");
  slider.max = D.duration_ms;
  slider.value = D.duration_ms;
  let playing = false, raf = null;

  function render() {
    const t = +slider.value;
    $("timeLabel").textContent = fmt(t) + " / " + fmt(D.duration_ms);
    for (const hand of ["Left", "Right"]) {
      const on = $("show" + hand).checked;
      const { d, last } = pathUpTo(D.paths[hand] || [], t);
      handPaths[hand].setAttribute("d", d);
      handPaths[hand].style.display = on ? "" : "none";
      // current fingertip dot: only if the hand was seen within the last 0.5 s
      const showDot = on && last && t - last[0] <= 500 && t < D.duration_ms;
      handDots[hand].style.display = showDot ? "" : "none";
      if (showDot) { handDots[hand].setAttribute("cx", last[1]); handDots[hand].setAttribute("cy", last[2]); }
    }
    boxLayer.style.display = $("showBoxes").checked ? "" : "none";
    for (const n in boxEls) boxEls[n].label.style.display = $("showLabels").checked ? "" : "none";
    // boxes occupied at time t get a heavier outline
    for (const b of D.boxes) {
      const inside = b.events.some((e) => e.enter_ms <= t && t <= e.exit_ms) && t < D.duration_ms;
      boxEls[b.name].r.setAttribute("stroke-width", inside ? stroke * 2.5 : stroke);
    }
    cursorLine && cursorLine.setAttribute("transform", `translate(${xScale(t)},0)`);
  }
  slider.addEventListener("input", () => { stop(); render(); });
  for (const id of ["showLeft", "showRight", "showBoxes", "showLabels"])
    $(id).addEventListener("change", render);

  function stop() { playing = false; $("play").textContent = "Play"; cancelAnimationFrame(raf); }
  $("play").addEventListener("click", () => {
    if (playing) return stop();
    if (+slider.value >= D.duration_ms) slider.value = 0;
    playing = true; $("play").textContent = "Pause";
    let prev = performance.now();
    const speed = Math.max(1, D.duration_ms / 30000); // whole session in ≤ ~30 s
    const step = (now) => {
      if (!playing) return;
      slider.value = Math.min(D.duration_ms, +slider.value + (now - prev) * speed);
      prev = now; render();
      if (+slider.value >= D.duration_ms) return stop();
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  });

  // ---------- table ----------
  const order = { room: 0, symbol: 1 };
  const sorted = [...D.boxes].sort((a, b) =>
    (order[a.type] - order[b.type]) || a.name.localeCompare(b.name, undefined, { numeric: true }));
  const legendCounts = { visited: 0, brushed: 0, missed: 0 };
  D.boxes.forEach((b) => legendCounts[b.status]++);
  $("statusLegend").innerHTML = D.boxes.length ? Object.keys(STATUS).map((s) =>
    statusHTML(s) + `<span style="color:var(--muted)">&nbsp;${legendCounts[s]}</span>`).join("") : "";
  $("boxRows").innerHTML = sorted.length ? sorted.map((b) => `
    <tr data-name="${esc(b.name)}">
      <td>${esc(b.name)}</td><td class="type">${esc(typeText(b))}</td>
      <td>${statusHTML(b.status)}</td>
      <td class="num">${b.visits}</td><td class="num">${b.visits ? secs(b.total_ms) : "—"}</td>
      <td class="num">${b.visits ? secs(b.longest_ms) : "—"}</td>
      <td class="num">${fmt(b.first_enter_ms)}</td>
    </tr>`).join("") :
    `<tr><td colspan="7" class="empty">No rooms or symbols were found on this map.</td></tr>`;
  const rowEls = {};
  for (const tr of $("boxRows").querySelectorAll("tr[data-name]")) {
    rowEls[tr.dataset.name] = tr;
    tr.addEventListener("mouseenter", () => highlight(tr.dataset.name, true));
    tr.addEventListener("mouseleave", () => highlight(tr.dataset.name, false));
  }
  function highlight(name, on) {
    const b = boxEls[name];
    if (b) b.r.setAttribute("fill-opacity", on ? 0.35 :
      (D.boxes.find((x) => x.name === name).type === "room" ? 0.10 : 0.04));
    if (rowEls[name]) rowEls[name].classList.toggle("hl", on);
  }

  // ---------- timeline ----------
  let xScale = () => 0, cursorLine = null;
  function drawTimeline() {
    const host = $("timeline");
    host.innerHTML = "";
    const W = host.clientWidth || 800, labelW = Math.min(170, W * 0.3), padR = 12;
    const rowH = 24, top = 4, axisH = 22;
    const rows = [{ name: "Page tracking", tracking: true },
      ...(D.paper ? [D.paper] : []), ...sorted];
    const H = top + rows.length * rowH + axisH;
    const s = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H,
      role: "img", "aria-label": "Enter and exit timeline per box" }, host);
    const plotW = W - labelW - padR;
    xScale = (t) => labelW + (D.duration_ms ? t / D.duration_ms : 0) * plotW;

    // ticks: ~6 round intervals
    const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].map((x) => x * 1000);
    const stepMs = nice.find((n) => D.duration_ms / n <= 6) || 1200000;
    for (let t = 0; t <= D.duration_ms; t += stepMs) {
      const x = xScale(t);
      el("line", { x1: x, x2: x, y1: top, y2: H - axisH, stroke: "var(--grid)", "stroke-width": 1 }, s);
      const tx = el("text", { x, y: H - 6, "text-anchor": "middle", class: "tick" }, s);
      tx.textContent = fmt(t).replace(/\.\d$/, "");
    }
    el("line", { x1: labelW, x2: W - padR, y1: H - axisH, y2: H - axisH, stroke: "var(--axis)" }, s);

    rows.forEach((row, i) => {
      const y = top + i * rowH;
      const lab = el("text", { x: labelW - 8, y: y + rowH / 2 + 4, "text-anchor": "end", class: "rowlabel" }, s);
      lab.textContent = row.name.length > 22 ? row.name.slice(0, 21) + "…" : row.name;
      if (row.tracking) {
        if (!D.tracking.length) return;
        for (const [t0, t1, q] of D.tracking) {
          const c = q === "good" ? "var(--good)" : q === "weak" ? "var(--warning)" : "var(--critical)";
          el("rect", { x: xScale(t0), y: y + 9, width: Math.max(0.5, xScale(t1) - xScale(t0)),
            height: 6, fill: c }, s);
        }
        return;
      }
      if (!row.events.length) {
        const t = el("text", { x: labelW + 6, y: y + rowH / 2 + 4, class: "tick" }, s);
        t.textContent = row.status === "missed" ? "missed — never entered" : "only brief passes (< " + secs(D.min_visit_ms) + ")";
        return;
      }
      for (const e of row.events) {
        const x0 = xScale(e.enter_ms), w = Math.max(3, xScale(e.exit_ms) - x0);
        const r = el("rect", { x: x0, y: y + 5, width: w, height: rowH - 10, rx: 3,
          fill: handVar(e.hands) }, s);
        // invisible larger hit target for small bars
        const hit = el("rect", { x: x0 - 4, y: y + 1, width: w + 8, height: rowH - 2,
          fill: "transparent" }, s);
        const lines = [fmt(e.enter_ms) + " → " + fmt(e.exit_ms), secs(e.duration_ms) + " · " + handText(e.hands)];
        hit.addEventListener("mousemove", (ev) => { showTip(ev, row.name, lines); highlight(row.name, true); });
        hit.addEventListener("mouseleave", () => { hideTip(); highlight(row.name, false); });
        hit.addEventListener("click", () => { stop(); slider.value = e.enter_ms; render(); });
        hit.style.cursor = "pointer";
      }
    });
    cursorLine = el("line", { x1: 0, x2: 0, y1: top, y2: H - axisH, stroke: "var(--ink-2)",
      "stroke-width": 1.5 }, s);
    s.addEventListener("click", (ev) => {
      const r = s.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width);
      if (x < labelW) return;
      stop(); slider.value = Math.round(((x - labelW) / plotW) * D.duration_ms); render();
    });
  }

  // ---------- sequence ----------
  $("sequence").innerHTML = D.sequence.length
    ? "<ol class=\"seq\">" + D.sequence.map((e) =>
        `<li><b>${esc(e.name)}</b> <span class="type">(${e.type})</span> · ${fmt(e.enter_ms)}–${fmt(e.exit_ms)} · ${secs(e.duration_ms)} · ${handText(e.hands)}</li>`
      ).join("") + "</ol>"
    : `<p class="empty">No visits of ≥ ${secs(D.min_visit_ms)} recorded${D.boxes.length ? "" : " (no boxes labeled for this map)"}.</p>`;

  drawTimeline();
  render();
  let resizeTimer;
  addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawTimeline(); render(); }, 120); });
})();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", help="a <video>.dashboard.json file")
    parser.add_argument("map", nargs="?", help="map PNG (default: the one named in the JSON)")
    parser.add_argument("-o", "--output", help="output .html (default: next to the JSON)")
    args = parser.parse_args()
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    map_path = args.map or data.get("map_path")
    if not map_path or not os.path.exists(map_path):
        map_path = os.path.join(os.path.dirname(args.data) or ".", data["map"])
    out = args.output or os.path.splitext(args.data)[0] + ".html"
    build_dashboard(data, map_path, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
