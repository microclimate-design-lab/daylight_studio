/* Daylight Studio – bubble diagram editor + live daylight prediction */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";
const CUT = 127 / 255;
const DEFAULT_AREA = [30, 14, 12, 12, 12, 7, 5, 9, 3]; // m², used for the default bubble radius
const SHORT = ["Living", "Master 0", "Master 1", "Master 2", "Dining", "Kitchen", "Bath", "Study", "Storage"];

const S = {
  meta: null, types: [],
  outline: [], draft: null,
  rooms: [], links: [], windows: {}, nextId: 1,
  sel: null, tool: "select", curType: 0,
  model: "XGBoost", ori: 0, wwr_s: 0.35, wwr_n: 0.27,
  view: { s: 40, tx: 0, ty: 0 },
  result: null, sample: null, pristine: false,
  overlay: true, showBubbles: true, labels: true, cmap: "grey", mapView: "pred",
  linkFrom: null, ghost: null, hover: null, drag: null,
};

/* ================================================================ geometry helpers */
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1));
const snap = (v, g = 0.1) => Math.round(v / g) * g;
const W2S = (x, y) => [x * S.view.s + S.view.tx, y * S.view.s + S.view.ty];
const S2W = (X, Y) => [(X - S.view.tx) / S.view.s, (Y - S.view.ty) / S.view.s];
function polyArea(p) { let a = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; a += x1 * y2 - x2 * y1; } return Math.abs(a) / 2; }
function polyPerim(p) { let l = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; l += Math.hypot(x2 - x1, y2 - y1); } return l; }
function bbox(p) { const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]); return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; }
function inPoly([x, y], p) {
  let c = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i], [xj, yj] = p[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function segDist(px, py, [x1, y1], [x2, y2]) {
  const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy || 1e-9;
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / L));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}
const rgb = (t) => { const c = S.types[t]?.rgb || [200, 200, 200]; return `rgb(${c[0]},${c[1]},${c[2]})`; };
const roomById = (id) => S.rooms.find((r) => r.id === id);
const resRoom = (id) => S.result?.rooms.find((r) => r.id === id);
const roomLabel = (r) => `${SHORT[r.type]} · #${r.id}`;

/* ================================================================ colour ramps */
const HEAT = [[0, [20, 18, 38]], [0.25, [74, 32, 110]], [0.5, [186, 62, 84]], [0.75, [245, 146, 60]], [1, [252, 245, 190]]];
function ramp(stops, v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = (v - a) / (b - a);
      return ca.map((c, k) => Math.round(c + (cb[k] - c) * t));
    }
  }
  return stops[stops.length - 1][1];
}
function colour(v, mode) {
  if (mode === "err") { // v in [-1,1]
    const t = Math.max(-1, Math.min(1, v / 0.4)), a = Math.round(205 * Math.abs(t));
    return t >= 0 ? [255, 255 - a, 255 - a] : [255 - a, 255 - a, 255];
  }
  if (S.cmap === "heat") return ramp(HEAT, v);
  if (S.cmap === "pass") return v > CUT ? [236, 236, 232] : [52, 52, 52];
  const g = Math.round(v * 255); return [g, g, g];
}
function mapImage(values, rows, cols, mode = "pred", ref = null) {
  const c = document.createElement("canvas");
  c.width = cols; c.height = rows;
  const ctx = c.getContext("2d"), img = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    let v = values[i];
    if (v < 0 || v == null) { img.data[i * 4 + 3] = 0; continue; }
    if (mode === "err") { const t = ref?.[i]; if (t == null || t < 0) { img.data[i * 4 + 3] = 0; continue; } v = v - t; }
    const [r, g, b] = colour(v, mode);
    img.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
function drawLegend() {
  const c = $("#legend"), ctx = c.getContext("2d");
  for (let x = 0; x < c.width; x++) {
    const v = S.mapView === "err" ? (x / (c.width - 1)) * 0.8 - 0.4 : x / (c.width - 1);
    const [r, g, b] = colour(v, S.mapView === "err" ? "err" : "pred");
    ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(x, 0, 1, c.height);
  }
  const t = $(".legend-ticks");
  t.innerHTML = S.mapView === "err"
    ? `<span>−0.4 (under)</span><span class="cut">0</span><span>+0.4 (over)</span>`
    : `<span>0</span><span class="cut">0.498 (sDA cut)</span><span>1</span>`;
}

/* ================================================================ plan rendering */
const svg = $("#plan");
function el(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
let overlayURL = null, overlayKey = "";
function render() {
  const { width: Wp, height: Hp } = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${Wp} ${Hp}`);
  svg.replaceChildren();
  const s = S.view.s;
  // grid
  const [x0, y0] = S2W(0, 0), [x1, y1] = S2W(Wp, Hp);
  const step = s > 25 ? 0.5 : 1;
  const grid = el("g");
  for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) {
    const [X] = W2S(x, 0), major = Math.abs(x - Math.round(x)) < 1e-6 && Math.round(x) % 5 === 0;
    grid.append(el("line", { x1: X, y1: 0, x2: X, y2: Hp, class: major ? "g-major" : "g-minor" }));
  }
  for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) {
    const [, Y] = W2S(0, y), major = Math.abs(y - Math.round(y)) < 1e-6 && Math.round(y) % 5 === 0;
    grid.append(el("line", { x1: 0, y1: Y, x2: Wp, y2: Y, class: major ? "g-major" : "g-minor" }));
  }
  svg.append(grid);

  // outline fill + DA overlay
  if (S.outline.length >= 3) {
    const pts = S.outline.map((p) => W2S(...p).join(",")).join(" ");
    svg.append(el("polygon", { points: pts, class: "outline-fill" }));
    const R = S.result;
    if (R && S.overlay) {
      const key = `${R._seq}|${S.cmap}`;
      if (key !== overlayKey) {
        overlayURL = mapImage(R.da_map.values, R.da_map.rows, R.da_map.cols).toDataURL();
        overlayKey = key;
      }
      const [bx, by] = R.da_map.bbox_m, cm = R.da_map.cell_m;
      const [X, Y] = W2S(bx, by);
      const clip = el("clipPath", { id: "oclip" }); clip.append(el("polygon", { points: pts }));
      svg.append(clip);
      svg.append(el("image", { href: overlayURL, x: X, y: Y, width: R.da_map.cols * cm * s, height: R.da_map.rows * cm * s,
        preserveAspectRatio: "none", "clip-path": "url(#oclip)", style: "image-rendering:pixelated", opacity: 0.95 }));
    }
    svg.append(el("polygon", { points: pts, class: "outline" }));
    // dimensions
    const [bx0, by0, bx1, by1] = bbox(S.outline);
    const [DX0, DY0] = W2S(bx0, by0), [DX1, DY1] = W2S(bx1, by1);
    svg.append(el("text", { x: (DX0 + DX1) / 2, y: DY0 - 10, class: "dim", "text-anchor": "middle" }, `${fmt(bx1 - bx0)} m`));
    svg.append(el("text", { x: DX0 - 10, y: (DY0 + DY1) / 2, class: "dim", "text-anchor": "middle", transform: `rotate(-90 ${DX0 - 10} ${(DY0 + DY1) / 2})` }, `${fmt(by1 - by0)} m`));
    if (S.tool === "select") {
      S.outline.forEach((p) => { const [X, Y] = W2S(...p); svg.append(el("rect", { x: X - 4, y: Y - 4, width: 8, height: 8, class: "vtx" })); });
    }
  }
  // draft boundary
  if (S.draft) {
    const pts = [...S.draft, ...(S.hover ? [S.hover] : [])].map((p) => W2S(...p).join(",")).join(" ");
    svg.append(el("polyline", { points: pts, class: "draft" }));
    S.draft.forEach((p, i) => { const [X, Y] = W2S(...p); svg.append(el("rect", { x: X - 4, y: Y - 4, width: 8, height: 8, class: "vtx", ...(i === 0 ? { style: "fill:var(--ink)" } : {}) })); });
    if (S.hover && S.draft.length) {
      const a = S.draft[S.draft.length - 1], b = S.hover, [X, Y] = W2S((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      svg.append(el("text", { x: X + 6, y: Y - 6, class: "dim" }, `${fmt(Math.hypot(b[0] - a[0], b[1] - a[1]))} m`));
    }
  }
  // links
  if (S.showBubbles) {
    for (const [a, b] of S.links) {
      const A = roomById(a), B = roomById(b); if (!A || !B) continue;
      const [X1, Y1] = W2S(A.x, A.y), [X2, Y2] = W2S(B.x, B.y);
      svg.append(el("line", { x1: X1, y1: Y1, x2: X2, y2: Y2, class: "link" }));
    }
  }
  // bubbles
  const fo = S.overlay && S.result ? 0.38 : 0.85;
  const sorted = [...S.rooms].sort((a, b) => b.r - a.r);
  if (S.showBubbles) for (const r of sorted) {
    const [X, Y] = W2S(r.x, r.y), inside = S.outline.length >= 3 && inPoly([r.x, r.y], S.outline);
    svg.append(el("circle", { cx: X, cy: Y, r: r.r * s, fill: rgb(r.type), "fill-opacity": fo,
      class: `bubble${S.sel === r.id ? " sel" : ""}${inside ? "" : " out"}` }));
  }
  for (const r of S.rooms) {
    const [X, Y] = W2S(r.x, r.y);
    svg.append(el("circle", { cx: X, cy: Y, r: S.linkFrom === r.id ? 7 : 4.5, class: "cdot" }));
    if (S.labels) {
      const rr = resRoom(r.id);
      svg.append(el("text", { x: X, y: Y - 9, class: "lbl small", "text-anchor": "middle" }, `${SHORT[r.type]}`));
      if (rr) svg.append(el("text", { x: X, y: Y + 17, class: "lbl", "text-anchor": "middle" }, `${pct(rr.sda)}%`));
    }
  }
  // selected handle
  const sr = roomById(S.sel);
  if (sr && S.tool === "select") {
    const [X, Y] = W2S(sr.x + sr.r, sr.y);
    svg.append(el("rect", { x: X - 5, y: Y - 5, width: 10, height: 10, class: "handle", style: "cursor:ew-resize" }));
    svg.append(el("text", { x: X + 9, y: Y + 4, class: "dim" }, `r ${fmt(sr.r)} m`));
  }
  // ghost room (placing)
  if (S.ghost) {
    const [X, Y] = W2S(S.ghost.x, S.ghost.y);
    svg.append(el("circle", { cx: X, cy: Y, r: S.ghost.r * s, fill: rgb(S.ghost.type), "fill-opacity": 0.5, class: "bubble" }));
  }
  // link rubber band
  if (S.tool === "link" && S.linkFrom && S.hover) {
    const A = roomById(S.linkFrom);
    if (A) { const [X1, Y1] = W2S(A.x, A.y), [X2, Y2] = W2S(...S.hover); svg.append(el("line", { x1: X1, y1: Y1, x2: X2, y2: Y2, class: "draft" })); }
  }
  // scale bar
  const sb = el("g"), [sx, sy] = [16, Hp - 18];
  sb.append(el("line", { x1: sx, y1: sy, x2: sx + s, y2: sy, stroke: "currentColor", "stroke-width": 2 }));
  sb.append(el("line", { x1: sx, y1: sy - 4, x2: sx, y2: sy + 4, stroke: "currentColor" }));
  sb.append(el("line", { x1: sx + s, y1: sy - 4, x2: sx + s, y2: sy + 4, stroke: "currentColor" }));
  sb.append(el("text", { x: sx + s + 6, y: sy + 4, class: "dim" }, "1 m"));
  sb.style.color = "var(--ink)";
  svg.append(sb);
}

/* ================================================================ fitting */
function fit() {
  const { width: Wp, height: Hp } = svg.getBoundingClientRect();
  const pts = S.outline.length ? S.outline : S.rooms.length ? S.rooms.flatMap((r) => [[r.x - r.r, r.y - r.r], [r.x + r.r, r.y + r.r]]) : [[0, 0], [12, 12]];
  const [x0, y0, x1, y1] = bbox(pts), pad = 70;
  const s = Math.min((Wp - 2 * pad) / Math.max(x1 - x0, 1), (Hp - 2 * pad) / Math.max(y1 - y0, 1));
  S.view.s = Math.max(8, Math.min(140, s));
  S.view.tx = Wp / 2 - ((x0 + x1) / 2) * S.view.s;
  S.view.ty = Hp / 2 - ((y0 + y1) / 2) * S.view.s;
  S.autoFit = true;
  render();
}

/* ================================================================ interaction */
function evtWorld(e) { const r = svg.getBoundingClientRect(); return S2W(e.clientX - r.left, e.clientY - r.top); }
function hitRoom([x, y]) {
  let best = null, bd = Infinity;
  for (const r of S.rooms) { const d = Math.hypot(x - r.x, y - r.y); if (d <= Math.max(r.r, 8 / S.view.s) && d < bd) { best = r; bd = d; } }
  return best;
}
function hitLink([x, y]) {
  return S.links.findIndex(([a, b]) => {
    const A = roomById(a), B = roomById(b);
    return A && B && segDist(x, y, [A.x, A.y], [B.x, B.y]) * S.view.s < 6;
  });
}
svg.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const r = hitRoom(evtWorld(e));
  if (r) deleteRoom(r.id);
});
function touchGeom() { S.pristine = false; schedule(); }
function setTool(t) {
  if (S.draft && t !== "boundary") S.draft = null;
  S.tool = t; S.linkFrom = null; S.ghost = null;
  $$(".tool").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
  $("#tool-label").textContent = { select: "Select", boundary: "Draw boundary", room: `Place · ${S.types[S.curType]?.name || ""}`, link: "Link rooms", erase: "Delete" }[t];
  $("#stage-hint").textContent = {
    select: "Drag a bubble to move · drag the square handle to resize · drag a boundary edge to push it · drag empty space to pan · wheel to zoom · Del removes the selected bubble",
    boundary: "Click to add vertices (snaps to 0.1 m, orthogonal) · Alt = free angle · click the first point / Enter to close · Esc to cancel",
    room: "Click to drop a bubble with the default size, or press-drag to set its radius · keys 1–9 pick the room type",
    link: "Click two bubbles to add / remove a connection (adjacency edge in the graph)",
    erase: "Click a bubble to delete it (its links go too) · click a link line to remove only the link · right-click a bubble deletes it in any mode",
  }[t];
  svg.style.cursor = t === "select" ? "default" : t === "erase" ? "not-allowed" : "crosshair";
  render();
}
function orthoSnap(p, alt) {
  let [x, y] = [snap(p[0]), snap(p[1])];
  const d = S.draft;
  if (!alt && d && d.length) {
    const [px, py] = d[d.length - 1];
    if (Math.abs(x - px) > Math.abs(y - py)) y = py; else x = px;
    // snap to first vertex axis to make closing easy
    const [fx, fy] = d[0];
    if (Math.abs(x - fx) * S.view.s < 8) x = fx;
    if (Math.abs(y - fy) * S.view.s < 8) y = fy;
  }
  return [x, y];
}
function closeDraft() {
  if (!S.draft || S.draft.length < 3) return;
  S.outline = S.draft.map((p) => [+p[0].toFixed(3), +p[1].toFixed(3)]);
  S.draft = null; S.hover = null; S.sample = null;
  updateBoundaryMeta(); setTool("select"); touchGeom();
}

svg.addEventListener("pointerdown", (e) => {
  svg.focus();
  const p = evtWorld(e);
  svg.setPointerCapture(e.pointerId);
  if (e.button === 1 || (e.button === 0 && e.shiftKey && S.tool !== "select")) { S.drag = { kind: "pan", X: e.clientX, Y: e.clientY, tx: S.view.tx, ty: S.view.ty }; return; }
  if (S.tool === "boundary") {
    const q = orthoSnap(p, e.altKey);
    if (S.draft && S.draft.length >= 3) {
      const [fx, fy] = W2S(...S.draft[0]), [qx, qy] = W2S(...q);
      if (Math.hypot(fx - qx, fy - qy) < 10) { closeDraft(); return; }
    }
    S.draft = S.draft || [];
    const last = S.draft[S.draft.length - 1];
    if (!last || last[0] !== q[0] || last[1] !== q[1]) S.draft.push(q);
    render(); return;
  }
  if (S.tool === "room") {
    const r0 = 0.55 * Math.sqrt(DEFAULT_AREA[S.curType] / Math.PI);
    S.ghost = { x: snap(p[0], 0.05), y: snap(p[1], 0.05), r: r0, r0, type: S.curType };
    S.drag = { kind: "place" }; render(); return;
  }
  if (S.tool === "erase") {
    const r = hitRoom(p);
    if (r) { deleteRoom(r.id); return; }
    const li = hitLink(p);
    if (li >= 0) { S.links.splice(li, 1); touchGeom(); render(); }
    return;
  }
  if (S.tool === "link") {
    const r = hitRoom(p);
    if (!r) { S.linkFrom = null; render(); return; }
    if (!S.linkFrom) { S.linkFrom = r.id; render(); return; }
    if (S.linkFrom !== r.id) {
      const a = Math.min(S.linkFrom, r.id), b = Math.max(S.linkFrom, r.id);
      const i = S.links.findIndex(([x, y]) => x === a && y === b);
      if (i >= 0) S.links.splice(i, 1); else S.links.push([a, b]);
      touchGeom();
    }
    S.linkFrom = e.ctrlKey ? r.id : null; render(); return;
  }
  // select tool
  const sr = roomById(S.sel), [PX, PY] = W2S(...p);
  if (sr) {
    const [HX, HY] = W2S(sr.x + sr.r, sr.y);
    const dEdge = Math.abs(Math.hypot(p[0] - sr.x, p[1] - sr.y) - sr.r) * S.view.s;
    if (Math.hypot(PX - HX, PY - HY) < 9 || dEdge < 5) { S.drag = { kind: "resize", id: sr.id }; return; }
  }
  // boundary vertices
  for (let i = 0; i < S.outline.length; i++) {
    const [VX, VY] = W2S(...S.outline[i]);
    if (Math.hypot(PX - VX, PY - VY) < 8) { S.drag = { kind: "vertex", i }; return; }
  }
  const r = hitRoom(p);
  if (r) { select(r.id); S.drag = { kind: "move", id: r.id, dx: p[0] - r.x, dy: p[1] - r.y }; return; }
  for (let i = 0; i < S.outline.length; i++) {
    const a = S.outline[i], b = S.outline[(i + 1) % S.outline.length];
    if (segDist(p[0], p[1], a, b) * S.view.s < 6) {
      S.drag = { kind: "edge", i, start: p, orig: S.outline.map((q) => [...q]) }; return;
    }
  }
  select(null);
  S.drag = { kind: "pan", X: e.clientX, Y: e.clientY, tx: S.view.tx, ty: S.view.ty };
});

svg.addEventListener("pointermove", (e) => {
  const p = evtWorld(e);
  $("#cursor-pos").textContent = `x ${fmt(p[0])}  y ${fmt(p[1])} m`;
  if (S.tool === "boundary") { S.hover = S.draft ? orthoSnap(p, e.altKey) : null; render(); }
  if (S.tool === "link") { S.hover = p; if (S.linkFrom) render(); }
  const d = S.drag; if (!d) return;
  if (d.kind === "pan") { S.autoFit = false; S.view.tx = d.tx + e.clientX - d.X; S.view.ty = d.ty + e.clientY - d.Y; render(); return; }
  if (d.kind === "place") { const r = Math.hypot(p[0] - S.ghost.x, p[1] - S.ghost.y); S.ghost.r = r > 0.25 ? snap(r, 0.05) : S.ghost.r0; render(); return; }
  if (d.kind === "move") { const r = roomById(d.id); r.x = snap(p[0] - d.dx, 0.05); r.y = snap(p[1] - d.dy, 0.05); render(); touchGeom(); return; }
  if (d.kind === "resize") { const r = roomById(d.id); r.r = Math.max(0.25, snap(Math.hypot(p[0] - r.x, p[1] - r.y), 0.05)); syncEditor(); render(); touchGeom(); return; }
  if (d.kind === "vertex") { S.outline[d.i] = [snap(p[0]), snap(p[1])]; updateBoundaryMeta(); render(); touchGeom(); return; }
  if (d.kind === "edge") {
    const n = d.orig.length, a = d.orig[d.i], b = d.orig[(d.i + 1) % n];
    let dx = p[0] - d.start[0], dy = p[1] - d.start[1];
    if (Math.abs(a[1] - b[1]) < 1e-6) dx = 0; else if (Math.abs(a[0] - b[0]) < 1e-6) dy = 0;
    else { const L = Math.hypot(b[0] - a[0], b[1] - a[1]), nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L, t = dx * nx + dy * ny; dx = nx * t; dy = ny * t; }
    dx = snap(dx); dy = snap(dy);
    S.outline = d.orig.map((q) => [...q]);
    for (const k of [d.i, (d.i + 1) % n]) S.outline[k] = [+(d.orig[k][0] + dx).toFixed(3), +(d.orig[k][1] + dy).toFixed(3)];
    S.sample = null; updateBoundaryMeta(); render(); touchGeom();
  }
});

svg.addEventListener("pointerup", () => {
  const d = S.drag; S.drag = null;
  if (d?.kind === "place" && S.ghost) {
    const g = S.ghost, id = S.nextId++;
    S.rooms.push({ id, type: g.type, x: g.x, y: g.y, r: g.r });
    S.ghost = null; select(id); touchGeom();
  }
  render();
});
svg.addEventListener("dblclick", () => { if (S.tool === "boundary") closeDraft(); });
svg.addEventListener("wheel", (e) => {
  e.preventDefault();
  S.autoFit = false;
  const r = svg.getBoundingClientRect(), X = e.clientX - r.left, Y = e.clientY - r.top, [wx, wy] = S2W(X, Y);
  S.view.s = Math.max(6, Math.min(220, S.view.s * Math.exp(-e.deltaY * 0.0015)));
  S.view.tx = X - wx * S.view.s; S.view.ty = Y - wy * S.view.s; render();
}, { passive: false });

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea") || $("#lib").open) return;
  const k = e.key.toLowerCase();
  if (k === "v") setTool("select");
  else if (k === "b") setTool("boundary");
  else if (k === "r") setTool("room");
  else if (k === "l") setTool("link");
  else if (k === "d") setTool("erase");
  else if (k === "f") fit();
  else if (/^[1-9]$/.test(k)) { pickType(+k - 1); if (S.sel) { roomById(S.sel).type = +k - 1; syncEditor(); touchGeom(); render(); } }
  else if (k === "enter" && S.draft) closeDraft();
  else if (k === "escape") { S.draft = null; S.ghost = null; S.linkFrom = null; select(null); render(); }
  else if ((k === "delete" || k === "backspace") && S.sel) { deleteRoom(S.sel); e.preventDefault(); }
});

/* ================================================================ rooms */
function deleteRoom(id) {
  S.rooms = S.rooms.filter((r) => r.id !== id);
  S.links = S.links.filter(([a, b]) => a !== id && b !== id);
  delete S.windows[id];
  if (S.sel === id) select(null);
  touchGeom(); render();
}
function select(id) {
  S.sel = id;
  $("#room-editor").hidden = !id;
  syncEditor(); renderRooms(); render();
}
function syncEditor() {
  const r = roomById(S.sel); if (!r) return;
  $("#ed-swatch").style.background = rgb(r.type);
  $("#ed-title").textContent = `${S.types[r.type].name} · #${r.id}`;
  $("#ed-type").value = r.type;
  $("#ed-r").value = r.r; $("#ed-r-out").textContent = `${fmt(r.r)} m`;
  const w = S.windows[r.id] || "auto";
  $$("#ed-win button").forEach((b) => b.classList.toggle("on", b.dataset.v === w));
}
function pickType(t) {
  S.curType = t;
  $$("#palette button").forEach((b) => b.classList.toggle("on", +b.dataset.t === t));
  if (S.tool === "room") setTool("room");
}

/* ================================================================ prediction */
let timer = null, inflight = null, seq = 0;
function schedule(delay = 70) { clearTimeout(timer); timer = setTimeout(runPredict, delay); }
function status(kind, text) { $("#status-dot").className = `dot ${kind}`; $("#status-text").textContent = text; }
function showMsg(t) { const m = $("#msg"); m.hidden = !t; m.textContent = t || ""; }
async function runPredict() {
  if (S.outline.length < 3) { status("", "draw or load a boundary"); clearResult(); return; }
  if (!S.rooms.length) { status("", "place room bubbles"); clearResult(); return; }
  inflight?.abort();
  inflight = new AbortController();
  const my = ++seq;
  status("busy", "predicting…");
  const body = {
    model: S.model, ori: 0, wwr_s: S.wwr_s, wwr_n: S.wwr_n, outline: S.outline,
    rooms: S.rooms.map(({ id, type, x, y, r }) => ({ id, type, x, y, r })), links: S.links,
    windows: Object.fromEntries(Object.entries(S.windows).filter(([, v]) => v !== "auto")),
  };
  try {
    const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: inflight.signal });
    const j = await res.json();
    if (my !== seq) return;
    if (!res.ok) { status("err", j.error || "error"); showMsg(j.error); return; }
    j._seq = my;
    S.result = j;
    showMsg(j.warnings?.length ? j.warnings.join(" · ") : "");
    status("ok", `${j.model} · ${j.ms} ms`);
    renderResults(); render();
  } catch (err) {
    if (err.name !== "AbortError") { status("err", "server unreachable"); showMsg(String(err)); }
  }
}
function clearResult() { S.result = null; renderResults(); render(); }

/* ================================================================ results panel */
function truthActive() {
  const sm = S.sample;
  return sm && S.pristine && sm.params.ori === 0 && Math.abs(sm.params.wwr_s - S.wwr_s) < 0.005 && Math.abs(sm.params.wwr_n - S.wwr_n) < 0.005;
}
function renderResults() {
  const R = S.result;
  $("#model-tag").textContent = R ? R.model : S.model;
  $("#kpi-sda").textContent = R ? pct(R.unit.sda) : "—";
  $("#gauge-fill").style.width = R ? `${R.unit.sda * 100}%` : "0";
  $("#kpi-sda-map").textContent = R ? `${pct(R.unit.sda_map)} %` : "—";
  $("#kpi-mean").textContent = R ? fmt(R.unit.mean_da, 3) : "—";
  $("#kpi-mean-map").textContent = R ? fmt(R.unit.mean_da_map, 3) : "—";
  $("#kpi-graph").textContent = R ? `${R.graph.node_features.length} · ${R.graph.edge_index[0].length / 2}` : "—";

  const tb = $("#truth-box"), gm = $("#gauge-truth"), sm = S.sample;
  $("#btn-sample-params").hidden = !(sm && S.pristine && sm.params.ori === 0) || truthActive();
  if (sm && S.pristine) {
    tb.hidden = false;
    if (truthActive() && R) {
      const e = R.unit.sda - sm.truth.sda;
      tb.innerHTML = `Sample <b>#${sm.id}</b> (${sm.split}) · simulated sDA <b>${pct(sm.truth.sda)} %</b> · error <span class="${Math.abs(e) > 0.05 ? "err" : ""}">${e >= 0 ? "+" : ""}${pct(e)} pt</span>`;
      gm.hidden = false; gm.style.left = `calc(${sm.truth.sda * 100}% - 1px)`;
    } else if (sm.params.ori !== 0) {
      tb.innerHTML = `Sample <b>#${sm.id}</b> (${sm.split}) was simulated rotated ${sm.params.ori === 1 ? "30° E" : "30° W"}; orientation is fixed to 0° here, so no simulated reference is shown.`;
      gm.hidden = true;
    } else {
      tb.innerHTML = `Sample <b>#${sm.id}</b> (${sm.split}) was simulated at WWR<sub>S</sub> ${fmt(sm.params.wwr_s)}, WWR<sub>N</sub> ${fmt(sm.params.wwr_n)} — match these to compare with the simulation.`;
      gm.hidden = true;
    }
  } else { tb.hidden = true; gm.hidden = true; }
  renderRooms(); renderMap(); renderGraph();
}
function renderRooms() {
  const box = $("#room-list"), R = S.result;
  if (!S.rooms.length) { box.innerHTML = `<p class="empty">No bubbles yet. Pick a room type and press R to place.</p>`; return; }
  const ta = truthActive();
  const rows = [...S.rooms].sort((a, b) => a.type - b.type || a.id - b.id).map((r) => {
    const q = R?.rooms.find((x) => x.id === r.id);
    const sm = ta ? S.sample.rooms.find((x) => x.id === r.id) : null;
    const win = q ? `${q.win_source === "user" ? "fixed" : "p"} N ${fmt(q.p_win_N)} · S ${fmt(q.p_win_S)}` : "";
    return `<div class="room-row${S.sel === r.id ? " sel" : ""}" data-id="${r.id}">
      <span class="sw" style="background:${rgb(r.type)}"></span>
      <span class="nm">${S.types[r.type].name}<small>#${r.id}${q ? ` · ${fmt(q.area, 1)} m²` : ""}</small></span>
      <span class="val">${q ? pct(q.sda) : "—"}<small style="font-size:10px;color:var(--ink-3)">%</small><button class="del" data-del="${r.id}" title="Delete bubble #${r.id}" aria-label="Delete bubble #${r.id}">×</button></span>
      <span class="bar"><em></em><span style="width:${q ? q.sda * 100 : 0}%"></span>${sm?.true_sda != null ? `<i style="left:calc(${sm.true_sda * 100}% - 1px)"></i>` : ""}</span>
      <span class="sub2">${q ? `<span>map <b>${q.sda_map == null ? "—" : pct(q.sda_map) + "%"}</b></span><span>mean DA <b>${fmt(q.mean_da, 3)}</b></span><span>${win}</span>` : "<span>outside boundary</span>"}${sm?.true_sda != null ? `<span>sim <b style="color:var(--warn)">${pct(sm.true_sda)}%</b></span>` : ""}</span>
    </div>`;
  });
  box.innerHTML = rows.join("");
  $$(".room-row", box).forEach((row) => row.addEventListener("click", (e) => {
    const d = e.target.closest("[data-del]");
    if (d) { e.stopPropagation(); deleteRoom(+d.dataset.del); } else select(+row.dataset.id);
  }));
}
function renderMap() {
  drawLegend();
  const c = $("#map-canvas"), R = S.result;
  const truthOk = S.sample && S.pristine && R && S.sample.truth.rows === R.da_map.rows && S.sample.truth.cols === R.da_map.cols;
  $$("#seg-mapview button").forEach((b) => { b.disabled = b.dataset.v !== "pred" && !truthOk; b.classList.toggle("on", b.dataset.v === S.mapView); });
  if (!R) { c.width = 1; c.height = 1; return; }
  let mode = S.mapView; if (!truthOk) mode = "pred";
  const vals = mode === "truth" ? S.sample.truth.values : R.da_map.values;
  const img = mapImage(vals, R.da_map.rows, R.da_map.cols, mode === "err" ? "err" : "pred", mode === "err" ? S.sample.truth.values : null);
  const k = Math.max(2, Math.floor(Math.min(340 / R.da_map.cols, 420 / R.da_map.rows)));
  c.width = R.da_map.cols * k; c.height = R.da_map.rows * k;
  const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = false; ctx.drawImage(img, 0, 0, c.width, c.height);
  c._k = k; c._mode = mode;
}
$("#map-canvas").addEventListener("mousemove", (e) => {
  const c = e.currentTarget, R = S.result; if (!R || !c._k) return;
  const b = c.getBoundingClientRect(), col = Math.floor(((e.clientX - b.left) / b.width) * R.da_map.cols), row = Math.floor(((e.clientY - b.top) / b.height) * R.da_map.rows);
  const i = row * R.da_map.cols + col, v = R.da_map.values[i];
  if (v == null || v < 0) { $("#map-hover").textContent = ""; return; }
  const ri = R.da_map.room[i], rm = R.rooms.find((x) => x.index === ri);
  const t = S.sample && c._mode !== "pred" ? S.sample.truth.values[i] : null;
  $("#map-hover").textContent = `r${row} c${col} · pred ${fmt(v, 3)}${t != null && t >= 0 ? ` · sim ${fmt(t, 3)}` : ""} · ${rm ? SHORT[rm.type] + " #" + rm.id : "—"}`;
});
$("#map-canvas").addEventListener("mouseleave", () => { $("#map-hover").textContent = ""; });

function renderGraph() {
  const box = $("#graph-view"), R = S.result;
  if (!R) { box.innerHTML = `<p class="empty">The graph appears once a boundary and at least one bubble exist.</p>`; return; }
  const G = R.graph;
  const gf = G.graph_feature_names.map((n, i) => `<tr><td>${n}</td><td>${G.graph_features[i] == null ? "— (not used)" : fmt(G.graph_features[i], n === "ORI" ? 0 : 3)}</td></tr>`).join("");
  const head = ["#", "room", "area", "ratio", "x", "y", "d_N", "d_S", "d_E", "d_W", "win_N", "win_S"];
  const nf = G.node_features.map((r, k) => `<tr><td>${k}</td><td><span class="sw" style="background:${rgb(r[0])};width:9px;height:9px"></span> ${SHORT[r[0]]} #${G.order[k]}</td>${r.slice(1).map((v, j) => `<td>${j >= 8 ? v : fmt(v, j === 0 ? 1 : 2)}</td>`).join("")}</tr>`).join("");
  const E = []; for (let i = 0; i < G.edge_index[0].length; i += 2) { const a = G.edge_index[0][i], b = G.edge_index[1][i]; E.push(`<tr><td>${a} → ${b}</td><td>${SHORT[G.node_features[a][0]]} – ${SHORT[G.node_features[b][0]]}</td><td>${fmt(G.edge_attr[i][0])}</td><td>${fmt(G.edge_attr[i][1])}</td></tr>`); }
  box.innerHTML = `
    <h3>graph_features</h3><table class="gt"><tbody>${gf}</tbody></table>
    <h3>node_features <span class="muted">(${G.node_features.length} nodes · win_* from the window model)</span></h3>
    <div class="scroll-x"><table class="gt"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${nf}</tbody></table></div>
    <h3>edges <span class="muted">(${E.length} undirected · stored both ways)</span></h3>
    ${E.length ? `<table class="gt"><thead><tr><th>i → j</th><th>rooms</th><th>dx</th><th>dy</th></tr></thead><tbody>${E.join("")}</tbody></table>` : `<p class="empty">No links drawn. The MLP / XGBoost surrogates do not read edges; they are kept for the graph export (GNN).</p>`}`;
}
function renderAbout() {
  const m = S.meta.metrics || {};
  const row = (k) => `<tr><td>${k}</td><td>${fmt(m[k]?.unit_sda_R2, 3)}</td><td>${fmt(m[k]?.room_sda_R2, 3)}</td><td>${fmt(m[k]?.map_R2, 3)}</td><td>${fmt(m[k]?.unit_sda_MAE, 3)}</td></tr>`;
  $("#about").innerHTML = `<div class="about">
    <h3>Test-set performance</h3>
    <table class="gt"><thead><tr><th>model</th><th>layout sDA R²</th><th>room sDA R²</th><th>DA map R²</th><th>layout MAE</th></tr></thead><tbody>${S.meta.models.map(row).join("")}</tbody></table>
    <h3>Pipeline</h3>
    <p>The drawn boundary is rasterised at 199.735 px / m and every bubble becomes a node with the dataset features (type, area from r²/Σr² × 1.0017 × A, relative position, N/S/E/W depth to the outline). Window presence per room is predicted by two XGBoost classifiers (AUC ≈ 0.98) unless fixed in the room editor.</p>
    <p><b>Room</b> and <b>layout</b> sDA come from dedicated regressors; the <b>DA map</b> is predicted cell by cell on a 0.1 m grid, and “from map” values count cells above 127/255.</p>
    <p>Valid design range of the training data: WWR<sub>S</sub> 0.20–0.50, WWR<sub>N</sub> 0.20–0.35, 1–10 rooms, residential units. Orientation is fixed to 0° (south-facing) for the models' ORI input.</p>
    <p class="muted" style="font-family:var(--mono);font-size:10.5px">models: ${S.meta.models_dir}</p></div>`;
}

/* ================================================================ library */
let LIB = null, libPage = 0;
const PER = 48;
async function openLibrary() {
  $("#lib").showModal();
  if (!LIB) {
    $("#lib-grid").innerHTML = `<p class="empty" style="padding:20px">loading…</p>`;
    LIB = await (await fetch("/api/library")).json();
    const ns = [...new Set(LIB.map((x) => x.n_rooms).filter(Boolean))].sort((a, b) => a - b);
    $("#lib-rooms").innerHTML = `<option value="">any rooms</option>` + ns.map((n) => `<option>${n}</option>`).join("");
  }
  renderLib();
}
function libFiltered() {
  const q = $("#lib-q").value.trim(), sp = $("#lib-split").value, nr = $("#lib-rooms").value;
  return LIB.filter((x) => (!q || String(x.id).startsWith(q)) && (!sp || x.split === sp) && (!nr || x.n_rooms === +nr));
}
function renderLib() {
  const L = libFiltered(), pages = Math.max(1, Math.ceil(L.length / PER));
  libPage = Math.min(libPage, pages - 1);
  $("#lib-count").textContent = `${L.length} / ${LIB.length}`;
  $("#lib-page").textContent = `${libPage + 1} / ${pages}`;
  $("#lib-grid").innerHTML = L.slice(libPage * PER, (libPage + 1) * PER).map((x) => {
    const [x0, y0, x1, y1] = bbox(x.outline), pad = 0.4;
    return `<div class="card" data-id="${x.id}" title="Load sample #${x.id}">
      <button class="only ghost" data-only="${x.id}">outline</button>
      <svg viewBox="${x0 - pad} ${y0 - pad} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}" preserveAspectRatio="xMidYMid meet"><polygon points="${x.outline.map((p) => p.join(",")).join(" ")}"/></svg>
      <div class="cm"><b>#${x.id}</b><span>${x.split}</span></div>
      <div class="cm"><span>${x.n_rooms ?? "?"} rooms</span><span>${x.area ?? "?"} m²</span></div></div>`;
  }).join("");
}
$("#lib-grid").addEventListener("click", async (e) => {
  const only = e.target.closest("[data-only]");
  const card = e.target.closest(".card"); if (!card) return;
  const id = +card.dataset.id;
  $("#lib").close();
  if (only) {
    const x = LIB.find((q) => q.id === id);
    S.outline = x.outline.map((p) => [...p]); S.sample = null; S.pristine = false;
    updateBoundaryMeta(); fit(); schedule(0);
  } else loadSample(id);
});
async function loadSample(id) {
  status("busy", `loading sample #${id}…`);
  const s = await (await fetch(`/api/library/${id}`)).json();
  s.split = LIB?.find((x) => x.id === id)?.split || "";
  S.outline = s.outline; S.rooms = s.rooms.map(({ id, type, x, y, r }) => ({ id, type, x, y, r }));
  S.links = s.links; S.windows = {}; S.nextId = Math.max(0, ...S.rooms.map((r) => r.id)) + 1;
  S.sample = s; S.pristine = true; S.sel = null; $("#room-editor").hidden = true;
  setParams(s.params);
  updateBoundaryMeta(); fit(); schedule(0);
}
function setParams(p) {
  S.wwr_s = Math.min(0.5, Math.max(0.2, p.wwr_s)); S.wwr_n = Math.min(0.35, Math.max(0.2, p.wwr_n));
  $("#wwr-s").value = S.wwr_s; $("#wwr-n").value = S.wwr_n; syncParamUI();
}
["#lib-q", "#lib-split", "#lib-rooms"].forEach((s) => $(s).addEventListener("input", () => { libPage = 0; renderLib(); }));
$("#lib-prev").onclick = () => { libPage = Math.max(0, libPage - 1); renderLib(); };
$("#lib-next").onclick = () => { libPage++; renderLib(); };
$("#lib-close").onclick = () => $("#lib").close();

/* ================================================================ UI wiring */
function updateBoundaryMeta() {
  const o = S.outline;
  if (o.length < 3) { $("#boundary-meta").innerHTML = `<dt>boundary</dt><dd>none</dd>`; return; }
  const [x0, y0, x1, y1] = bbox(o), A = polyArea(o), P = polyPerim(o);
  $("#boundary-meta").innerHTML = `<dt>W × H</dt><dd>${fmt(x1 - x0)} × ${fmt(y1 - y0)} m</dd><dt>area</dt><dd>${fmt(A, 1)} m²</dd>
    <dt>perimeter</dt><dd>${fmt(P, 1)} m</dd><dt>compactness</dt><dd>${fmt((4 * Math.PI * A) / (P * P), 3)}</dd><dt>vertices</dt><dd>${o.length}</dd>`;
}
function syncParamUI() {
  $("#out-wwr-s").textContent = S.wwr_s.toFixed(2);
  $("#out-wwr-n").textContent = S.wwr_n.toFixed(2);
  $$("#seg-model button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.model));
  $$("#seg-cmap button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.cmap));
}
function wire() {
  $("#palette").innerHTML = S.types.map((t) => `<button data-t="${t.id}"><span class="sw" style="background:rgb(${t.rgb})"></span>${t.name}<kbd style="margin-left:auto">${t.id + 1}</kbd></button>`).join("");
  $$("#palette button").forEach((b) => b.addEventListener("click", () => { pickType(+b.dataset.t); setTool("room"); }));
  $("#ed-type").innerHTML = S.types.map((t) => `<option value="${t.id}">${t.name}</option>`).join("");
  $("#ed-type").onchange = (e) => { roomById(S.sel).type = +e.target.value; syncEditor(); render(); touchGeom(); };
  $("#ed-r").oninput = (e) => { roomById(S.sel).r = +e.target.value; syncEditor(); render(); touchGeom(); };
  $("#ed-delete").onclick = () => deleteRoom(S.sel);
  $$("#ed-win button").forEach((b) => b.addEventListener("click", () => { S.windows[S.sel] = b.dataset.v; syncEditor(); schedule(0); }));
  $$(".tool").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));
  $$("#seg-model button").forEach((b) => b.addEventListener("click", () => { S.model = b.dataset.v; syncParamUI(); schedule(0); }));
  $$("#seg-cmap button").forEach((b) => b.addEventListener("click", () => { S.cmap = b.dataset.v; syncParamUI(); overlayKey = ""; render(); renderMap(); }));
  $$("#seg-mapview button").forEach((b) => b.addEventListener("click", () => { S.mapView = b.dataset.v; renderMap(); }));
  $("#wwr-s").oninput = (e) => { S.wwr_s = +(+e.target.value).toFixed(2); syncParamUI(); schedule(40); };
  $("#wwr-n").oninput = (e) => { S.wwr_n = +(+e.target.value).toFixed(2); syncParamUI(); schedule(40); };
  $("#btn-sample-params").onclick = () => { setParams(S.sample.params); schedule(0); };
  $("#chk-overlay").onchange = (e) => { S.overlay = e.target.checked; render(); };
  $("#chk-bubbles").onchange = (e) => { S.showBubbles = e.target.checked; render(); };
  $("#chk-labels").onchange = (e) => { S.labels = e.target.checked; render(); };
  $("#btn-library").onclick = openLibrary;
  $("#btn-rect-toggle").onclick = () => { $("#rect-form").hidden = !$("#rect-form").hidden; };
  $("#btn-rect").onclick = () => {
    const w = Math.max(3, +$("#rect-w").value), h = Math.max(3, +$("#rect-h").value);
    S.outline = [[0, 0], [w, 0], [w, h], [0, h]]; S.sample = null; S.pristine = false;
    updateBoundaryMeta(); fit(); touchGeom();
  };
  $("#btn-fit").onclick = fit;
  $("#btn-clear").onclick = () => {
    Object.assign(S, { outline: [], rooms: [], links: [], windows: {}, nextId: 1, sel: null, sample: null, pristine: false, draft: null });
    $("#room-editor").hidden = true; updateBoundaryMeta(); clearResult(); setTool("boundary"); status("", "draw a boundary");
  };
  $("#btn-export").onclick = () => {
    if (!S.result) return;
    const data = { ...S.result.graph, inputs: S.result.inputs, prediction: { model: S.result.model, unit: S.result.unit, rooms: S.result.rooms } };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
    a.download = "graph.json"; a.click(); URL.revokeObjectURL(a.href);
  };
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".tab").forEach((t) => (t.hidden = t.id !== `tab-${b.dataset.tab}`));
  }));
  new ResizeObserver(() => (S.autoFit ? fit() : render())).observe($("#canvas-wrap"));
}

/* ================================================================ boot */
(async function boot() {
  try {
    S.meta = await (await fetch("/api/meta")).json();
  } catch { status("err", "server unreachable — run python app.py"); return; }
  S.types = S.meta.room_types;
  wire(); pickType(0); syncParamUI(); updateBoundaryMeta(); renderAbout(); setTool("select");
  status("ok", "ready");
  LIB = await (await fetch("/api/library")).json();
  loadSample(0);
})();
