"""Side-by-side comparison page: EFIT++ (ground truth) | Challenge-2 hybrid | E4, for one shot.

* ψ maps per frame come from ``/api/frame`` (app.py): contours, LCFS (truth LCFS dotted on the model
  panels), O-point and X-points (saddles of each map), with per-frame R²ψ and D_LCFS.
* Scalar traces and per-frame boundary / point metrics come from the observable audit
  (``pipeline.audit``) of each prediction file, ``outputs/audit/<tag>/frames.csv``:
  q95, βN (the submitted values) and κ, δ_top, δ_bot, V, ℓi, R_axis, Z_axis derived from each ψ map
  with the scorer's own extractors; LCFS Hausdorff, tip gap, O/X-point offsets, flips, tags.
  ``scripts/make_compare_preds.py`` writes the E4 predictions and both audits.
Nothing is written by the viewer.
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache

from app.fec_site.app import SPLIT_TITLE, _page, _score
from pipeline.paths import output_dir

SOURCES = (("hybrid", "FEC Challenge 2: M9 + OSF (c*, s*)"), ("e4", "New calibration: E4 (FAIR-MAST only)"))
AUDIT_TAG = {"hybrid": "{split}_hybrid_shipped", "e4": "compare_{split}_e4"}
TRACES = (("q95", "q95"), ("betaN", "βN"), ("kappa", "κ"), ("tri_top", "δ_top"), ("tri_bot", "δ_bot"),
          ("li", "ℓi"), ("volume", "V (m³)"), ("R_axis", "R_axis (m)"), ("Z_axis", "Z_axis (m)"))
FRAME = (("r2_psi_frame", 1.0), ("rmse_psi", 1.0), ("hausdorff_m", 1000.0), ("tip_gap_m", 1000.0),
         ("opoint_disp_m", 1000.0), ("xpt_disp_m", 1000.0), ("area_symdiff_rel", 1.0), ("topology_flip", 1.0),
         ("xpt_lost", 1.0))


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@lru_cache(maxsize=16)
def _frames(split: str, mode: str) -> dict:
    p = output_dir() / "audit" / AUDIT_TAG[mode].format(split=split) / "frames.csv"
    out: dict = {}
    if p.is_file():
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                out.setdefault(r["key"], []).append(r)
    return out


def api_compare(split: str, key: str) -> dict:
    res = {"split": split, "key": key, "t": None, "gt": {}, "src": {}}
    for mode, label in SOURCES:
        rows = _frames(split, mode).get(key)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: int(r["k"]))
        if res["t"] is None:
            res["t"] = [_num(r["t_ms"]) for r in rows]
            res["gt"] = {q: [_num(r[f"{q}_gt"]) for r in rows] for q, _ in TRACES}
        rec = {"label": label, "trace": {q: [_num(r[f"{q}_pred"]) for r in rows] for q, _ in TRACES},
               "frame": {m: [None if _num(r[m]) is None else round(_num(r[m]) * f, 6) for r in rows] for m, f in FRAME},
               "tags": [r["tags"] for r in rows]}
        sc = _score(split, mode)
        if sc:
            rec["fold_S"] = sc["components"]["S"]
            rec["shot"] = next((s for s in sc["per_shot"] if s["key"] == key), None)
        summ = output_dir() / "audit" / AUDIT_TAG[mode].format(split=split) / "summary.json"
        if summ.is_file():
            rec["fold_audit"] = {k: v for k, v in json.loads(summ.read_text()).items()
                                 if k in ("S", "R2_psi", "R2_q95", "R2_betaN", "one_minus_D_LCFS", "Consistency",
                                          "rate_jump", "p95_hausdorff_mm", "mean_xpt_disp_mm")}
        res["src"][mode] = rec
    return res


JS = r"""
const SPLIT = %SPLIT%, KEY = %KEY%;
const PANELS = ['efit', 'hybrid', 'e4'];
const COLS = {efit: '#111', hybrid: '#1e5aa8', e4: '#2f8f3a'};
const LABEL = {efit: 'Ground truth (EFIT++)', hybrid: 'FEC Challenge 2 (M9 + OSF)', e4: 'New: E4 (FAIR-MAST only)'};
let C = null;
function fmt(v, n=4) { return (v === null || v === undefined) ? '–' : Number(v).toFixed(n); }
async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function mapPlot(div, f, m) {
  const mp = f.maps[m]; if (!mp) { document.getElementById(div).innerHTML = '<p class=meta>n/a</p>'; return; }
  const tr = [{type: 'contour', x: f.R, y: f.Z, z: mp.psi, ncontours: 30, colorscale: 'Viridis',
               contours: {coloring: 'lines'}, line: {width: 1}, showscale: false, name: 'ψ'}];
  if (mp.lcfs) tr.push({x: mp.lcfs.map(p => p[0]), y: mp.lcfs.map(p => p[1]), mode: 'lines',
                        line: {color: m === 'efit' ? '#d62728' : COLS[m], width: 2.5}, name: 'LCFS'});
  if (m !== 'efit' && f.maps.efit && f.maps.efit.lcfs) tr.push({x: f.maps.efit.lcfs.map(p => p[0]),
      y: f.maps.efit.lcfs.map(p => p[1]), mode: 'lines', line: {color: '#d62728', width: 1.3, dash: 'dot'}, name: 'EFIT LCFS'});
  if (mp.o && mp.o[0] !== null) tr.push({x: [mp.o[0]], y: [mp.o[1]], mode: 'markers',
      marker: {symbol: 'circle-open', size: 13, color: '#000', line: {width: 2}}, name: 'O-point'});
  if (mp.x.length) tr.push({x: mp.x.map(p => p[0]), y: mp.x.map(p => p[1]), mode: 'markers',
      marker: {symbol: 'x', size: 12, color: '#d62728'}, name: 'X-point'});
  const met = f.metrics[m] || {};
  const k = f.k, s = C.src[m];
  let title = LABEL[m];
  if (m !== 'efit') title += `<br>R²ψ ${fmt(met.r2_psi_frame)} · D ${fmt(met.D_LCFS)} · Haus ${fmt(s ? s.frame.hausdorff_m[k] : null, 1)} mm`;
  Plotly.react(div, tr, {title: {text: title, font: {size: 13}}, margin: {l: 40, r: 10, t: 52, b: 30}, height: 560,
      xaxis: {title: 'R (m)', range: [0.06, 2.0]}, yaxis: {title: 'Z (m)', scaleanchor: 'x', range: [-2, 2]},
      showlegend: m === 'efit', legend: {x: 1, y: 1, xanchor: 'right', font: {size: 10}}}, {displayModeBar: false});
}
const ROWS = [['q95', 'q95', 3], ['betaN', 'βN', 3], ['kappa', 'κ', 3], ['tri_top', 'δ_top', 3], ['tri_bot', 'δ_bot', 3],
              ['li', 'ℓi', 3], ['volume', 'V (m³)', 2], ['R_axis', 'R_axis (m)', 3], ['Z_axis', 'Z_axis (m)', 3]];
const FROWS = [['r2_psi_frame', 'R²ψ (frame)', 4], ['rmse_psi', 'RMSE ψ (Wb)', 5], ['hausdorff_m', 'LCFS Hausdorff (mm)', 1],
               ['tip_gap_m', 'LCFS tip gap (mm)', 1], ['opoint_disp_m', 'O-point offset (mm)', 1], ['xpt_disp_m', 'X-point offset (mm)', 1],
               ['area_symdiff_rel', 'LCFS area mismatch', 4], ['topology_flip', 'limited/diverted flip', 0], ['xpt_lost', 'X-point lost', 0]];
function cards(f) {
  const k = f.k, ms = ['hybrid', 'e4'].filter(m => C.src[m]);
  let h = `<table><tr><th>frame ${k + 1} · t = ${f.t.toFixed(1)} ms</th><th>${LABEL.efit}</th>` + ms.map(m => '<th>' + LABEL[m] + '</th>').join('') + '</tr>';
  for (const [q, lab, n] of ROWS) h += `<tr><td>${lab}</td><td>${fmt(C.gt[q][k], n)}</td>` + ms.map(m => '<td>' + fmt(C.src[m].trace[q][k], n) + '</td>').join('') + '</tr>';
  h += `<tr><td>D_LCFS</td><td></td>` + ms.map(m => '<td>' + fmt((f.metrics[m] || {}).D_LCFS, 4) + '</td>').join('') + '</tr>';
  for (const [q, lab, n] of FROWS) h += `<tr><td>${lab}</td><td></td>` + ms.map(m => '<td>' + fmt(C.src[m].frame[q][k], n) + '</td>').join('') + '</tr>';
  h += `<tr><td>audit tags</td><td></td>` + ms.map(m => '<td>' + (C.src[m].tags[k] || '–') + '</td>').join('') + '</tr></table>';
  document.getElementById('framecard').innerHTML = h;
}
function shotcard() {
  const ms = ['hybrid', 'e4'].filter(m => C.src[m]);
  let h = '<table><tr><th>score (FEC scorer v3.1.0)</th>' + ms.map(m => '<th>' + LABEL[m] + '</th>').join('') + '</tr>';
  const rows = [['S (3-shot fold)', m => (C.src[m].fold_audit || {}).S, 4], ['R²ψ (fold)', m => (C.src[m].fold_audit || {}).R2_psi, 4],
    ['R²q95 (fold)', m => (C.src[m].fold_audit || {}).R2_q95, 4], ['R²βN (fold)', m => (C.src[m].fold_audit || {}).R2_betaN, 4],
    ['1−D_LCFS (fold)', m => (C.src[m].fold_audit || {}).one_minus_D_LCFS, 4], ['Consistency (fold)', m => (C.src[m].fold_audit || {}).Consistency, 4],
    ['R²ψ (this shot)', m => (C.src[m].shot || {}).r2_psi, 4], ['D_LCFS (this shot)', m => (C.src[m].shot || {}).dlcfs, 4]];
  for (const [lab, g, n] of rows) h += '<tr><td>' + lab + '</td>' + ms.map(m => '<td>' + fmt(g(m), n) + '</td>').join('') + '</tr>';
  document.getElementById('shotcard').innerHTML = h + '</table>';
}
const TS = [];
function series() {
  const t = C.t, lay = (title) => ({title: {text: title, font: {size: 13}}, margin: {l: 45, r: 10, t: 30, b: 32}, height: 250,
      xaxis: {title: 't (ms)'}, legend: {font: {size: 9}, orientation: 'h', y: -0.35}, showlegend: false});
  const grid = document.getElementById('ts');
  ROWS.concat([['hausdorff_m', 'LCFS Hausdorff (mm)'], ['xpt_disp_m', 'X-point offset (mm)'], ['r2_psi_frame', 'R²ψ (frame)']]).forEach(([q, lab], i) => {
    const id = 'ts_' + q; const d = document.createElement('div'); d.id = id; d.className = 'plot'; grid.appendChild(d); TS.push(id);
    const tr = [];
    if (C.gt[q]) tr.push({x: t, y: C.gt[q], name: LABEL.efit, mode: 'lines', line: {color: COLS.efit, width: 2.5}});
    for (const m of ['hybrid', 'e4']) if (C.src[m]) {
      const y = C.gt[q] ? C.src[m].trace[q] : C.src[m].frame[q];
      tr.push({x: t, y, name: LABEL[m], mode: 'lines', line: {color: COLS[m], width: 2, dash: m === 'hybrid' ? 'dash' : 'solid'}});
    }
    Plotly.newPlot(id, tr, Object.assign(lay(lab), {showlegend: i === 0}), {displayModeBar: false});
  });
}
async function showFrame(k) {
  const f = await getJSON(`/api/frame?split=${SPLIT}&key=${KEY}&t=${k}`);
  document.getElementById('tlabel').textContent = `frame ${f.k + 1} / ${C.t.length} · t = ${f.t.toFixed(1)} ms`;
  PANELS.forEach(m => mapPlot('map_' + m, f, m));
  cards(f);
  TS.forEach(id => Plotly.relayout(id, {shapes: [{type: 'line', x0: f.t, x1: f.t, yref: 'paper', y0: 0, y1: 1, line: {color: '#888', dash: 'dot'}}]}));
}
(async () => {
  C = await getJSON(`/api/compare?split=${SPLIT}&key=${KEY}`);
  shotcard(); series();
  const sl = document.getElementById('scrub'); sl.max = C.t.length - 1;
  let pend = null; sl.oninput = () => { clearTimeout(pend); pend = setTimeout(() => showFrame(+sl.value), 120); };
  document.getElementById('prev').onclick = () => { sl.value = Math.max(0, +sl.value - 1); showFrame(+sl.value); };
  document.getElementById('next').onclick = () => { sl.value = Math.min(+sl.max, +sl.value + 1); showFrame(+sl.value); };
  sl.value = Math.floor(C.t.length / 2); showFrame(+sl.value);
})();
"""


def page_compare(split: str, key: str) -> str:
    from app.fec_site.app import _keys
    nav = " · ".join(f"<a href='/compare/{split}/{k}'>{k}</a>" if k != key else f"<b>{k}</b>" for k in _keys(split))
    body = f"""<h1>{key}: truth vs Challenge 2 vs E4</h1><p class='meta'>{SPLIT_TITLE.get(split, split)} · {nav} ·
<a href='/shot/{split}/{key}'>full shot page</a></p>
<div class='scrub'><button class='btn' id='prev'>◀</button><input type='range' id='scrub' min='0' value='0'>
<button class='btn' id='next'>▶</button><span id='tlabel'></span></div>
<div class='row3'><div id='map_efit' class='plot'></div><div id='map_hybrid' class='plot'></div><div id='map_e4' class='plot'></div></div>
<div class='row2' style='margin-top:10px'><div id='framecard'></div><div id='shotcard'></div></div>
<h2>Time series (EFIT++ black, Challenge 2 blue dashed, E4 green)</h2><div class='row3' id='ts'></div>
<p class='meta'>Challenge 2 = the M9_win50 ensemble (i101/i202/i303) with OSF <code>mast_12_osf_0.0.12</code> and the
post-hoc factors c* = 1.062242, s* = 1.078508, the MAST recipe of the Phase-2 Final submission. E4 = the FAIR-MAST-only
route-guided ensemble (paper Section 10.6), q95 and βN from its scalar heads. κ, δ, V, ℓi and the axis are derived
from each ψ map by the scorer's extractors (pipeline.audit); X-points are saddles of each map.</p>
<script>{JS.replace('%SPLIT%', json.dumps(split)).replace('%KEY%', json.dumps(key))}</script>"""
    return _page(f"compare {key}", body, [("/", "home"), (f"/split/{split}", split), (f"/compare/{split}/{key}", "compare")])
