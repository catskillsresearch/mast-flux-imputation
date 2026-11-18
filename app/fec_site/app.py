"""MAST-only viewer: EFIT (truth) vs OSF vs M9+OSF.

Pages
    /                         splits, available predictions, fold scorecards
    /split/<split>            per-shot table (frames, per-shot R2_psi and D_LCFS per model)
    /shot/<split>/<key>       psi contours + LCFS + O/X-points + equal-Te pairs for each model,
                              residual maps, error histograms, q95 / betaN traces, time scrubber,
                              per-frame scorecards
JSON
    /api/series?split=&key=   time series, per-frame metrics, histograms for one shot
    /api/frame?split=&key=&t= one frame: psi maps, residuals, LCFS, O/X-points, Te pairs
    /healthz

Predictions are read from ``outputs/<split>_<mode>.npz`` (pipeline.predict) and fold scores from
``outputs/<split>_<mode>.score.json`` (pipeline.score). Nothing is written.

    python -m app.fec_site [--port 8770] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "fec_scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contour import symmetric_hausdorff  # noqa: E402
from derive import magnetic_axis  # noqa: E402
from lcfs import extract_lcfs_with_sign, major_radius  # noqa: E402

from pipeline.datasets import SPLITS, load_shot, split_keys  # noqa: E402
from pipeline.paths import fair_mast_labels_dir, output_dir  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MACHINE = "MAST"
MODELS = (("osf", "OSF"), ("hybrid", "M9 + OSF"))
SPLIT_TITLE = {"demo": "FEC labelled demo (3 shots)",
               "holdout": "FAIR-MAST sealed holdout (seed 42, diverted frames)",
               "public_test": "FEC mast_public_test (unlabelled)"}
PLOTLY = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def _env():
    z = np.load(REPO / "fec_scoring" / "masks" / "mast_envelope.npz")
    R = np.asarray(z["grid_R"], float)
    Z = np.asarray(z["grid_Z"], float)
    m = np.asarray(z["mask_coarse"]).astype(bool)
    return R, Z, m, m.astype(float)


R, Z, MASK, MASK_F = _env()


# ---------------------------------------------------------------- data access (read-only, cached)

@lru_cache(maxsize=16)
def _pred_meta(split: str, mode: str) -> dict | None:
    p = output_dir() / f"{split}_{mode}.json"
    return json.loads(p.read_text()) if p.is_file() else None


@lru_cache(maxsize=8)
def _pred_npz(split: str, mode: str):
    p = output_dir() / f"{split}_{mode}.npz"
    return np.load(p) if p.is_file() else None


@lru_cache(maxsize=16)
def _score(split: str, mode: str) -> dict | None:
    p = output_dir() / f"{split}_{mode}.score.json"
    return json.loads(p.read_text()) if p.is_file() else None


@lru_cache(maxsize=12)
def _shot(split: str, key: str):
    return load_shot(split, key)


def _pred(split: str, mode: str, key: str) -> dict | None:
    meta, z = _pred_meta(split, mode), _pred_npz(split, mode)
    if meta is None or z is None or key not in meta["keys"]:
        return None
    i = meta["keys"].index(key)
    return {"psi": z[f"shot_{i:04d}_psirz"].astype(np.float64),
            "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
            "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)}


@lru_cache(maxsize=8)
def _labels(key: str) -> dict | None:
    p = fair_mast_labels_dir() / f"{key}.npz"
    return dict(np.load(p)) if p.is_file() else None


def _keys(split: str) -> list[str]:
    for mode, _ in MODELS:
        m = _pred_meta(split, mode)
        if m:
            return list(m["keys"])
    try:
        return split_keys(split)
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------- geometry helpers

def _fin(v):
    v = float(v)
    return v if math.isfinite(v) else None


def _r(a, nd=5):
    a = np.asarray(a, float)
    return [[(None if not math.isfinite(x) else round(float(x), nd)) for x in row] for row in a]


def _lcfs(psi: np.ndarray):
    c, _ = extract_lcfs_with_sign(psi, R, Z, MACHINE, MASK, MASK_F, n_points=256, axis_sign=+1.0)
    return None if c is None else np.asarray(c, float)


def _opoint(psi):
    ra, za, _, _ = magnetic_axis(psi, R, Z, MACHINE, MASK, +1.0)
    return [_fin(ra), _fin(za)]


def _xpoints(psi: np.ndarray) -> list[list[float]]:
    """Saddle points of psi inside the vessel: local minima of |grad psi| with det(Hessian) < 0;
    the best one in each half-plane (|Z| > 0.3 m)."""
    dz, dr = Z[1] - Z[0], R[1] - R[0]
    gz, gr = np.gradient(psi, dz, dr)
    g2 = gz * gz + gr * gr
    gzz, gzr = np.gradient(gz, dz, dr)
    _, grr = np.gradient(gr, dz, dr)
    det = gzz * grr - gzr * gzr
    out = []
    for upper in (True, False):
        best, bv = None, np.inf
        for iz in range(2, len(Z) - 2):
            if (Z[iz] > 0.3) != upper or abs(Z[iz]) <= 0.3:
                continue
            for ir in range(2, len(R) - 2):
                if not MASK[iz, ir] or det[iz, ir] >= 0:
                    continue
                v = g2[iz, ir]
                if v <= g2[iz - 1:iz + 2, ir - 1:ir + 2].min() and v < bv:
                    best, bv = (R[ir], Z[iz]), v
        if best is not None:
            out.append([round(float(best[0]), 4), round(float(best[1]), 4)])
    return out


def _sample_psi(r, z, psi, rq, zq):
    if not (np.isfinite(rq) and np.isfinite(zq)):
        return float("nan")
    ir = int(np.clip(np.searchsorted(r, rq) - 1, 0, r.size - 2))
    iz = int(np.clip(np.searchsorted(z, zq) - 1, 0, z.size - 2))
    tr = (rq - r[ir]) / (r[ir + 1] - r[ir] + 1e-30)
    tz = (zq - z[iz]) / (z[iz + 1] - z[iz] + 1e-30)
    return float((1 - tz) * ((1 - tr) * psi[iz, ir] + tr * psi[iz, ir + 1])
                 + tz * ((1 - tr) * psi[iz + 1, ir] + tr * psi[iz + 1, ir + 1]))


def _te_pairs(row: dict, t_ms: float, psis: dict) -> dict:
    """Equal-T_e pairs (catskills_fec fec_site/diag.py): on the core Thomson profile nearest t,
    the chords where T_e = 0.3/0.5/0.7 x peak on the inboard and outboard sides should sit on one
    flux surface; a map passes a pair when |psi(r1) - psi(r2)| < 2% of psi(peak) - psi(inboard)."""
    empty = {"pairs": [], "valid": {}}
    if row.get("thomson_core_times") is None or row.get("thomson_core_Te") is None:
        return empty
    tt = np.asarray(row["thomson_core_times"], float).reshape(-1)
    te = np.asarray(row["thomson_core_Te"], float)
    rr = np.asarray(row.get("thomson_core_R", []), float).reshape(-1)
    if te.ndim != 2 or tt.size == 0 or rr.size == 0:
        return empty
    k = int(np.argmin(np.abs(tt - t_ms)))
    prof = te[k, : min(te.shape[1], rr.size)]
    rx = rr[: prof.size]
    keep = np.isfinite(prof) & np.isfinite(rx)
    if keep.sum() < 6:
        return empty
    prof, rx = prof[keep], rx[keep]
    peak = float(np.nanmax(prof))
    if peak <= 0:
        return empty
    o = np.argsort(rx)
    rx, prof = rx[o], prof[o]
    ip = int(np.argmax(prof))
    pairs, ok = [], {m: [] for m in psis}
    for frac in (0.3, 0.5, 0.7):
        lv = frac * peak
        left = np.where(np.diff(np.sign(prof[: ip + 1] - lv)))[0]
        right = np.where(np.diff(np.sign(prof[ip:] - lv)))[0]
        if left.size == 0 or right.size == 0:
            continue
        r1, r2 = float(rx[left[-1]]), float(rx[ip + right[0]])
        pairs.append({"frac": frac, "r": [r1, r2], "z": [0.0, 0.0]})
        for m, psi in psis.items():
            rng = max(abs(_sample_psi(R, Z, psi, float(rx[ip]), 0.0) - _sample_psi(R, Z, psi, float(rx[0]), 0.0)), 1e-9)
            p1, p2 = _sample_psi(R, Z, psi, r1, 0.0), _sample_psi(R, Z, psi, r2, 0.0)
            if np.isfinite(p1) and np.isfinite(p2):
                ok[m].append(abs(p1 - p2) / rng < 0.02)
    return {"pairs": pairs, "valid": {m: (None if not v else float(np.mean(v))) for m, v in ok.items()},
            "t_thomson_ms": float(tt[k])}


def _r2(gt, pr):
    g, p = np.asarray(gt, float).ravel(), np.asarray(pr, float).ravel()
    m = np.isfinite(g) & np.isfinite(p)
    if m.sum() < 2:
        return None
    ss = float(np.sum((g[m] - g[m].mean()) ** 2))
    return None if ss <= 0 else round(1.0 - float(np.sum((g[m] - p[m]) ** 2)) / ss, 6)


# ---------------------------------------------------------------- JSON endpoints

def api_series(split: str, key: str) -> dict:
    shot = _shot(split, key)
    t = np.asarray(shot.row["efit_times"], float).reshape(-1)
    out = {"split": split, "key": key, "t": [round(float(x), 3) for x in t], "models": {},
           "gt": None, "meta": shot.meta}
    gt = shot.gt
    if gt is not None:
        out["gt"] = {"q95": [_fin(v) for v in gt["q95"]], "betaN": [_fin(v) for v in gt["betaN"]]}
    for mode, label in MODELS:
        p = _pred(split, mode, key)
        if p is None:
            continue
        rec = {"label": label, "q95": [_fin(v) for v in p["q95"]], "betaN": [_fin(v) for v in p["betaN"]]}
        if gt is not None:
            res = (p["psi"] - gt["psi"])[:, MASK]
            rec["r2_frame"] = [_r2(gt["psi"][k], p["psi"][k]) for k in range(len(t))]
            rec["rmse_frame"] = [round(float(np.sqrt(np.nanmean(res[k] ** 2))), 6) for k in range(len(t))]
            lim = float(np.nanpercentile(np.abs(res), 99.5)) or 1e-6
            h, e = np.histogram(res[np.isfinite(res)], bins=60, range=(-lim, lim))
            rec["hist_psi"] = {"x": [round(float(x), 6) for x in 0.5 * (e[1:] + e[:-1])], "y": h.tolist()}
            for s in ("q95", "betaN"):
                d = p[s] - gt[s]
                d = d[np.isfinite(d)]
                if d.size:
                    lim = float(np.percentile(np.abs(d), 99)) or 1e-6
                    h, e = np.histogram(d, bins=30, range=(-lim, lim))
                    rec[f"hist_{s}"] = {"x": [round(float(x), 4) for x in 0.5 * (e[1:] + e[:-1])], "y": h.tolist()}
            rec["r2_psi_shot"] = _r2(gt["psi"], p["psi"])
        sc = _score(split, mode)
        if sc:
            ps = next((s for s in sc["per_shot"] if s["key"] == key), None)
            rec["score_shot"] = ps
        out["models"][mode] = rec
    return out


def api_frame(split: str, key: str, k: int) -> dict:
    shot = _shot(split, key)
    t = np.asarray(shot.row["efit_times"], float).reshape(-1)
    k = int(np.clip(k, 0, len(t) - 1))
    psis = {}
    if shot.gt is not None:
        psis["efit"] = shot.gt["psi"][k]
    preds = {}
    for mode, _ in MODELS:
        p = _pred(split, mode, key)
        if p is not None:
            psis[mode] = p["psi"][k]
            preds[mode] = p
    out = {"k": k, "t": float(t[k]), "R": [round(float(x), 4) for x in R], "Z": [round(float(x), 4) for x in Z],
           "maps": {}, "residual": {}, "metrics": {}}
    ref_c = None
    for m, psi in psis.items():
        c = _lcfs(psi)
        if m == "efit":
            ref_c = c
        out["maps"][m] = {"psi": _r(psi), "lcfs": None if c is None else _r(c[::2], 4),
                          "o": _opoint(psi), "x": _xpoints(psi)}
    rgeo = major_radius(ref_c) if ref_c is not None else None
    for m in preds:
        met = {"q95": _fin(preds[m]["q95"][k]), "betaN": _fin(preds[m]["betaN"][k])}
        if "efit" in psis:
            d = psis[m] - psis["efit"]
            out["residual"][m] = _r(np.where(MASK, d, np.nan), 6)
            met["r2_psi_frame"] = _r2(psis["efit"], psis[m])
            met["rmse_psi"] = round(float(np.sqrt(np.mean(d[MASK] ** 2))), 6)
            c = _lcfs(psis[m])
            if ref_c is not None:
                met["D_LCFS"] = 1.0 if c is None else round(min(1.0, symmetric_hausdorff(c, ref_c) / rgeo), 6)
        out["metrics"][m] = met
    if shot.gt is not None:
        out["metrics"]["efit"] = {"q95": _fin(shot.gt["q95"][k]), "betaN": _fin(shot.gt["betaN"][k])}
    lab = _labels(key) if split == "holdout" else None
    if lab is not None:
        sel = np.flatnonzero(lab["diverted"])
        j = sel[k] if k < sel.size else None
        if j is not None:
            out["efm_xpoints"] = [[_fin(lab[f"xpoint{i}_rc"][j]), _fin(lab[f"xpoint{i}_zc"][j])]
                                  for i in (1, 2) if np.isfinite(lab[f"xpoint{i}_rc"][j])]
    out["te_pairs"] = _te_pairs(shot.row, float(t[k]), psis)
    return out


# ---------------------------------------------------------------- HTML

def _page(title: str, body: str, crumbs: list[tuple[str, str]] = ()) -> str:
    nav = " / ".join(f'<a href="{h}">{t}</a>' for h, t in crumbs)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<link rel='stylesheet' href='/static/style.css'>{PLOTLY}</head><body><main>"
            f"<nav class='crumb'>{nav}</nav>{body}</main></body></html>")


def _fmt(v, nd=4):
    return "–" if v is None else f"{v:.{nd}f}"


def _scorecard(split: str, mode: str, label: str) -> str:
    sc = _score(split, mode)
    if not sc:
        meta = _pred_meta(split, mode)
        state = "predictions present, not scored" if meta else "no predictions"
        return f"<div class='card'><h3>{label}</h3><p class='meta'>{state}</p></div>"
    c, r = sc["components"], sc["result"]
    rows = "".join(f"<tr><td>{k}</td><td>{_fmt(c[k])}</td></tr>" for k in
                   ("R2_psi", "R2_q95", "R2_betaN", "one_minus_D_LCFS", "Consistency",
                    "R2_R_axis", "R2_Z_axis", "R2_kappa", "R2_tri_top", "R2_tri_bot", "R2_volume", "R2_li"))
    return (f"<div class='card'><h3>{label}</h3><div class='big'>S = {c['S']:.4f}</div>"
            f"<p class='meta'>{r['n_shots']} shots · {r['n_frames']} frames · psi_sign {r['psi_sign']:+d} · "
            f"LCFS fail {r['lcfs_fail_frac']:.2%}</p><table>{rows}</table></div>")


def page_index() -> str:
    parts = ["<h1>MAST poloidal-flux imputation</h1>",
             "<p class='lead'>EFIT++ (truth) vs the OSF Grad–Shafranov solver vs the M9 + OSF hybrid. "
             "No magnetic-probe or flux-loop input is used by either model.</p>"]
    for split in SPLITS:
        keys = _keys(split)
        parts.append(f"<h2><a href='/split/{split}'>{SPLIT_TITLE[split]}</a> "
                     f"<span class='meta'>({len(keys)} shots)</span></h2><div class='cards'>")
        parts += [_scorecard(split, m, lab) for m, lab in MODELS]
        parts.append("</div>")
    return _page("MAST flux imputation", "".join(parts), [("/", "home")])


def page_split(split: str) -> str:
    keys = _keys(split)
    sc = {m: _score(split, m) for m, _ in MODELS}
    per = {m: ({s["key"]: s for s in sc[m]["per_shot"]} if sc[m] else {}) for m, _ in MODELS}
    head = "<tr><th>shot</th><th>frames</th>" + "".join(
        f"<th>{lab} R²ψ</th><th>{lab} D_LCFS</th>" for _, lab in MODELS) + "</tr>"
    rows = []
    for k in keys:
        T = next((per[m][k]["T"] for m, _ in MODELS if k in per[m]), "")
        cells = "".join(f"<td>{_fmt(per[m].get(k, {}).get('r2_psi'))}</td><td>{_fmt(per[m].get(k, {}).get('dlcfs'))}</td>"
                        for m, _ in MODELS)
        rows.append(f"<tr><td><a href='/shot/{split}/{k}'>{k}</a></td><td>{T}</td>{cells}</tr>")
    body = (f"<h1>{SPLIT_TITLE[split]}</h1><div class='cards'>"
            + "".join(_scorecard(split, m, lab) for m, lab in MODELS) + "</div>"
            f"<table>{head}{''.join(rows)}</table>")
    return _page(split, body, [("/", "home"), (f"/split/{split}", split)])


SHOT_JS = r"""
const SPLIT = %SPLIT%, KEY = %KEY%;
const COLS = {efit: '#111', osf: '#b5651d', hybrid: '#1e5aa8'};
const LABEL = {efit: 'EFIT++', osf: 'OSF', hybrid: 'M9 + OSF'};
let SERIES = null;
function fmt(v, n=4) { return (v === null || v === undefined) ? '–' : Number(v).toFixed(n); }
async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function mapPlot(div, f, m) {
  const mp = f.maps[m]; if (!mp) { document.getElementById(div).innerHTML = '<p class=meta>n/a</p>'; return; }
  const tr = [{type: 'contour', x: f.R, y: f.Z, z: mp.psi, ncontours: 30, colorscale: 'Viridis',
               contours: {coloring: 'lines'}, line: {width: 1}, showscale: false, name: 'ψ'}];
  if (mp.lcfs) tr.push({x: mp.lcfs.map(p => p[0]), y: mp.lcfs.map(p => p[1]), mode: 'lines',
                        line: {color: '#d62728', width: 2.5}, name: 'LCFS'});
  if (m !== 'efit' && f.maps.efit && f.maps.efit.lcfs) tr.push({x: f.maps.efit.lcfs.map(p => p[0]),
      y: f.maps.efit.lcfs.map(p => p[1]), mode: 'lines', line: {color: '#111', width: 1.2, dash: 'dot'}, name: 'EFIT LCFS'});
  if (mp.o && mp.o[0] !== null) tr.push({x: [mp.o[0]], y: [mp.o[1]], mode: 'markers',
      marker: {symbol: 'circle-open', size: 13, color: '#000', line: {width: 2}}, name: 'O-point'});
  if (mp.x.length) tr.push({x: mp.x.map(p => p[0]), y: mp.x.map(p => p[1]), mode: 'markers',
      marker: {symbol: 'x', size: 12, color: '#d62728'}, name: 'X-point (saddle)'});
  if (m === 'efit' && f.efm_xpoints && f.efm_xpoints.length) tr.push({x: f.efm_xpoints.map(p => p[0]),
      y: f.efm_xpoints.map(p => p[1]), mode: 'markers', marker: {symbol: 'x-thin-open', size: 16, color: '#7a00a8', line: {width: 2}},
      name: 'EFIT X-point (efm)'});
  const tp = f.te_pairs.pairs || [];
  tp.forEach((p, i) => tr.push({x: p.r, y: p.z, mode: 'lines+markers', line: {color: '#e377c2', width: 2},
      marker: {size: 7}, name: 'equal-Te ' + p.frac, showlegend: i === 0}));
  const v = (f.te_pairs.valid || {})[m];
  const met = f.metrics[m] || {};
  let title = LABEL[m] + (met.r2_psi_frame !== undefined ? `  R²ψ ${fmt(met.r2_psi_frame)}  D ${fmt(met.D_LCFS)}` : '')
            + (v !== undefined && v !== null ? `  Te-pairs ${fmt(v, 2)}` : '');
  Plotly.react(div, tr, {title: {text: title, font: {size: 13}}, margin: {l: 40, r: 10, t: 34, b: 30}, height: 520,
      xaxis: {title: 'R (m)', range: [0.06, 2.0]}, yaxis: {title: 'Z (m)', scaleanchor: 'x', range: [-2, 2]},
      showlegend: m === 'efit', legend: {x: 1, y: 1, xanchor: 'right', font: {size: 10}}}, {displayModeBar: false});
}
function resPlot(div, f, m) {
  const r = f.residual[m]; if (!r) { document.getElementById(div).innerHTML = '<p class=meta>no truth</p>'; return; }
  let lim = 0; r.forEach(row => row.forEach(v => { if (v !== null) lim = Math.max(lim, Math.abs(v)); }));
  Plotly.react(div, [{type: 'heatmap', x: f.R, y: f.Z, z: r, colorscale: 'RdBu', reversescale: true, zmin: -lim, zmax: lim,
      colorbar: {title: 'Wb', thickness: 10}}], {title: {text: LABEL[m] + ' − EFIT  (ψ residual, vessel mask)', font: {size: 13}},
      margin: {l: 40, r: 10, t: 34, b: 30}, height: 420, xaxis: {title: 'R (m)'}, yaxis: {title: 'Z (m)', scaleanchor: 'x'}},
      {displayModeBar: false});
}
function cards(f) {
  const ms = Object.keys(f.metrics);
  let h = '<table><tr><th>frame ' + f.k + ' · t = ' + f.t.toFixed(1) + ' ms</th>' + ms.map(m => '<th>' + LABEL[m] + '</th>').join('') + '</tr>';
  for (const [k, lab, n] of [['q95', 'q95', 3], ['betaN', 'βN', 3], ['r2_psi_frame', 'R²ψ (frame)', 4], ['rmse_psi', 'RMSE ψ (Wb)', 5], ['D_LCFS', 'D_LCFS', 4]])
    h += '<tr><td>' + lab + '</td>' + ms.map(m => '<td>' + fmt((f.metrics[m] || {})[k], n) + '</td>').join('') + '</tr>';
  h += '</table>';
  document.getElementById('framecard').innerHTML = h;
}
async function showFrame(k) {
  const f = await getJSON(`/api/frame?split=${SPLIT}&key=${KEY}&t=${k}`);
  document.getElementById('tlabel').textContent = `frame ${f.k + 1} / ${SERIES.t.length} · t = ${f.t.toFixed(1)} ms`;
  ['efit', 'osf', 'hybrid'].forEach(m => mapPlot('map_' + m, f, m));
  ['osf', 'hybrid'].forEach(m => resPlot('res_' + m, f, m));
  cards(f);
  ['ts_q', 'ts_b', 'ts_r2'].forEach(id => Plotly.relayout(id, {shapes: [{type: 'line', x0: f.t, x1: f.t, yref: 'paper', y0: 0, y1: 1, line: {color: '#888', dash: 'dot'}}]}));
}
function series() {
  const S = SERIES, t = S.t, tr = (y, name, color, dash) => ({x: t, y, name, mode: 'lines', line: {color, dash, width: 2}});
  const lay = (title) => ({title: {text: title, font: {size: 13}}, margin: {l: 45, r: 10, t: 34, b: 35}, height: 300, xaxis: {title: 't (ms)'}, legend: {font: {size: 10}}});
  const q = [], b = [], r2 = [];
  if (S.gt) { q.push(tr(S.gt.q95, 'EFIT', COLS.efit)); b.push(tr(S.gt.betaN, 'EFIT', COLS.efit)); }
  for (const [m, rec] of Object.entries(S.models)) {
    q.push(tr(rec.q95, rec.label, COLS[m])); b.push(tr(rec.betaN, rec.label, COLS[m]));
    if (rec.r2_frame) r2.push(tr(rec.r2_frame, rec.label, COLS[m]));
  }
  Plotly.newPlot('ts_q', q, lay('q95'), {displayModeBar: false});
  Plotly.newPlot('ts_b', b, lay('βN'), {displayModeBar: false});
  Plotly.newPlot('ts_r2', r2, Object.assign(lay('per-frame R²ψ'), {yaxis: {range: [0, 1]}}), {displayModeBar: false});
  const hs = (key, title, id) => {
    const tr = []; for (const [m, rec] of Object.entries(S.models)) if (rec[key]) tr.push({type: 'bar', x: rec[key].x, y: rec[key].y, name: rec.label, marker: {color: COLS[m]}, opacity: 0.6});
    Plotly.newPlot(id, tr, Object.assign(lay(title), {barmode: 'overlay', xaxis: {title: 'prediction − EFIT'}}), {displayModeBar: false});
  };
  hs('hist_psi', 'ψ residual histogram (all frames, vessel mask)', 'h_psi');
  hs('hist_q95', 'q95 error histogram', 'h_q');
  hs('hist_betaN', 'βN error histogram', 'h_b');
  let h = '<table><tr><th>shot</th>' + Object.values(S.models).map(r => '<th>' + r.label + '</th>').join('') + '</tr>';
  for (const [k, lab] of [['r2_psi_shot', 'R²ψ (shot, own mean)'], ['score_r2', 'R²ψ (shot, fold mean)'], ['score_d', 'D_LCFS (shot mean)']]) {
    h += '<tr><td>' + lab + '</td>' + Object.values(S.models).map(r => {
      const v = k === 'score_r2' ? (r.score_shot || {}).r2_psi : k === 'score_d' ? (r.score_shot || {}).dlcfs : r[k];
      return '<td>' + fmt(v) + '</td>'; }).join('') + '</tr>';
  }
  document.getElementById('shotcard').innerHTML = h + '</table>';
}
(async () => {
  SERIES = await getJSON(`/api/series?split=${SPLIT}&key=${KEY}`);
  series();
  const sl = document.getElementById('scrub'); sl.max = SERIES.t.length - 1;
  let pend = null; sl.oninput = () => { clearTimeout(pend); pend = setTimeout(() => showFrame(+sl.value), 120); };
  document.getElementById('prev').onclick = () => { sl.value = Math.max(0, +sl.value - 1); showFrame(+sl.value); };
  document.getElementById('next').onclick = () => { sl.value = Math.min(+sl.max, +sl.value + 1); showFrame(+sl.value); };
  sl.value = Math.floor(SERIES.t.length / 2); showFrame(+sl.value);
})();
"""


def page_shot(split: str, key: str) -> str:
    body = f"""<h1>{key}</h1><p class='meta'>{SPLIT_TITLE[split]}</p>
<div class='scrub'><button class='btn' id='prev'>◀</button><input type='range' id='scrub' min='0' value='0'>
<button class='btn' id='next'>▶</button><span id='tlabel'></span></div>
<div class='row3'><div id='map_efit' class='plot'></div><div id='map_osf' class='plot'></div><div id='map_hybrid' class='plot'></div></div>
<div class='row2' style='margin-top:10px'><div id='framecard'></div><div id='shotcard'></div></div>
<h2>Residual maps</h2><div class='row2'><div id='res_osf' class='plot'></div><div id='res_hybrid' class='plot'></div></div>
<h2>Time traces</h2><div class='row3'><div id='ts_q' class='plot'></div><div id='ts_b' class='plot'></div><div id='ts_r2' class='plot'></div></div>
<h2>Error histograms</h2><div class='row3'><div id='h_psi' class='plot'></div><div id='h_q' class='plot'></div><div id='h_b' class='plot'></div></div>
<p class='meta'>X-points: saddle points of each map (min |∇ψ|, det H &lt; 0, |Z| &gt; 0.3 m); on the holdout the
EFIT++ efm X-points are overlaid. Equal-T<sub>e</sub> pairs: core Thomson chords at 0.3/0.5/0.7 × peak T<sub>e</sub> on either
side of the peak should lie on one flux surface; the fraction of pairs within 2 % is shown per map.</p>
<script>{SHOT_JS.replace('%SPLIT%', json.dumps(split)).replace('%KEY%', json.dumps(key))}</script>"""
    return _page(key, body, [("/", "home"), (f"/split/{split}", split), (f"/shot/{split}/{key}", key)])


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path == "/":
                return self._send(200, page_index().encode(), "text/html; charset=utf-8")
            if u.path == "/healthz":
                return self._send(200, b'{"ok": true}', "application/json")
            if u.path == "/static/style.css":
                return self._send(200, (STATIC / "style.css").read_bytes(), "text/css")
            if len(parts) == 2 and parts[0] == "split" and parts[1] in SPLITS:
                return self._send(200, page_split(parts[1]).encode(), "text/html; charset=utf-8")
            if len(parts) == 3 and parts[0] == "shot" and parts[1] in SPLITS and parts[2] in _keys(parts[1]):
                return self._send(200, page_shot(parts[1], parts[2]).encode(), "text/html; charset=utf-8")
            if u.path in ("/api/series", "/api/frame"):
                split, key = q.get("split", ""), q.get("key", "")
                if split not in SPLITS or key not in _keys(split):
                    return self._send(404, b'{"error": "unknown split/key"}', "application/json")
                data = api_series(split, key) if u.path == "/api/series" else api_frame(split, key, int(q.get("t", 0)))
                return self._send(200, json.dumps(data).encode(), "application/json")
            return self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001
            return self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("[fec_site] " + (fmt % args) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[fec_site] http://{a.host}:{a.port}/  (predictions from {output_dir()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
