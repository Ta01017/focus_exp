#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


BASELINE_METHODS = [
    "DSIFT",
    "FULX2.0_ORIGIN",
    "IFCNN",
    "FusionDiff",
    "ReDiffuse_ORIGIN",
    "SwinFusion",
    "ZMFF",
]

DATASETS = {
    "CommonBlurGeometryVal200": 200,
    "RealMFFAlignedVal110": 110,
}


def parse_args():
    p = argparse.ArgumentParser(description="Build all baseline and ablation manifests for three-route region evaluation")
    p.add_argument("--root", default="/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880")
    p.add_argument("--stamp", default="20260831_095326")
    p.add_argument("--compare-root", default="")
    p.add_argument("--output-root", default="")
    p.add_argument("--cb-ref-cache", default="")
    p.add_argument("--real-ref-cache", default="")
    p.add_argument("--cb-refined-cache", default="")
    p.add_argument("--real-refined-cache", default="")
    p.add_argument("--cb-fullgen-cache", default="")
    p.add_argument("--real-fullgen-cache", default="")
    p.add_argument("--real-control-cache", default="")
    p.add_argument("--real-severe-cache", default="")
    p.add_argument("--include-extra-real", action="store_true",
                   help="also require and build +5k Control and +5k Severe")
    return p.parse_args()


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list) or not obj:
        raise RuntimeError(f"cache must be a non-empty JSON list: {path}")
    return obj


def normalize_real(raw):
    raw = str(raw)
    if raw.startswith("RealMFF_"):
        return raw
    return "RealMFF_" + raw.replace("_F", "")


def index_rows(rows, dataset, reference_keys=None):
    out = {}
    for r in rows:
        sid = str(r.get("sample_id", ""))
        if not sid:
            raise RuntimeError(f"row without sample_id; keys={list(r)}")
        if dataset == "RealMFFAlignedVal110":
            sid = normalize_real(sid)
        elif reference_keys is not None:
            sid = normalize_common(sid, reference_keys)
        if sid in out:
            raise RuntimeError(f"duplicate sample_id {sid}")
        out[sid] = r
    return out


def normalize_common(raw, keys):
    raw = str(raw)
    if raw in keys:
        return raw
    candidates = []
    if raw.startswith("val_"):
        candidates.append(raw.split("_", 3)[-1])
    for x in list(candidates):
        parts = x.split("_")
        if len(parts) > 1:
            candidates.append("_".join(parts[:-1]))
    matches = [x for x in candidates if x in keys]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"COMMON ID ERROR raw={raw}; matches={matches}; first cache keys={list(keys)[:5]}")


def normalize_id(raw, dataset, keys):
    return normalize_common(raw, keys) if dataset == "CommonBlurGeometryVal200" else normalize_real(raw)


def find_baseline_manifest(folder: Path):
    for name in ["eval_manifest.csv", "inference_manifest.csv"]:
        p = folder / "manifest" / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no eval_manifest.csv or inference_manifest.csv under {folder / 'manifest'}")


def get_prediction(row):
    for k in ["prediction", "pred", "fused", "output", "result"]:
        value = str(row.get(k, "")).strip()
        if value:
            return Path(value)
    raise RuntimeError(f"no prediction column/value; columns={list(row)}")


def pick(row, keys, label, required=True):
    if row:
        for k in keys:
            value = row.get(k)
            if value is not None and str(value).strip() not in {"", "None", "nan", "null"}:
                return str(value)
    if required:
        raise KeyError(f"missing {label}; tried keys={keys}; available={list(row) if row else []}")
    return ""


def validate_reference_row(row, dataset, sid):
    required = {
        "source_a": ["source_a", "a", "image_a"],
        "source_b": ["source_b", "b", "image_b"],
        "gt": ["gt", "target", "image"],
        "m_a": ["m_a", "route_a"],
        "m_b": ["m_b", "route_b"],
        "m_g": ["m_g", "route_g"],
    }
    out = {}
    for name, keys in required.items():
        out[name] = pick(row, keys, f"{dataset}/{sid}/{name}")
        if not Path(out[name]).is_file():
            raise FileNotFoundError(f"{dataset}/{sid}: {name} file missing: {out[name]}")
    return out


def preflight_route_maps(refs, dataset, tolerance=0.05):
    ratio_sum = np.zeros(3, dtype=np.float64)
    worst = (-1.0, "")
    for index, (sid, row) in enumerate(refs.items(), 1):
        paths = validate_reference_row(row, dataset, sid)
        with Image.open(paths["source_a"]) as aim:
            size = aim.size
        maps = []
        for key in ["m_a", "m_b", "m_g"]:
            with Image.open(paths[key]) as im:
                x = im.convert("L")
                if x.size != size:
                    x = x.resize(size, Image.Resampling.BILINEAR)
                maps.append(np.asarray(x, dtype=np.float32) / 255.0)
        stack = np.stack(maps, axis=0)
        if not np.all(np.isfinite(stack)):
            raise RuntimeError(f"{dataset}/{sid}: route map contains NaN or Inf")
        error = float(np.mean(np.abs(stack.sum(axis=0) - 1.0)))
        if error > worst[0]:
            worst = (error, sid)
        if error > tolerance:
            raise RuntimeError(
                f"{dataset}/{sid}: mean|M_A+M_B+M_G-1|={error:.6f} > {tolerance}; "
                "these files do not look like normalized three-route maps"
            )
        label = np.argmax(stack, axis=0)
        ratio_sum += np.array([(label == j).mean() for j in range(3)])
    ratio_mean = ratio_sum / len(refs)
    print(
        f"[ROUTE OK] {dataset} n={len(refs)} "
        f"argmax A/B/G={ratio_mean[0]:.4f}/{ratio_mean[1]:.4f}/{ratio_mean[2]:.4f} "
        f"worst_sum_error={worst[0]:.6f} sample={worst[1]}"
    )


def crop_real_prediction(src: Path, dst: Path):
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im0:
        im = im0.convert("RGB")
        if im.size == (625, 433):
            im = im.crop((0, 0, 624, 432))
        elif im.size != (624, 432):
            raise RuntimeError(f"bad RealMFF prediction size {im.size}: {src}")
        im.save(dst)
    return str(dst.resolve())


def canonical_prediction(src, dataset, out_root, method, sid):
    p = Path(src)
    if not p.is_file():
        raise FileNotFoundError(f"prediction missing: {p}")
    if dataset != "RealMFFAlignedVal110":
        return str(p.resolve())
    suffix = p.suffix if p.suffix else ".png"
    dst = out_root / "aligned_predictions" / dataset / method / f"{sid}{suffix}"
    return crop_real_prediction(p, dst)


def write_manifest(out_root, dataset, method_dir, rows):
    expected = DATASETS[dataset]
    if len(rows) != expected:
        raise RuntimeError(f"{dataset}/{method_dir}: expected {expected} rows, got {len(rows)}")
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{dataset}/{method_dir}: duplicate sample IDs")
    path = out_root / "manifests" / dataset / method_dir / "region_manifest_route_v3.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "sample_id", "source_a", "source_b", "gt", "prediction", "m_a", "m_b", "m_g", "fused_mode"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[WRITE] {path} rows={len(rows)}")


def locate_refined_cache(run_dir: Path, explicit: str, ref_rows):
    if explicit:
        return Path(explicit)
    if all(any(str(r.get(k, "")).strip() for k in ["refined_prediction", "refined", "final_prediction"]) for r in ref_rows):
        return None
    found = sorted(run_dir.rglob("refined_cache.json"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(
            f"cannot find refined_cache.json under {run_dir}; pass --cb-refined-cache or --real-refined-cache"
        )
    raise RuntimeError(f"multiple refined_cache.json files under {run_dir}; pass explicit path:\n" + "\n".join(map(str, found)))


def locate_extra_cache(v7_runs: Path, token: str, explicit: str):
    if explicit:
        return Path(explicit)
    candidates = []
    for p in v7_runs.glob("*"):
        if p.is_dir() and token.lower() in p.name.lower():
            candidates.extend(p.rglob("*.json"))
    usable = []
    for p in sorted(set(candidates)):
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rows, list) and len(rows) == 110 and rows and "sample_id" in rows[0] and any(k in rows[0] for k in ["base_prediction", "prediction", "pred"]):
                usable.append(p)
        except Exception:
            pass
    if len(usable) == 1:
        return usable[0]
    if not usable:
        raise FileNotFoundError(f"cannot auto-detect RealMFF +5k {token} cache; pass --real-{token.lower()}-cache")
    raise RuntimeError(f"multiple +5k {token} cache candidates; pass explicit path:\n" + "\n".join(map(str, usable)))


def make_row(dataset, method, sid, ref, prediction="", fused_mode=""):
    base = validate_reference_row(ref, dataset, sid)
    return {
        "dataset": dataset,
        "method": method,
        "sample_id": sid,
        **base,
        "prediction": prediction,
        "fused_mode": fused_mode,
    }


def build_baselines(compare_root, out_root, dataset, refs):
    keys = set(refs)
    for method in BASELINE_METHODS:
        mf = find_baseline_manifest(compare_root / dataset / method)
        built = []
        with mf.open("r", encoding="utf-8-sig", newline="") as f:
            for item in csv.DictReader(f):
                if "sample_id" not in item:
                    raise KeyError(f"sample_id missing in {mf}")
                sid = normalize_id(item["sample_id"], dataset, keys)
                if sid not in refs:
                    raise RuntimeError(f"reference cache missing {item['sample_id']} -> {sid}")
                pred = canonical_prediction(get_prediction(item), dataset, out_root, method, sid)
                built.append(make_row(dataset, method, sid, refs[sid], pred))
        write_manifest(out_root, dataset, method, built)


def build_ablations(out_root, dataset, refs, refined, fullgen, extras):
    specs = [
        ("AvgBlend", "AvgBlend", None, [], "avgblend"),
        ("G_Diagnostic", "G Diagnostic", refs, ["g_prediction"], ""),
        ("wo_Generation", "w/o Generation", None, [], "no_generation"),
        ("wo_Refiner", "w/o Refiner", refs, ["base_prediction"], ""),
        ("Ours", "Ours", refined, ["refined_prediction", "final_prediction", "prediction", "pred", "output"], ""),
        ("FullGen", "FullGen", fullgen, ["g_prediction", "prediction", "pred", "output", "base_prediction"], ""),
    ]
    for directory, label, source, pred_keys, mode in specs:
        built = []
        for sid, ref in refs.items():
            pred = ""
            if source is not None:
                if sid not in source:
                    raise RuntimeError(f"{dataset}/{label}: prediction cache missing sample {sid}")
                pred = canonical_prediction(pick(source[sid], pred_keys, f"{dataset}/{label}/{sid} prediction"), dataset, out_root, directory, sid)
            built.append(make_row(dataset, label, sid, ref, pred, mode))
        write_manifest(out_root, dataset, directory, built)

    for directory, label, source in extras:
        built = []
        for sid, ref in refs.items():
            if sid not in source:
                raise RuntimeError(f"{dataset}/{label}: prediction cache missing sample {sid}")
            pred_raw = pick(source[sid], ["base_prediction", "refined_prediction", "prediction", "pred", "output"], f"{label}/{sid} prediction")
            pred = canonical_prediction(pred_raw, dataset, out_root, directory, sid)
            built.append(make_row(dataset, label, sid, ref, pred))
        write_manifest(out_root, dataset, directory, built)


def main():
    args = parse_args()
    root = Path(args.root)
    v7 = root / "focus/pixrestore_mfif_paper_suite_v7_20260831"
    runs = v7 / "runs"
    compare_root = Path(args.compare_root) if args.compare_root else root / "focus/models/COMPARE_RESULTS_TWO_DATASETS_20260827"
    out_root = Path(args.output_root) if args.output_root else root / "focus/models/COMPARE_RESULTS_REGION_V3"

    cb_run = runs / f"paper_{args.stamp}_cb_hybrid40k_refiner"
    real_run = runs / f"paper_{args.stamp}_realmff_hybrid20k_refiner"
    cb_ref_path = Path(args.cb_ref_cache) if args.cb_ref_cache else cb_run / "refiner_val_cache/refiner_cache.json"
    real_ref_path = Path(args.real_ref_cache) if args.real_ref_cache else real_run / "refiner_val_cache/refiner_cache.json"
    cb_fullgen_path = Path(args.cb_fullgen_cache) if args.cb_fullgen_cache else runs / f"paper_{args.stamp}_cb_fullgen40k/val_fullgen/refiner_cache.json"
    real_fullgen_path = Path(args.real_fullgen_cache) if args.real_fullgen_cache else runs / f"paper_{args.stamp}_realmff_fullgen20k/val_fullgen/refiner_cache.json"

    print("[PREFLIGHT] loading reference caches")
    cb_ref_rows, real_ref_rows = read_json(cb_ref_path), read_json(real_ref_path)
    cb_refs = index_rows(cb_ref_rows, "CommonBlurGeometryVal200")
    real_refs = index_rows(real_ref_rows, "RealMFFAlignedVal110")
    if len(cb_refs) != 200 or len(real_refs) != 110:
        raise RuntimeError(f"reference cache counts invalid: CommonBlur={len(cb_refs)} RealMFF={len(real_refs)}")
    preflight_route_maps(cb_refs, "CommonBlurGeometryVal200")
    preflight_route_maps(real_refs, "RealMFFAlignedVal110")

    cb_refined_path = locate_refined_cache(cb_run, args.cb_refined_cache, cb_ref_rows)
    real_refined_path = locate_refined_cache(real_run, args.real_refined_cache, real_ref_rows)
    cb_refined = cb_refs if cb_refined_path is None else index_rows(read_json(cb_refined_path), "CommonBlurGeometryVal200", set(cb_refs))
    real_refined = real_refs if real_refined_path is None else index_rows(read_json(real_refined_path), "RealMFFAlignedVal110")
    cb_fullgen = index_rows(read_json(cb_fullgen_path), "CommonBlurGeometryVal200", set(cb_refs))
    real_fullgen = index_rows(read_json(real_fullgen_path), "RealMFFAlignedVal110")

    real_extras = []
    if args.include_extra_real:
        control_path = locate_extra_cache(runs, "control", args.real_control_cache)
        severe_path = locate_extra_cache(runs, "severe", args.real_severe_cache)
        real_extras = [
            ("plus5k_Control", "+5k Control", index_rows(read_json(control_path), "RealMFFAlignedVal110")),
            ("plus5k_Severe", "+5k Severe", index_rows(read_json(severe_path), "RealMFFAlignedVal110")),
        ]

    print("[BUILD] baselines")
    build_baselines(compare_root, out_root, "CommonBlurGeometryVal200", cb_refs)
    build_baselines(compare_root, out_root, "RealMFFAlignedVal110", real_refs)
    print("[BUILD] ablations")
    build_ablations(out_root, "CommonBlurGeometryVal200", cb_refs, cb_refined, cb_fullgen, [])
    build_ablations(out_root, "RealMFFAlignedVal110", real_refs, real_refined, real_fullgen, real_extras)
    print(f"[DONE] all manifests passed preflight: {out_root / 'manifests'}")


if __name__ == "__main__":
    main()
