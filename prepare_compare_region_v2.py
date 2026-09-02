#!/usr/bin/env python3
from pathlib import Path
import json
import csv
from PIL import Image


ROOT = Path(
    "/data/vjuicefs_ai_camera_3drg_ql/public_data/11193880"
)

COMPARE = ROOT / (
    "focus/models/"
    "COMPARE_RESULTS_TWO_DATASETS_20260827"
)

V7 = ROOT / (
    "focus/pixrestore_mfif_paper_suite_v7_20260831"
)

STAMP = "20260831_095326"


CACHE = {
    "CommonBlurGeometryVal200":
        V7 /
        f"runs/paper_{STAMP}_cb_hybrid40k_refiner/"
        "refiner_val_cache/refiner_cache.json",

    "RealMFFAlignedVal110":
        V7 /
        f"runs/paper_{STAMP}_realmff_hybrid20k_refiner/"
        "refiner_val_cache/refiner_cache.json",
}


OUT = COMPARE.parent / "COMPARE_RESULTS_REGION_V2"


METHODS = [
    "DSIFT",
    "FULX2.0_ORIGIN",
    "IFCNN",
    "FusionDiff",
    "ReDiffuse_ORIGIN",
    "SwinFusion",
    "ZMFF",
]



def load_cache(dataset):

    rows = json.loads(
        CACHE[dataset].read_text()
    )

    out = {}

    for r in rows:
        out[str(r["sample_id"])] = r

    print(
        dataset,
        "cache",
        len(out)
    )

    return out



def normalize_common(raw, keys):

    # already match
    if raw in keys:
        return raw


    candidates=[]


    # DSIFT:
    # val_000000_000000_xxx_hash

    if raw.startswith("val_"):

        x = raw.split("_",3)[-1]

        candidates.append(x)


    # remove final token candidates

    for x in list(candidates):

        parts=x.split("_")

        if len(parts)>1:

            candidates.append(
                "_".join(parts[:-1])
            )


    for x in candidates:

        if x in keys:
            return x


    raise RuntimeError(
        "\nCOMMON ID ERROR\n"
        f"raw={raw}\n"
        f"first cache={list(keys)[:5]}"
    )



def normalize_real(raw):

    if raw.startswith("RealMFF_"):
        return raw

    raw = raw.replace("_F","")

    return "RealMFF_" + raw



def find_manifest(folder):

    for name in [
        "eval_manifest.csv",
        "inference_manifest.csv",
    ]:

        p = folder / "manifest" / name

        if p.exists():
            return p


    return None



def get_prediction(row):

    for k in [
        "prediction",
        "pred",
        "fused",
        "output",
        "result",
    ]:

        if k in row and row[k]:

            return row[k]


    raise RuntimeError(
        "No prediction column\n"
        f"columns={list(row.keys())}"
    )



def crop_real(src,dst):

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    im=Image.open(src).convert("RGB")


    if im.size==(624,432):

        im.save(dst)

        return str(dst)


    if im.size!=(625,433):

        raise RuntimeError(
            f"bad RealMFF size {im.size}: {src}"
        )


    im=im.crop(
        (0,0,624,432)
    )

    im.save(dst)

    return str(dst)



def main():

    manifest_root = OUT/"manifests"

    pred_root = OUT/"aligned_predictions"


    for dataset in [
        "CommonBlurGeometryVal200",
        "RealMFFAlignedVal110",
    ]:


        cache = load_cache(dataset)

        keys=set(cache.keys())


        for method in METHODS:


            folder = (
                COMPARE /
                dataset /
                method
            )


            mf=find_manifest(folder)


            if mf is None:

                print(
                    "[SKIP]",
                    folder
                )

                continue


            print(
                "\nPROCESS",
                dataset,
                method,
                mf
            )


            rows=[]


            with mf.open(
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as f:


                reader=csv.DictReader(f)


                for item in reader:


                    raw_id=item["sample_id"]


                    if dataset=="CommonBlurGeometryVal200":

                        sid=normalize_common(
                            raw_id,
                            keys
                        )

                    else:

                        sid=normalize_real(
                            raw_id
                        )


                    if sid not in cache:

                        raise RuntimeError(
                            f"cache missing {raw_id}->{sid}"
                        )


                    c=cache[sid]


                    pred=get_prediction(item)


                    if dataset=="RealMFFAlignedVal110":


                        dst=(
                            pred_root /
                            dataset /
                            method /
                            Path(pred).name
                        )


                        pred=crop_real(
                            Path(pred),
                            dst
                        )


                    else:

                        pred=str(
                            Path(pred).resolve()
                        )


                    rows.append({

                        "dataset":
                            dataset,

                        "method":
                            method,

                        "sample_id":
                            sid,

                        "source_a":
                            c["source_a"],

                        "source_b":
                            c["source_b"],

                        "gt":
                            c["gt"],

                        "prediction":
                            pred,

                        "m_a":
                            c["m_a"],

                        "m_b":
                            c["m_b"],

                        "m_g":
                            c["m_g"],

                    })



            outdir=(
                manifest_root /
                dataset /
                method
            )

            outdir.mkdir(
                parents=True,
                exist_ok=True
            )


            outcsv=(
                outdir /
                "region_manifest_v2.csv"
            )


            with outcsv.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as f:


                writer=csv.DictWriter(
                    f,
                    fieldnames=list(rows[0].keys())
                )

                writer.writeheader()

                writer.writerows(rows)


            print(
                "[WRITE]",
                outcsv,
                len(rows)
            )



if __name__=="__main__":
    main()