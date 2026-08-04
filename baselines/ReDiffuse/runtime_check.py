from pathlib import Path


def require_official_b_conv():
    required_source = Path(__file__).resolve().parent / "Condition_Noise_Predictor" / "B_Conv.py"
    if not required_source.is_file():
        raise RuntimeError(
            "ReDiffuse cannot run because the official source file is missing: "
            f"{required_source}. Do not reconstruct this file by guessing. "
            "Obtain the verified source from the authors."
        )
    return required_source
