"""Validation shared by diffusion inference CLIs."""


def validated_sampling_steps(config, requested_steps):
    trained_steps = int(config["diffusion_model"]["T"])
    sampling_steps = trained_steps if requested_steps is None else int(requested_steps)
    if sampling_steps != trained_steps:
        raise ValueError(
            f"This checkpoint was trained with T={trained_steps}. Current implementation "
            "does not provide verified timestep respacing or DDIM sampling. "
            f"Please use --sampling-steps {trained_steps}."
        )
    return trained_steps
