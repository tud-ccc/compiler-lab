"""Train the example models once and keep the weights (doit build file).

pixi run examples                       # train what is missing, render with algeco
pixi run examples list                  # all tasks
pixi run examples train:deep            # one model
pixi run examples forget train:deep     # retrain it next time

doit runs in this directory. Weights go to `weights/`, images to `out/`.
A model is retrained only when its configuration below changes, not when
image_mlp.py changes: delete the .npz (or `forget`) to retrain after editing
the training code.
"""

from doit.tools import config_changed

DOIT_CONFIG = {"default_tasks": ["train", "render"], "verbosity": 2}

# Arguments of `image_mlp.py train`; unset ones take its defaults.
IMAGE_MLPS = {
    "raw": dict(features=0, depth=4, width=256),  # no Fourier features: blurry
    "small": dict(depth=2, width=64),
    "default": dict(depth=4, width=256),
    "deep": dict(depth=8, width=256, steps=2000),
}


def weights(name):
    return f"weights/image_mlp-{name}.npz"


def task_train():
    """Train the image MLPs with PyTorch."""
    for name, config in IMAGE_MLPS.items():
        args = " ".join(f"--{key} {value}" for key, value in config.items())
        yield {
            "name": name,
            "actions": [f"python image_mlp.py train {weights(name)} {args}"],
            "targets": [weights(name)],
            "uptodate": [config_changed(config)],
            "clean": True,
        }


def task_render():
    """Render each trained model with algeco; check against NumPy and time it."""
    for name in IMAGE_MLPS:
        png = f"out/image_mlp-{name}.png"
        yield {
            "name": name,
            "actions": [
                "mkdir -p out",
                # One dump directory per model: algeco names the subdirectory
                # after the function and shapes, which models can share.
                f"python image_mlp.py run {weights(name)} --png {png}"
                f" --dump out/dump/{name}",
            ],
            "file_dep": [weights(name), "image_mlp.py"],
            "targets": [png],
            "clean": True,
        }
