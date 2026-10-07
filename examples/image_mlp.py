"""Fit an image with an MLP: pixel coordinates (x, y) -> color (r, g, b).

The network memorizes a single image (scikit-learn's sample image china.jpg).
A ReLU MLP fed raw coordinates only learns a blurry, low-frequency version of
it ("spectral bias"). Mapping the coordinates to random Fourier features first
(Tancik et al., 2020) lets it fit fine details, given enough depth and width:

    features = [sin(2π c B), cos(2π c B)]        c: (N, 2), B: (2, F) random
    rgb      = relu(...relu(features @ w1 + b1)...) @ wn + bn

Training runs in PyTorch. Inference runs in two stages: the Fourier features in
NumPy (algeco has no sin/cos), then the MLP compiled by algeco with the trained
weights as constants, in fixed-size chunks of pixels.

Training takes a minute or more, so it is a separate step that saves the
weights. `examples/dodo.py` trains a few configurations once (`pixi run examples`):

python examples/image_mlp.py train weights.npz --depth 6 --width 256
python examples/image_mlp.py run weights.npz                # render, compare, time
python examples/image_mlp.py run weights.npz --scale 1      # at full resolution
python examples/image_mlp.py run weights.npz --emit         # print the IR
python examples/image_mlp.py run weights.npz --dump dump/   # IR of every stage
"""

import json
import time
from pathlib import Path

import cyclopts
import numpy as np
import threadpoolctl
import torch
from PIL import Image
from sklearn.datasets import load_sample_image

import algeco
from algeco.pipeline import to_text

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_image(scale: float = 0.5) -> np.ndarray:
    """china.jpg resized by `scale`, as float32 (H, W, 3) in [0, 1]."""
    image = Image.fromarray(load_sample_image("china.jpg"))
    if scale != 1:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255


def pixel_coords(height: int, width: int) -> np.ndarray:
    """Centers of all pixels, row by row, as (H * W, 2) coordinates (x, y) in [0, 1].

    Both axes are scaled by the longer side, so that a frequency means the
    same in x and y.
    """
    side = max(height, width)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    return np.stack([x.ravel() + 0.5, y.ravel() + 0.5], axis=1) / side


def random_frequencies(n: int, sigma: float, seed: int = 0) -> np.ndarray | None:
    """`n` random 2-d frequencies, as a (2, n) matrix B; `None` for n = 0.

    `sigma` is the standard deviation, in periods per image side. It decides
    the finest detail the network can represent: too low gives a blurry
    image, too high a noisy one.
    """
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, n)) * sigma).astype(np.float32)


def fourier_features(coords: np.ndarray, B: np.ndarray | None) -> np.ndarray:
    """(N, 2) coordinates -> (N, 2F) features [sin(2π c B), cos(2π c B)].

    Without frequencies (`B is None`), the network sees the raw coordinates.
    """
    if B is None:
        return coords
    proj = 2 * np.pi * (coords @ B)
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=1).astype(np.float32)


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]; higher is better."""
    mse = np.mean((np.clip(image, 0, 1) - reference) ** 2)
    return float(-10 * np.log10(mse))


# ---------------------------------------------------------------------------
# Training (PyTorch)
# ---------------------------------------------------------------------------


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_mlp(d_in: int, width: int, depth: int, d_out: int = 3) -> torch.nn.Sequential:
    """`depth` hidden layers of `width` units with ReLU, then a linear output."""
    layers: list[torch.nn.Module] = []
    for i in range(depth):
        layers += [torch.nn.Linear(d_in if i == 0 else width, width), torch.nn.ReLU()]
    layers.append(torch.nn.Linear(width if depth else d_in, d_out))
    return torch.nn.Sequential(*layers)


def train(
    model: torch.nn.Module,
    features: np.ndarray,
    target: np.ndarray,
    steps: int = 2000,
    lr: float = 2e-3,
    batch_size: int | None = 8192,
    device: torch.device | None = None,
    log_every: int = 0,
) -> list[float]:
    """Fit `model` so that `model(features) ≈ target` (MSE), with Adam.

    Each step uses a random batch of `batch_size` pixels (all pixels if None).
    The learning rate decays to zero along a cosine. Returns the loss per step.
    """
    device = device or default_device()
    model.to(device)
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(target.reshape(len(features), -1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for step in range(steps):
        if batch_size is None or batch_size >= len(x):
            xb, yb = x, y
        else:
            idx = torch.randint(len(x), (batch_size,), device=device)
            xb, yb = x[idx], y[idx]
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:5d}  loss {losses[-1]:.5f}")
    return losses


def export_weights(model: torch.nn.Sequential) -> list[tuple[np.ndarray, np.ndarray]]:
    """The (w, b) of each linear layer, with w as (in, out) for `x @ w + b`."""
    return [
        (
            m.weight.detach().cpu().numpy().T.copy(),
            m.bias.detach().cpu().numpy().copy(),
        )
        for m in model
        if isinstance(m, torch.nn.Linear)
    ]


# ---------------------------------------------------------------------------
# Inference: NumPy reference and algeco
# ---------------------------------------------------------------------------


def mlp_numpy(layers):
    def infer(x):
        for w, b in layers[:-1]:
            x = np.maximum(x @ w + b, 0)
        w, b = layers[-1]
        return x @ w + b

    return infer


def mlp_algeco(layers, passes: str = "", dump=None):
    """The MLP compiled by algeco, with the weights as constants.

    The loop runs while algeco traces the function, so the compiled code is a
    straight sequence of layers. With `dump`, algeco writes the IR of every
    compilation stage to that directory.
    """

    @algeco.jit(passes=passes, dump=dump)
    def infer(x):
        for w, b in layers[:-1]:
            x = algeco.relu(x @ w + b)
        w, b = layers[-1]
        return x @ w + b

    return infer


def render(infer, features: np.ndarray, shape, chunk: int = 4096) -> np.ndarray:
    """Run `infer` on all pixels in chunks of `chunk` rows; returns the (H, W, 3) image.

    algeco compiles for fixed shapes, so the last chunk is zero-padded to the
    same size, and the code is compiled only once.
    """
    n = len(features)
    out = np.empty((n, 3), dtype=np.float32)
    batch = np.zeros((chunk, features.shape[1]), dtype=np.float32)
    for start in range(0, n, chunk):
        rows = min(chunk, n - start)
        batch[:rows] = features[start : start + rows]
        out[start : start + rows] = infer(batch)[:rows]
    return np.clip(out.reshape(*shape, 3), 0, 1)


def best_time(f, *args, repeat=5):
    f(*args)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        f(*args)
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# Trained models on disk
# ---------------------------------------------------------------------------


def save_model(path: Path, B, layers, **config) -> None:
    """Frequencies, weights and training configuration as one .npz file."""
    arrays = {}
    for i, (w, b) in enumerate(layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
    if B is not None:
        arrays["B"] = B
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, config=np.array(json.dumps(config)), **arrays)


def load_model(path: Path):
    """The (B, layers, config) saved by `save_model`."""
    with np.load(path) as data:
        n = sum(key.startswith("w") for key in data)
        layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(n)]
        B = data["B"] if "B" in data else None
        return B, layers, json.loads(str(data["config"]))


# ---------------------------------------------------------------------------


app = cyclopts.App(help=__doc__.split("\n\n")[0])


@app.command(name="train")
def main_train(
    out: Path,
    *,
    scale: float = 0.5,
    features: int = 128,
    sigma: float = 10.0,
    depth: int = 4,
    width: int = 256,
    steps: int = 1000,
    seed: int = 0,
):
    """Train with PyTorch and save the weights.

    Parameters
    ----------
    out
        Output .npz file.
    scale
        Image scale.
    features
        Number of Fourier features (0: raw coordinates).
    sigma
        Frequency scale, in periods per image side.
    depth
        Hidden layers.
    width
        Units per hidden layer.
    steps
        Training steps.
    seed
        Seed for the frequencies and the initial weights.
    """
    torch.manual_seed(seed)
    image = load_image(scale)
    h, w, _ = image.shape
    B = random_frequencies(features, sigma, seed)
    x = fourier_features(pixel_coords(h, w), B)
    model = make_mlp(x.shape[1], width, depth)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"image {w}x{h}, {x.shape[1]} inputs, "
        f"{depth}x{width} hidden, {n_params:,} parameters"
    )
    t0 = time.perf_counter()
    train(model, x, image, steps=steps, log_every=max(steps // 10, 1))
    print(f"trained in {time.perf_counter() - t0:.1f} s")

    layers = export_weights(model)
    result = render(mlp_numpy(layers), x, (h, w))
    print(f"PSNR {psnr(result, image):.2f} dB")
    config = dict(scale=scale, features=features, sigma=sigma, depth=depth)
    config.update(width=width, steps=steps, seed=seed)
    save_model(out, B, layers, **config)
    print(f"saved to {out}")


@app.command(name="run")
def main_run(
    model: Path,
    *,
    scale: float | None = None,
    chunk: int = 4096,
    png: Path | None = None,
    emit: bool = False,
    passes: str = "",
    dump: Path | None = None,
):
    """Render with algeco, compare with NumPy and time both.

    Parameters
    ----------
    model
        .npz file written by `train`.
    scale
        Image scale (default: as trained).
    chunk
        Pixels per algeco call.
    png
        Write the original and algeco's image side by side to this file.
    emit
        Print the IR emitted by the frontend and exit.
    passes
        algeco pass pipeline.
    dump
        Write the IR of every compilation stage to this directory.
    """
    B, layers, config = load_model(model)
    image = load_image(scale or config["scale"])
    h, w, _ = image.shape
    features = fourier_features(pixel_coords(h, w), B)
    infer = mlp_algeco(layers, passes, dump)
    if emit:
        print(to_text(infer.emit(features[:chunk])))
        return

    reference = render(mlp_numpy(layers), features, (h, w), chunk)
    t0 = time.perf_counter()
    result = render(infer, features, (h, w), chunk)
    print(f"algeco compiled and rendered in {time.perf_counter() - t0:.1f} s")
    np.testing.assert_allclose(result, reference, rtol=1e-3, atol=1e-3)
    print(f"algeco output matches NumPy, {w}x{h}, PSNR {psnr(result, image):.2f} dB")

    # One chunk is enough for the timing, the unoptimized loops are slow.
    # algeco's code is single-threaded, so the NumPy (OpenBLAS) baseline is too.
    pixels = features[:chunk]
    for name, f in [("algeco", infer), ("NumPy", mlp_numpy(layers))]:
        with threadpoolctl.threadpool_limits(1):
            t = best_time(f, pixels, repeat=3) / len(pixels)
        print(f"{name:>7}: {t * 1e6:8.2f} µs per pixel, {t * h * w:6.2f} s per image")

    if png:
        side_by_side = np.concatenate([image, result], axis=1)
        Image.fromarray((side_by_side * 255).round().astype(np.uint8)).save(png)
        print(f"original | algeco written to {png}")


if __name__ == "__main__":
    app()
