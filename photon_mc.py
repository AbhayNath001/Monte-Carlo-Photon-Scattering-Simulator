"""
Monte Carlo photon transport in a homogeneous scattering/absorbing medium.

Implements the four components described in the manuscript
"Monte Carlo-Simulated Photon Scattering Patterns in Biological Tissues for AI Imaging":

  1. Photon transport model   -> position / direction / weight tracked per photon
  2. Henyey-Greenstein phase  -> sampling of the scattering angle, anisotropy g
  3. Beer-Lambert absorption  -> weight *= exp(-mu_a * step)
  4. Random walk              -> exponential step lengths, s = -ln(xi) / mu_s

Output: a 256x256 grayscale photon-density image (0-255) plus a metadata CSV row,
matching the dataset layout (Image_Path, Wavelength_nm, Scattering_Coefficient,
Anisotropy, Absorption_Coefficient).

ASSUMPTIONS (the paper does not specify these; change them if you know better):
  * Medium: slab of thickness `depth_cm` along z, wide laterally; photons launched
    at x=y=z=0 travelling in +z (pencil beam).
  * Image plane: x-z cross-section (lateral position vs depth), so penetration depth
    and lateral spread are both visible.
  * Step length uses mu_s only, as in the paper's random-walk description
    (absorption is applied separately via Beer-Lambert weight decay).
  * Wavelength dependence: mu_s(lambda) = mu_s * (lambda / 650 nm) ** (-b).
    The paper reports "shorter wavelength -> more scattering" but gives no formula.
    b = 1.0 is a typical tissue value; set b = 0 to disable.
  * Density = sum of photon weight deposited at every interaction site.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    n_photons: int = 5000          # photons per image (paper: 5000)
    img_size: int = 256            # output resolution (paper: 256 x 256)
    depth_cm: float = 5.0          # slab thickness along z
    half_width_cm: float = 2.5     # lateral half-extent covered by the image
    max_steps: int = 2000          # safety cap on scattering events per photon
    weight_threshold: float = 1e-4 # Russian-roulette threshold
    roulette_m: int = 10           # roulette survival 1/m
    lambda_ref_nm: float = 650.0   # reference wavelength for mu_s scaling
    lambda_exponent: float = 1.0   # b in (lambda/lambda_ref)^-b ; 0 disables


# --------------------------------------------------------------------------- #
# Henyey-Greenstein phase function
# --------------------------------------------------------------------------- #
def sample_hg_cos_theta(g: float, xi: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling of cos(theta) from the Henyey-Greenstein function.

    g =  1 -> fully forward, g = 0 -> isotropic, g = -1 -> fully backward.
    """
    if abs(g) < 1e-6:
        return 2.0 * xi - 1.0
    tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    cos_t = (1.0 + g * g - tmp * tmp) / (2.0 * g)
    return np.clip(cos_t, -1.0, 1.0)


def rotate_directions(d: np.ndarray, cos_t: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate unit vectors d (N,3) by polar angle theta and azimuth phi."""
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    # General case (direction not along +/- z)
    denom = np.sqrt(np.maximum(1e-12, 1.0 - uz ** 2))
    nx = sin_t * (ux * uz * cos_p - uy * sin_p) / denom + ux * cos_t
    ny = sin_t * (uy * uz * cos_p + ux * sin_p) / denom + uy * cos_t
    nz = -sin_t * cos_p * denom + uz * cos_t

    # Degenerate case: travelling (almost) exactly along +/- z
    near_z = np.abs(uz) > 0.99999
    if np.any(near_z):
        sgn = np.sign(uz[near_z])
        nx[near_z] = sin_t[near_z] * cos_p[near_z]
        ny[near_z] = sin_t[near_z] * sin_p[near_z]
        nz[near_z] = sgn * cos_t[near_z]

    out = np.stack([nx, ny, nz], axis=1)
    out /= np.linalg.norm(out, axis=1, keepdims=True)  # guard against drift
    return out


# --------------------------------------------------------------------------- #
# Core Monte Carlo loop (vectorised across photons)
# --------------------------------------------------------------------------- #
def simulate_photon_density(
    wavelength_nm: float,
    mu_s: float,
    g: float,
    mu_a: float,
    cfg: SimConfig = SimConfig(),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the simulation and return a float (img_size, img_size) density map.

    Rows = depth z (top = surface), columns = lateral x (centre = beam axis).
    """
    rng = rng or np.random.default_rng()
    n = cfg.n_photons

    # Wavelength-dependent scattering (assumption, see module docstring)
    mu_s_eff = mu_s * (wavelength_nm / cfg.lambda_ref_nm) ** (-cfg.lambda_exponent)

    pos = np.zeros((n, 3))
    dirs = np.tile([0.0, 0.0, 1.0], (n, 1))
    w = np.ones(n)
    alive = np.ones(n, dtype=bool)

    # Accumulators for the x-z density histogram
    xs, zs, ws = [], [], []

    for _ in range(cfg.max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        # 4. Random walk: exponential free path with the scattering coefficient
        xi = rng.random(idx.size)
        step = -np.log(1.0 - xi) / mu_s_eff
        pos[idx] += dirs[idx] * step[:, None]

        # Photons leaving the slab (top surface z<0 or bottom z>depth) or the
        # lateral window are terminated.
        out = (
            (pos[idx, 2] < 0.0)
            | (pos[idx, 2] > cfg.depth_cm)
            | (np.abs(pos[idx, 0]) > cfg.half_width_cm)
            | (np.abs(pos[idx, 1]) > cfg.half_width_cm * 4)  # y unbounded-ish (projected out)
        )
        alive[idx[out]] = False
        idx = idx[~out]
        step = step[~out]
        if idx.size == 0:
            break

        # 3. Beer-Lambert absorption along the step
        w_before = w[idx]
        w[idx] = w_before * np.exp(-mu_a * step)
        deposited = w_before - w[idx]

        # Record interaction site: weight present at the site -> photon "density"
        xs.append(pos[idx, 0].copy())
        zs.append(pos[idx, 2].copy())
        ws.append(w[idx].copy())

        # 2. Henyey-Greenstein scattering
        cos_t = sample_hg_cos_theta(g, rng.random(idx.size))
        phi = 2.0 * np.pi * rng.random(idx.size)
        dirs[idx] = rotate_directions(dirs[idx], cos_t, phi)

        # Russian roulette for low-weight photons (unbiased termination)
        low = idx[w[idx] < cfg.weight_threshold]
        if low.size:
            survive = rng.random(low.size) < 1.0 / cfg.roulette_m
            w[low[survive]] *= cfg.roulette_m
            alive[low[~survive]] = False

    if not xs:
        return np.zeros((cfg.img_size, cfg.img_size))

    x = np.concatenate(xs)
    z = np.concatenate(zs)
    wt = np.concatenate(ws)

    hist, _, _ = np.histogram2d(
        z, x,
        bins=cfg.img_size,
        range=[[0.0, cfg.depth_cm], [-cfg.half_width_cm, cfg.half_width_cm]],
        weights=wt,
    )
    return hist


def density_to_uint8(density: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """Normalise a density map to 0-255 (log compression keeps the faint tails visible)."""
    d = np.log1p(density) if log_scale else density
    rng_ = d.max() - d.min()
    if rng_ <= 0:
        return np.zeros_like(d, dtype=np.uint8)
    return np.round(255.0 * (d - d.min()) / rng_).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Dataset generation (metadata layout from Tables 1 and 2)
# --------------------------------------------------------------------------- #
WAVELENGTHS_NM = [450, 532, 650, 780]
# Table 2's values (e.g. 0.7222, 0.7644, 0.2556) correspond to 10-point linspace grids,
# which is inconsistent with the grid sizes stated in Table 1 (25 / 30 / 15 values).
MU_S_GRID = np.linspace(0.5, 2.5, 10)
G_GRID = np.linspace(0.70, 0.99, 10)
MU_A_GRID = np.linspace(0.1, 1.5, 10)


def generate_dataset(out_dir: str, n_images: int = 2000, seed: int = 0,
                     cfg: SimConfig = SimConfig()) -> None:
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "IMAGES")
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_Path", "Wavelength_nm", "Scattering_Coefficient",
                         "Anisotropy", "Absorption_Coefficient"])
        for i in range(n_images):
            wl = int(rng.choice(WAVELENGTHS_NM))
            mu_s = float(rng.choice(MU_S_GRID))
            g = float(rng.choice(G_GRID))
            mu_a = float(rng.choice(MU_A_GRID))

            density = simulate_photon_density(wl, mu_s, g, mu_a, cfg, rng)
            img = density_to_uint8(density)
            rel = f"IMAGES/{i}.png"
            Image.fromarray(img, mode="L").save(os.path.join(out_dir, rel))
            writer.writerow([rel, wl, mu_s, g, mu_a])

            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{n_images} images done")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Monte Carlo photon scattering simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single", help="simulate one image")
    s.add_argument("--wavelength", type=float, default=650)
    s.add_argument("--mu_s", type=float, default=1.5)
    s.add_argument("--g", type=float, default=0.9)
    s.add_argument("--mu_a", type=float, default=0.5)
    s.add_argument("--photons", type=int, default=5000)
    s.add_argument("--out", default="sample.png")
    s.add_argument("--seed", type=int, default=0)

    d = sub.add_parser("dataset", help="generate the full dataset + metadata.csv")
    d.add_argument("--out", default="dataset")
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--photons", type=int, default=5000)
    d.add_argument("--seed", type=int, default=0)

    a = p.parse_args()
    cfg = SimConfig(n_photons=a.photons)

    if a.cmd == "single":
        rng = np.random.default_rng(a.seed)
        dens = simulate_photon_density(a.wavelength, a.mu_s, a.g, a.mu_a, cfg, rng)
        Image.fromarray(density_to_uint8(dens), mode="L").save(a.out)
        print(f"saved {a.out}")
    else:
        generate_dataset(a.out, a.n, a.seed, cfg)


if __name__ == "__main__":
    main()
