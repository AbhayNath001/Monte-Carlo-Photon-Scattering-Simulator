# Monte Carlo Photon Scattering Simulator

Python implementation of the photon-transport model described in the manuscript
*"Monte Carlo-Simulated Photon Scattering Patterns in Biological Tissues for AI Imaging"*.
It generates 256 × 256 grayscale photon-density images of light propagating through a
homogeneous scattering and absorbing medium, together with a metadata CSV that links each
image to the optical parameters used to produce it.

## Physical model

| Component | Implementation |
|---|---|
| Photon transport | Position, direction and weight tracked for every photon |
| Henyey-Greenstein phase function | Inverse-CDF sampling of the scattering angle, controlled by the anisotropy factor `g` (1 = forward, 0 = isotropic, -1 = backward) |
| Beer-Lambert absorption | Photon weight multiplied by `exp(-mu_a * step)` along each step |
| Random walk | Step lengths drawn from an exponential distribution, `s = -ln(xi) / mu_s` |

The simulation is vectorised with NumPy, so all photons are advanced together. Photons
with very low weight are terminated by Russian roulette, which keeps the result unbiased.

## Installation

Requires Python 3.9+.

```bash
pip install numpy pillow
```

## Usage

### Single image

```bash
python photon_mc.py single --wavelength 780 --mu_s 0.5 --g 0.99 --mu_a 0.5 --out sample.png
```

| Argument | Default | Meaning |
|---|---|---|
| `--wavelength` | 650 | Wavelength in nm |
| `--mu_s` | 1.5 | Scattering coefficient (cm⁻¹) |
| `--g` | 0.9 | Anisotropy factor |
| `--mu_a` | 0.5 | Absorption coefficient (cm⁻¹) |
| `--photons` | 5000 | Photons per simulation |
| `--out` | `sample.png` | Output file |
| `--seed` | 0 | Random seed |

### Full dataset

```bash
python photon_mc.py dataset --out dataset --n 2000 --photons 5000 --seed 0
```

This writes:

```
dataset/
├── IMAGES/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── metadata.csv
```

`metadata.csv` has the same columns as Table 2 of the paper:
`Image_Path, Wavelength_nm, Scattering_Coefficient, Anisotropy, Absorption_Coefficient`.

### From Python

```python
import numpy as np
from photon_mc import SimConfig, simulate_photon_density, density_to_uint8

cfg = SimConfig(n_photons=5000)
density = simulate_photon_density(
    wavelength_nm=650, mu_s=2.5, g=0.958, mu_a=0.5,
    cfg=cfg, rng=np.random.default_rng(0),
)
image = density_to_uint8(density)   # (256, 256) uint8 array
```

## Parameter ranges (from the paper)

| Parameter | Range |
|---|---|
| Wavelength | 450, 532, 650, 780 nm |
| Scattering coefficient µs | 0.5 – 2.5 cm⁻¹ |
| Anisotropy g | 0.7 – 0.99 |
| Absorption coefficient µa | 0.1 – 1.5 cm⁻¹ |


## Expected behaviour

Quick sanity checks that the output follows the trends described in the paper:

- Higher `g` gives a narrower, more forward-directed beam.
- Higher `mu_s` gives more lateral spread.
- Shorter wavelengths give more diffuse patterns with shallower penetration.
- Higher `mu_a` gives faster intensity decay with depth.

Images can look sparse at 5,000 photons with low `mu_s`, because there are few interaction
sites. Increase `--photons` for denser maps.

## Configuration reference (`SimConfig`)

| Field | Default | Meaning |
|---|---|---|
| `n_photons` | 5000 | Photons per image |
| `img_size` | 256 | Output resolution |
| `depth_cm` | 5.0 | Slab thickness |
| `half_width_cm` | 2.5 | Lateral half-width covered by the image |
| `max_steps` | 2000 | Cap on scattering events per photon |
| `weight_threshold` | 1e-4 | Weight below which Russian roulette applies |
| `roulette_m` | 10 | Roulette survival probability is 1/m |
| `lambda_ref_nm` | 650 | Reference wavelength for µs scaling |
| `lambda_exponent` | 1.0 | Exponent `b` in the wavelength scaling |

---

## 📬 Contact

<table>
<tr>
<td width="65%">

### 👤 Research Profile

**Name:** Abhay Nath |
**Affiliation:** AI-ML Research Scientist

**📧 Email:** [dr.abhaynath001@gmail.com](mailto:dr.abhaynath001@gmail.com)
**🔗 GitHub:** https://github.com/AbhayNath001
**🎓 LinkedIn:** https://www.linkedin.com/in/abhaynath001/ 

**🧪 Research Areas:**

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Generative AI
* MedTech

</td>

<td width="35%" align="center">

<img src="https://raw.githubusercontent.com/AbhayNath001/AbhayNath001/main/Passport%20Size%20Image%20(1).jpg" width="160" style="border:2px solid #e0e0e0; border-radius:12px; padding:6px;" />

<br><br>

<i>Abhay Nath | Kolkata, West Bengal, India</i>

</td>
</tr>
</table>

---
