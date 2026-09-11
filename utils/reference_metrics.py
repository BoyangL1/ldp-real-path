"""Reference metrics matching the Nagoya adaptive experiments.

Fixed normalized-coordinate grid: [-2.6,2.6], 40 cells; density smoothing sigma=2.
OD-SSIM averages start/end marginals; OD-JSD measures their joint distribution.
len_jsd is STEP-length JSD, not total trajectory length.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

EXTENT, GRID, SIGMA = 2.6, 40, 2.0
BINS = np.linspace(-EXTENT, EXTENT, GRID + 1)
NOISES = ["0.00", "0.10", "0.20", "0.30", "0.40", "0.50",
          "0.60", "0.70", "0.80", "0.90", "1.00"]


def hist2d(pts):
    h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=(BINS, BINS))
    return h


def smooth(h, transform):
    if transform == "vst":
        a = 2.0 * np.sqrt(h + 0.375)
    elif transform == "lin":
        a = h.astype(float)
    else:
        a = np.log1p(h)
    return gaussian_filter(a / max(a.sum(), 1e-9), SIGMA)


def ssim_pair(ha, hb, transform):
    a, b = smooth(ha, transform), smooth(hb, transform)
    dr = max(a.max(), b.max()) - min(a.min(), b.min())
    return float(ssim(a, b, data_range=dr))


def od_ssim(real, gen, transform):
    return 0.5 * (ssim_pair(hist2d(real[:, 0]), hist2d(gen[:, 0]), transform) +
                  ssim_pair(hist2d(real[:, -1]), hist2d(gen[:, -1]), transform))


def _jsd(p, q, eps=1e-12):
    p, q = p / max(p.sum(), eps), q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cell(pts):
    ix = np.clip(np.digitize(pts[:, 0], BINS) - 1, 0, GRID - 1)
    iy = np.clip(np.digitize(pts[:, 1], BINS) - 1, 0, GRID - 1)
    return ix * GRID + iy


def od_jsd(real, gen):
    n = GRID * GRID
    def counts(t):
        k = cell(t[:, 0]).astype(np.int64) * n + cell(t[:, -1]).astype(np.int64)
        return np.bincount(k, minlength=n * n).astype(float)
    return _jsd(counts(real), counts(gen))


def len_jsd(real, gen, nb=60):
    def steps(t):
        return np.linalg.norm(np.diff(t, axis=1), axis=2).ravel()
    r, g = steps(real), steps(gen)
    hr, bins = np.histogram(r, bins=nb, range=(0, np.percentile(r, 99.5)), density=True)
    hg, _ = np.histogram(g, bins=bins, density=True)
    return _jsd(hr, hg)

