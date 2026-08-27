from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.transform import iradon, radon, resize


@dataclass(frozen=True)
class SimulationResult:
    corrupted_hu: np.ndarray
    artifact_mask: np.ndarray
    metal_mask: np.ndarray
    metadata: dict[str, float | int | str]


class ArtifactSimulator:
    """Correlated 3D CT corruption simulator.

    The metal path uses slice-wise forward projection and filtered backprojection.
    It is an approximation for augmentation, not a scanner or dose simulator.
    """

    def __init__(
        self,
        seed: int | None = None,
        angles: int = 180,
        physics_probability: float = 0.65,
    ) -> None:
        if angles < 30:
            raise ValueError("angles must be >= 30")
        self.rng = np.random.default_rng(seed)
        self.angles = angles
        self.physics_probability = physics_probability

    def _dental_implants(self, shape: tuple[int, int, int]) -> np.ndarray:
        depth, height, width = shape
        zz, yy, xx = np.ogrid[:depth, :height, :width]
        mask = np.zeros(shape, dtype=bool)
        count = int(self.rng.integers(1, 5))
        for _ in range(count):
            center = (
                self.rng.uniform(0.48, 0.72) * depth,
                self.rng.uniform(0.48, 0.68) * height,
                self.rng.choice((self.rng.uniform(0.28, 0.45), self.rng.uniform(0.55, 0.72)))
                * width,
            )
            radii = (
                self.rng.uniform(1.5, max(2.0, depth * 0.045)),
                self.rng.uniform(2.0, max(2.5, height * 0.025)),
                self.rng.uniform(2.0, max(2.5, width * 0.025)),
            )
            ellipsoid = (
                sum(
                    ((coord - c) / r) ** 2
                    for coord, c, r in zip((zz, yy, xx), center, radii, strict=True)
                )
                <= 1.0
            )
            mask |= ellipsoid
        return mask

    @staticmethod
    def _hu_to_mu(image_hu: np.ndarray) -> np.ndarray:
        return np.maximum((image_hu + 1000.0) * 2.0e-5, 0.0)

    @staticmethod
    def _mu_to_hu(mu: np.ndarray) -> np.ndarray:
        return mu / 2.0e-5 - 1000.0

    def _project_slice(
        self, clean_hu: np.ndarray, metal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        original_shape = clean_hu.shape
        side = max(original_shape)
        if original_shape[0] != original_shape[1]:
            clean_square = resize(clean_hu, (side, side), preserve_range=True, anti_aliasing=True)
            metal_square = (
                resize(metal.astype(np.float32), (side, side), order=0, preserve_range=True) > 0.5
            )
        else:
            clean_square, metal_square = clean_hu, metal
        theta = np.linspace(0.0, 180.0, self.angles, endpoint=False)
        clean_mu = self._hu_to_mu(clean_square)
        metal_mu = metal_square.astype(np.float32) * self.rng.uniform(0.35, 0.65)
        clean_sino = radon(clean_mu, theta=theta, circle=False, preserve_range=True)
        metal_sino = radon(metal_mu, theta=theta, circle=False, preserve_range=True)
        ideal = clean_sino + metal_sino
        metal_trace = metal_sino > 1e-5

        # Approximate beam hardening, photon starvation, scatter, and detector response.
        scale = np.percentile(metal_sino[metal_trace], 95) if metal_trace.any() else 1.0
        relative_metal = np.clip(metal_sino / max(float(scale), 1e-6), 0.0, 4.0)
        measured = ideal + self.rng.uniform(0.03, 0.12) * np.square(relative_metal)
        photons = float(self.rng.uniform(2e4, 1.2e5))
        counts = self.rng.poisson(np.maximum(photons * np.exp(-measured), 1.0))
        measured = -np.log(np.maximum(counts, 1.0) / photons)
        measured += self.rng.normal(0.0, self.rng.uniform(2e-4, 1.5e-3), measured.shape)
        if self.rng.random() < 0.35:
            detector = int(self.rng.integers(0, measured.shape[0]))
            measured[max(0, detector - 1) : detector + 2] *= self.rng.uniform(0.85, 1.15)

        recon_clean = iradon(
            clean_sino, theta=theta, circle=False, filter_name="ramp", output_size=side
        )
        recon_metal = iradon(
            measured, theta=theta, circle=False, filter_name="ramp", output_size=side
        )
        artifact_hu = self._mu_to_hu(recon_metal) - self._mu_to_hu(recon_clean)
        corrupted = clean_square + artifact_hu
        corrupted[metal_square] = self.rng.uniform(3000.0, 6000.0)
        affected = (np.abs(artifact_hu) > self.rng.uniform(35.0, 80.0)) | metal_square
        if original_shape != (side, side):
            corrupted = resize(corrupted, original_shape, preserve_range=True, anti_aliasing=True)
            affected = (
                resize(affected.astype(np.float32), original_shape, order=0, preserve_range=True)
                > 0.5
            )
        return corrupted.astype(np.float32), affected

    def _analytic_streaks(
        self, clean_hu: np.ndarray, metal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        depth, height, width = clean_hu.shape
        yy, xx = np.mgrid[:height, :width]
        corrupted = clean_hu.copy()
        affected = metal.copy()
        centers = np.argwhere(metal)
        if not len(centers):
            return corrupted, affected
        z0, y0, x0 = np.median(centers, axis=0)
        angles = self.rng.uniform(0, np.pi, size=int(self.rng.integers(10, 24)))
        envelope = np.exp(-(((np.arange(depth) - z0) / max(2.0, depth * 0.08)) ** 2))
        field = np.zeros((height, width), dtype=np.float32)
        for angle in angles:
            distance = np.abs((xx - x0) * np.sin(angle) - (yy - y0) * np.cos(angle))
            width_px = self.rng.uniform(0.5, 2.5)
            amplitude = self.rng.uniform(-700.0, 900.0)
            field += amplitude * np.exp(-((distance / width_px) ** 2))
        corrupted += envelope[:, None, None] * field[None]
        corrupted[metal] = self.rng.uniform(3000.0, 6000.0)
        affected |= np.abs(envelope[:, None, None] * field[None]) > 40.0
        return corrupted, affected

    def __call__(self, clean_hu: np.ndarray) -> SimulationResult:
        clean_hu = np.asarray(clean_hu, dtype=np.float32)
        if clean_hu.ndim != 3 or min(clean_hu.shape) < 8:
            raise ValueError("clean_hu must be a 3D array with every dimension >= 8")
        metal = self._dental_implants(clean_hu.shape)
        use_physics = self.rng.random() < self.physics_probability
        if use_physics:
            corrupted = clean_hu.copy()
            affected = metal.copy()
            for z in np.flatnonzero(metal.reshape(clean_hu.shape[0], -1).any(axis=1)):
                corrupted[z], slice_mask = self._project_slice(clean_hu[z], metal[z])
                affected[z] |= slice_mask
            mode = "slice_radon"
        else:
            corrupted, affected = self._analytic_streaks(clean_hu, metal)
            mode = "analytic"

        noise_std = float(self.rng.uniform(3.0, 35.0))
        body_weight = np.clip((clean_hu + 1000.0) / 1500.0, 0.15, 1.0)
        noise = self.rng.normal(0.0, noise_std, clean_hu.shape) / np.sqrt(body_weight)
        noise = ndimage.gaussian_filter(noise, sigma=(0.35, 0.15, 0.15))
        corrupted += noise.astype(np.float32)
        affected = ndimage.binary_dilation(affected, iterations=2)
        corrupted = np.clip(corrupted, -1200.0, 6000.0).astype(np.float32)
        return SimulationResult(
            corrupted_hu=corrupted,
            artifact_mask=affected.astype(np.float32),
            metal_mask=metal.astype(np.float32),
            metadata={
                "mode": mode,
                "angles": self.angles if use_physics else 0,
                "noise_std_hu": noise_std,
                "metal_voxels": int(metal.sum()),
                "artifact_voxels": int(affected.sum()),
            },
        )
