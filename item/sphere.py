from numpy_utils import *
from item.item import Item
from ray import Ray
import numpy as np

class Sphere(Item):
    def __init__(self, x, y, z, radius):
        super().__init__(x, y, z)
        self.radius = radius

    def reflect(self, incoming_ray: Ray) -> Ray | None:
        """
        Calculates the reflection of a ray off the sphere.
        
        Expects a vectorized ray bundle (dir shape (H, W, 3)).
        Returns None if no ray in the bundle intersects the sphere.
        """
        # Vector from ray origin (O) to sphere center (C)
        oc = incoming_ray.center - self.center  # (3,)

        d = incoming_ray.dir  # (H, W, 3)

        a = np.sum(d * d, axis=-1)           # (H, W)
        b = 2.0 * np.sum(oc * d, axis=-1)   # (H, W)
        c = float(np.dot(oc, oc)) - self.radius**2  # scalar

        discriminant = b**2 - 4 * a * c  # (H, W) or scalar

        hit = discriminant >= 0
        if not np.any(hit):
            return None

        sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        eps = 1e-4
        t1_ok = np.where(hit & (t1 > eps), t1, np.inf)
        t2_ok = np.where(hit & (t2 > eps), t2, np.inf)
        t = np.minimum(t1_ok, t2_ok)  # (H, W)

        valid = t < np.inf
        if not np.any(valid):
            return None

        t3d = t[..., np.newaxis]  # (H, W, 1)
        intersection_point = incoming_ray.center + t3d * d  # (H, W, 3)

        surface_normal = intersection_point - self.center   # (H, W, 3)
        sn_norms = np.linalg.norm(surface_normal, axis=-1, keepdims=True)
        sn_norms = np.where(sn_norms == 0, 1.0, sn_norms)
        surface_normal = surface_normal / sn_norms

        dot_dn = np.sum(d * surface_normal, axis=-1, keepdims=True)  # (H, W, 1)
        out_dir = d - 2.0 * dot_dn * surface_normal  # (H, W, 3)
        od_norms = np.linalg.norm(out_dir, axis=-1, keepdims=True)
        od_norms = np.where(od_norms == 0, 1.0, od_norms)
        out_dir = out_dir / od_norms

        # Non-hit pixels get inf center so they never win the distance comparison
        intersection_point[~valid] = np.inf
        out_dir[~valid] = 0.0

        return Ray(intersection_point, out_dir)