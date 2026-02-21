from numpy_utils import *
from item.item import Item
from ray import Ray
import numpy as np

class Plane(Item):
    def __init__(self, x, y, z, nx, ny, nz):
        super().__init__(x, y, z)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        if norm == 0:
            self.normal = np.array([0.0, 1.0, 0.0])
        else:
            self.normal = np.array([nx / norm, ny / norm, nz / norm])

    def reflect(self, incoming_ray: Ray) -> Ray | None:
        """
        Calculates the reflection of a ray off the plane.
        
        Expects a vectorized ray bundle (dir shape (H, W, 3)).
        Returns None if no ray in the bundle intersects the plane.
        """
        # Vector from ray origin (O) to a point on the plane (C = self.center)
        # Formula: t = ((C - O) . N) / (D . N)
        co = self.center - incoming_ray.center  # (3,) or (H, W, 3)
        d = incoming_ray.dir  # (H, W, 3)

        # D . N
        dn = np.sum(d * self.normal, axis=-1)  # (H, W)

        # (C - O) . N
        con = np.sum(co * self.normal, axis=-1)  # scalar or (H, W)

        # Calculate t, handling division by zero for rays parallel to the plane
        t = np.divide(con, dn, out=np.inf * np.ones_like(dn), where=np.abs(dn) > 1e-6)

        eps = 1e-4
        hit = (t > eps)

        if not np.any(hit):
            return None

        # Replace invalid t with 0.0 to prevent inf/nan in downstream calculations
        t_safe = np.where(hit, t, 0.0)
        t3d = t_safe[..., np.newaxis]  # (H, W, 1)
        intersection_point = incoming_ray.center + t3d * d  # (H, W, 3)

        # Determine surface normal based on which side the ray hits
        dn_3d = dn[..., np.newaxis]
        # self.normal is (3,), broadcasting cleanly since dn_3d is (H, W, 1)
        surface_normal = np.where(dn_3d < 0, self.normal, -self.normal)

        # Calculate reflection direction
        dot_dn = np.sum(d * surface_normal, axis=-1, keepdims=True)  # (H, W, 1)
        out_dir = d - 2.0 * dot_dn * surface_normal  # (H, W, 3)

        # Normalize reflection direction
        od_norms = np.linalg.norm(out_dir, axis=-1, keepdims=True)
        od_norms = np.where(od_norms == 0, 1.0, od_norms)
        out_dir = out_dir / od_norms

        # Non-hit pixels get inf center and zero direction
        intersection_point[~hit] = np.inf
        out_dir[~hit] = 0.0

        return Ray(intersection_point, out_dir)
