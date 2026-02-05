from vec3 import Vec3
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

        First, it solves for the intersection point using the quadratic formula.
        If an intersection is found, it calculates the surface normal and the
        reflected ray's direction, returning a new Ray object.
        Returns None if the ray does not intersect the sphere.
        """
        # Vector from ray origin (O) to sphere center (C)
        oc = incoming_ray.center - self.center

        a = np.dot(incoming_ray.dir, incoming_ray.dir)
        b = 2.0 * np.dot(oc, incoming_ray.dir)
        c = np.dot(oc, oc) - self.radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Find the smallest, positive intersection distance 't'
        # A small epsilon (1e-4) is used to avoid self-intersection artifacts.
        ts = [t for t in (t1, t2) if t > 1e-4]
        if not ts:
            return None
        t = min(ts)

        intersection_point = incoming_ray.center + t*incoming_ray.dir
        surface_normal = (intersection_point - self.center).normalize()
        out_dir = incoming_ray.dir - 2 * np.dot(incoming_ray.dir, surface_normal) * surface_normal
        return Ray(intersection_point, out_dir)