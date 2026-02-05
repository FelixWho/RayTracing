import numpy as np
from math import sin, cos

class Vec3(np.ndarray):
    def __new__(cls, x: float, y: float, z: float) -> 'Vec3':
        return np.asarray([x, y, z], dtype=float).view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return np.asarray(out_arr).view(Vec3)

    def magnitude(self) -> float:
        return np.linalg.norm(self)
    
    def normalize(self) -> "Vec3":
        mag = self.magnitude()
        if mag != 0:
            self /= mag
        return self
    
    def rotate(self, x_rad: float, y_rad: float, z_rad: float) -> "Vec3":
        # This implementation uses the ZYX extrinsic rotation convention (or XYZ intrinsic).
        # This means rotations are applied in the order: Z-axis, then Y-axis, then X-axis
        # relative to the fixed (world) coordinate system.
        # The combined rotation matrix R is R_x(x_rad) @ R_y(y_rad) @ R_z(z_rad).

        # Rotation matrix around X-axis
        Rx = np.array([
            [1, 0, 0],
            [0, cos(x_rad), -sin(x_rad)],
            [0, sin(x_rad), cos(x_rad)]
        ])

        # Rotation matrix around Y-axis
        Ry = np.array([
            [cos(y_rad), 0, sin(y_rad)],
            [0, 1, 0],
            [-sin(y_rad), 0, cos(y_rad)]
        ])

        # Rotation matrix around Z-axis
        Rz = np.array([
            [cos(z_rad), -sin(z_rad), 0],
            [sin(z_rad), cos(z_rad), 0],
            [0, 0, 1]
        ])

        # Combine rotations: R = Rx @ Ry @ Rz
        rotation_matrix = Rx @ Ry @ Rz
        
        return (rotation_matrix @ self).view(Vec3)

    def rotate_degrees(self, x_deg: float, y_deg: float, z_deg: float) -> "Vec3":
        return self.rotate(np.deg2rad(x_deg), np.deg2rad(y_deg), np.deg2rad(z_deg))