from canvas import Canvas
from ray import Ray
from item.sphere import Sphere
from vec3 import Vec3
import numpy as np

if __name__ == '__main__':
    print("--- Example 1: Rotating a vector that is ON the axis of rotation ---")
    v_on_axis = Vec3(1, 0, 0)
    print(f"Rotating the vector {v_on_axis} by 90 degrees around the X-axis...")
    rotated_v1 = v_on_axis.rotate_degrees(90, 0, 0)
    print(f"Result: {np.round(rotated_v1, decimals=2)}")
    print("This is correct, as a vector on the axis of rotation does not move.\n")

    print("--- Example 2: Rotating a vector that is NOT on the axis of rotation ---")
    v_off_axis = Vec3(0, 1, 0)
    print(f"Rotating the vector {v_off_axis} by 90 degrees around the X-axis...")
    rotated_v2 = v_off_axis.rotate_degrees(90, 0, 0)
    print(f"Result: {np.round(rotated_v2, decimals=2)}")
    print("As you can see, the vector on the Y-axis correctly rotates to the Z-axis.")

    # Center eye at (0, 0, 0) looking towards positive z-axis
    canvas = Canvas(Vec3(0, 0, 0), Vec3(-1, -1, 1), Vec3(1, 1, 1), 300, 300)
    canvas.add_item(Sphere(0, 0, 2, 0.5))
    canvas.add_item(Sphere(1, 0, 2, 0.4))
    canvas.render()