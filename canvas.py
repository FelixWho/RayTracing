import math
from item.item import Item
from ray import Ray
from vec3 import Vec3
import numpy as np
from PIL import Image


class Canvas():
    def __init__(self, eye: Vec3, bottom_left: Vec3, top_right: Vec3, pixel_width: int, pixel_height: int):
        self.eye = eye
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.top_left = Vec3(bottom_left[0], top_right[1], bottom_left[2])
        self.bottom_right = Vec3(top_right[0], bottom_left[1], top_right[2])
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.items = []

    def add_item(self, it: Item) -> None:
        self.items.append(it)

    def render(self) -> None:
        print(f"Rendering image ({self.pixel_width}x{self.pixel_height})...")
        # Create a numpy array to hold pixel data, with shape (height, width, 3) for RGB
        image_data = np.zeros((self.pixel_height, self.pixel_width, 3), dtype=np.uint8)

        # Define the viewport geometry
        viewport_horizontal = self.bottom_right - self.bottom_left
        viewport_vertical = self.top_left - self.bottom_left

        # For every pixel
        for w in range(self.pixel_width):
            for h in range(self.pixel_height):
                # Calculate the ray's direction to pass through the center of the pixel
                u = (w + 0.5) / self.pixel_width
                v = (h + 0.5) / self.pixel_height
                
                ray_direction = self.bottom_left + u * viewport_horizontal + v * viewport_vertical - self.eye
                
                r = Ray(self.eye, ray_direction.normalize())

                # Check collision with an item
                minimum_t = math.inf
                nearest_item = None
                for item in self.items:
                    reflected_ray = item.reflect(r)
                    
                    if reflected_ray is not None:
                        # To find the closest object, we calculate the distance 't'.
                        t = np.linalg.norm(reflected_ray.center - r.center)
                        if t < minimum_t:
                            minimum_t = t
                            nearest_item = item
                
                if nearest_item is not None:
                    color_vec = Vec3(1, 1, 0.25)
                else:
                    color_vec = Vec3(0,0,0)
                    
                # Convert color vector (0.0-1.0) to RGB bytes (0-255) and write to the array.
                image_data[self.pixel_height - 1 - h, w] = (255.999 * color_vec).astype(int)
                
        print("Creating image from pixel data...")
        image = Image.fromarray(image_data, 'RGB')
        image.show()