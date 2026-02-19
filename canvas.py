import math
from item.item import Item
from ray import Ray
from numpy_utils import *
import numpy as np
from PIL import Image


class Canvas():
    def __init__(self, eye: np.ndarray, bottom_left: np.ndarray, top_right: np.ndarray, pixel_width: int, pixel_height: int):
        self.eye = eye
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.top_left = np.array([bottom_left[0], top_right[1], bottom_left[2]])
        self.bottom_right = np.array([top_right[0], bottom_left[1], top_right[2]])
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.items = []

    def add_item(self, it: Item) -> None:
        self.items.append(it)

    def render(self) -> None:
        print(f"Rendering image ({self.pixel_width}x{self.pixel_height})...")
        # Create a numpy array to hold pixel data, with shape (height, width, 3) for RGB

        # Define the viewport geometry
        viewport_horizontal = self.bottom_right - self.bottom_left
        viewport_vertical = self.top_left - self.bottom_left

        # 1. Generate grid of coordinates (Vectorized)
        # We use linspace to generate all w and h values at once
        w_vals = np.linspace(0, self.pixel_width - 1, self.pixel_width)
        # We reverse h_vals so the image is rendered top-to-bottom (h=height-1 is top)
        h_vals = np.linspace(self.pixel_height - 1, 0, self.pixel_height)
        
        # Create 2D grids for pixel coordinates
        pixel_x, pixel_y = np.meshgrid(w_vals, h_vals)

        # Calculate x and y for every pixel simultaneously
        x = (pixel_x + 0.5) / self.pixel_width
        y = (pixel_y + 0.5) / self.pixel_height

        # Calculate ray directions: (H, W, 1) * (3,) -> (H, W, 3)
        # We add a new axis to x and y to broadcast against the 3D vectors
        ray_directions = (self.bottom_left + 
                          (x[..., np.newaxis] * viewport_horizontal) + 
                          (y[..., np.newaxis] * viewport_vertical) - 
                          self.eye)

        # Normalize directions manually (numpy_utils.normalize is not vectorized for batches)
        norms = np.linalg.norm(ray_directions, axis=2, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized_dirs = ray_directions / norms

        # Create a single Ray object containing the bundle of all rays
        r = Ray(self.eye, normalized_dirs)

        # Initialize depth buffer (infinity) and hit mask
        min_t = np.full((self.pixel_height, self.pixel_width), np.inf)
        hit_mask = np.zeros((self.pixel_height, self.pixel_width), dtype=bool)

        # Check collision with items
        for item in self.items:
            # item.reflect is expected to handle the (H, W, 3) ray bundle
            reflected_ray = item.reflect(r)
            
            if reflected_ray is not None:
                # Calculate distance 't' for all pixels: shape (H, W)
                # reflected_ray.center should be (H, W, 3)
                dists = np.linalg.norm(reflected_ray.center - r.center, axis=2)
                
                # Find where this object is closer than previous ones
                closer = dists < min_t
                
                # Update buffers
                min_t[closer] = dists[closer]
                hit_mask[closer] = True

        # Generate image data based on hits
        # Start with background color (black)
        image_data = np.zeros((self.pixel_height, self.pixel_width, 3), dtype=np.uint8)
        
        # Apply object color where we had a hit
        # Note: Original code used hardcoded yellow [1, 1, 0.25] for any hit
        object_color = np.array([1, 1, 0.25])
        image_data[hit_mask] = (255.999 * object_color).astype(np.uint8)

        print("Creating image from pixel data...")
        image = Image.fromarray(image_data, 'RGB')
        image.show()