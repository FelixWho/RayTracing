from abc import ABC, abstractmethod
from ray import Ray
import numpy as np
from numpy_utils import *


class Item(ABC):
    def __init__(self, x, y, z):
        self.center = array_of(x, y, z)

    @abstractmethod
    def reflect(self, incoming_ray: Ray) -> Ray | None:
        """Compute the normalized ray reflection when bounced off object, or None if there's no intersection"""
        pass
