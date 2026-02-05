from vec3 import Vec3

class Ray:
    center: Vec3
    dir: Vec3

    def __init__(self, center: Vec3, dir: Vec3):
        self.center = center
        self.dir = dir
        self.dir.normalize()
