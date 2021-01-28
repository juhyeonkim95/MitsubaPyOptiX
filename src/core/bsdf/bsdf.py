from enum import Enum

class Conductor:
    def __init__(self):
        self.eta = None
        self.k = None
        self.extEta = None
        self.specularReflectance = None


class RoughConductor:
    def __init__(self):
        self.distribution = "ggx"
        self.alpha = 0.1
        self.alphaU = 0.1
        self.alphaV = 0.1
        self.material = "cu"
        self.eta = None
        self.k = None
        self.extEta = None
        self.specularReflectance = None






