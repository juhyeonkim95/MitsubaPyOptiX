class SceneConfig:
    def __init__(self, scene_name, spp_sqrt, sigma_s=0.0, sigma_a=0.0, hg_g=0.9):
        self.sigma_s = sigma_s
        self.sigma_a = sigma_a
        self.sigma_t = sigma_s + sigma_a
        self.hg_g = hg_g
        self.scene_name = scene_name
        self.spp_sqrt = spp_sqrt
        self.spp = spp_sqrt * spp_sqrt
        self.width = 512
        self.height = 512


