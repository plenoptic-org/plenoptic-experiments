import torchvision
import numpy as np
import torch.nn.functional as F


class ChannelModel(torch.nn.Module):
    def __init__(self, layer, channel_n, upsample_scales=False, jitter_pix=False):
        super().__init__()
        if isinstance(layer, str):
            layer = [layer]
        if len(layer) != 1:
            raise ValueError("Only works with single layer!")
        self._layer = layer[0]
        weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
        deepnet = torchvision.models.googlenet(weights=weights).eval()
        transform = weights.transforms()
        norm = [torchvision.transforms.Normalize(transform.mean, transform.std)]
        if jitter_pix:
            def inner_pad(x):
                return F.pad(x, [0, 0, int(jitter_pix * 1.5), int(jitter_pix*1.5)], mode="constant", value=0.5)
            def inner_jitter1(x):
                return torch.roll(x, (np.random.choice(jitter_pix),
                                      np.random.choice(jitter_pix)),
                                  (-2, -1))
            norm = [*norm, inner_pad, inner_jitter1]
        if upsample_scales:
            def inner_upsample(x):
                scale = np.random.choice(upsample_scales)
                upsample = torch.nn.Upsample(scale_factor=scale, mode="bilinear",
                                             align_corners=True)
                return upsample(x)
            norm = [*norm, inner_upsample]
        if jitter_pix:
            def inner_jitter2(x):
                return torch.roll(x, (np.random.choice(jitter_pix//2),
                                      np.random.choice(jitter_pix//2)),
                                  (-2, -1))
            norm = [*norm, inner_jitter2]
        norm = torchvision.transforms.Compose(norm)
        self.model = po.models.DeepNetFeatures(deepnet, layer, norm)
        self.channel_n = channel_n

    def forward(self, x):
        x = self.model.transform(x)
        out = self.model.extractor(x)
        out = {k: v[:, self.channel_n] for k, v in out.items()}
        out = self.model.convert_to_tensor(out)
        if (out < 0).any():
            raise ValueError("NEGATIVE VALUES")
        return out


class FeatureVis(torch.nn.Module):
    def __init__(self, layer, channel_n, upsample, jitter, minmax="max"):
        super().__init__()
        self.model = ChannelModel(layer, channel_n, upsample, jitter)
        if minmax == "min":
            self.process = lambda x: x
        elif minmax == "max":
            self.process = lambda x: torch.exp(-x)
        elif minmax == "realmax":
            self.process = lambda x: -x

    def forward(self, x):
        return self.process(self.model(x).mean())


# def feature_vis_min_4c_448():
#     return FeatureVis("inception4c", 448, "min")


def feature_vis_max_4c_448():
    return FeatureVis("inception4c", 448, False, False, "max")


# def feature_vis_min_4c_448_transformed():
#     return FeatureVis("inception4c", 448, 1.1, True, "min")


def feature_vis_max_4c_448_transformed():
    return FeatureVis("inception4c", 448, [0.8, 0.9, 1.1, 1.2] + 4*[1], 8, "max")


# def feature_vis_realmax_4a_476_transformed():
#     return FeatureVis("inception4a", 476, [0.8, 0.9, 1.1, 1.2] + 4*[1], 8, "realmax")

# def feature_vis_realmax_4a_476():
#     return FeatureVis("inception4a", 476, False, False, "realmax")

def feature_vis_max_4a_476_transformed():
    return FeatureVis("inception4a", 476, [0.8, 0.9, 1.1, 1.2] + 4*[1], 8, "max")

def feature_vis_max_4a_476():
    return FeatureVis("inception4a", 476, False, False, "max")
