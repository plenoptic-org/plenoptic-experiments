import torchvision
import einops

def gatys():
    # gatys actually use a modified version of VGG19, see
    # https://github.com/honzukka/texture-synthesis-pytorch/
    weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
    tv_model = torchvision.models.vgg19(weights=weights).eval()
    # This model's transform consists of resizing, cropping, and normalizing.
    # We recommend only including the normalizing in the transform.
    tv_transform = weights.transforms()
    norm = torchvision.transforms.Normalize(tv_transform.mean, tv_transform.std)
    layers = ["features.1", "features.4", "features.9", "features.18", "features.27"]
    def gram(x):
        return {
            k: einops.einsum(v, v, "b c1 h w, b c2 h w -> b c1 c2")
            for k, v in x.items()
        }
    return po.models.DeepNetFeatures(tv_model, layers, norm, gram)
