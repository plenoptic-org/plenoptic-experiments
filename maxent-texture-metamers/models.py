from maxent_texture import load_pretrained

class MaxEntTexture(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = load_pretrained()

    def forward(self, *args, **kwargs):
        return self.model.stat_model(*args, **kwargs)

def maxent_texture():
    return MaxEntTexture()
