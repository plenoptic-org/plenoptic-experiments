class MEI(torch.nn.Module):
    def __init__(self, pad_mode, minmax, model="gaussian"):
        super().__init__()
        if model == "gaussian":
            self.model = po.models.Gaussian(30, pad_mode=pad_mode, cache_filt=True)
        elif model == "centersurround":
            self.model = po.models.CenterSurround(30, pad_mode=pad_mode, cache_filt=True)
        elif model == "LGC":
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, pretrained=True, cache_filt=True)
        elif model == "LGC-big":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] *= 2
            state_dict["center_surround.surround_std"] *= 2
            self.model.load_state_dict(state_dict)
        elif model == "LGC-small":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] /= 2
            state_dict["center_surround.surround_std"] /= 2
            self.model.load_state_dict(state_dict)
        elif model == "LGC-bigish":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] *= 1.5
            state_dict["center_surround.surround_std"] *= 1.5
            self.model.load_state_dict(state_dict)
        elif model == "LGC-smallish":
            orig_model = po.models.LuminanceGainControl(31, pretrained=True)
            self.model = po.models.LuminanceGainControl(31, pad_mode=pad_mode, cache_filt=True)
            state_dict = orig_model.state_dict()
            state_dict["center_surround.center_std"] /= 1.5
            state_dict["center_surround.surround_std"] /= 1.5
            self.model.load_state_dict(state_dict)
        self.minmax = minmax
        if minmax == "expmax":
            self.process = lambda x: torch.exp(-x)
        elif minmax == "max":
            self.process = lambda x: -x
        self.eval()
        po.remove_grad(self)

    def forward(self, x):
        return self.process(self.model(x).mean())


# def mei_gaussian_circular_expmax():
#     return MEI("circular", "expmax", "gaussian")

# def mei_gaussian_constant_expmax():
#     return MEI("constant", "expmax", "gaussian")

# def mei_gaussian_circular_max():
#     return MEI("circular", "max", "gaussian")

# def mei_gaussian_constant_max():
#     return MEI("constant", "max", "gaussian")

# def mei_centersurround_circular_expmax():
#     return MEI("circular", "expmax", "centersurround")

# def mei_centersurround_constant_expmax():
#     return MEI("constant", "expmax", "centersurround")

# def mei_centersurround_circular_max():
#     return MEI("circular", "max", "centersurround")

# def mei_centersurround_constant_max():
#     return MEI("constant", "max", "centersurround")

# def mei_LGC_circular_expmax():
#     return MEI("circular", "expmax", "LGC")

# def mei_LGC_constant_expmax():
#     return MEI("constant", "expmax", "LGC")

def mei_LGC_circular_max():
    return MEI("circular", "max", "LGC")

def mei_LGC_constant_max():
    return MEI("constant", "max", "LGC")

def mei_LGCbig_circular_max():
    return MEI("circular", "max", "LGC-big")

def mei_LGCbig_constant_max():
    return MEI("constant", "max", "LGC-big")

def mei_LGCbigish_circular_max():
    return MEI("circular", "max", "LGC-bigish")

def mei_LGCbigish_constant_max():
    return MEI("constant", "max", "LGC-bigish")

def mei_LGCsmallish_circular_max():
    return MEI("circular", "max", "LGC-smallish")

def mei_LGCsmallish_constant_max():
    return MEI("constant", "max", "LGC-smallish")

def mei_LGCsmall_circular_max():
    return MEI("circular", "max", "LGC-small")

def mei_LGCsmall_constant_max():
    return MEI("constant", "max", "LGC-small")
