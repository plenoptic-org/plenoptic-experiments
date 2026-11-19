#!/usr/bin/env python3

import itertools
import stat
from collections import OrderedDict
from pathlib import Path
from cyclopts import App, Parameter
from typing import Annotated
import warnings


app = App()


def dict_str_to_path(v):
    for char in ["{", "}", " ", "'",'"']:
        v = v.replace(char, "")
    v = v.replace(":", "-")
    v = v.replace(",", "_")
    return v.replace(".", "-")


def get_specific_kwargs(kwargs_dict, arg_string):
    val, i = arg_string.split("-")
    tmp = kwargs_dict.get(val, "{}")
    if tmp != "{}":
        tmp = tmp[int(i)]
    return val, tmp, dict_str_to_path(tmp)


@app.default
def main(
    base_out: Path = Path("/mnt/ceph/users/wbroderick/plenoptic_experiments/deepdream_new/run-4"),
    to_script: bool = False,
    write_to_log: bool = True,
    out_file: Path = Path("hyperparam_search.sh"),
    max_iter: None | int = None,
    device: Annotated[list[str], Parameter(consume_multiple=True)] = ["cpu"],
    dtype:  Annotated[list[str], Parameter(consume_multiple=True)] = ["torch.float32"],
):
    """Write script calling synthesize.py for all combinations of hyperparameters.

    Created file contains independent calls to synthesize.py, each on a separate line.
    This file can be executed as a bash script (in which case, you should pass
    --to-script) or used as input to other process.

    Parameters
    ----------
    base_out :
        Base output directory. Results files and logs will be placed within this
        directory.
    to_script :
        Whether the created file will be executed as a bash script, in which case
        we add ``set -euxo pipefail`` at the top and make the resulting file executable.
    write_to_log :
        Whether each call to synthesize.py should write its output to a log file
        (created alongside the results files) or stdout/stderr
    out_file :
        Name of the file to create.
    max_iter :
        Number of iterations to run synthesis for. Setting max_iter in this manner will give
        all calls to synthesize.py identical values. If you instead wish to loop over max_iter,
        edit this file and set its value in the params dictionary.
    device :
        Which device to run on. If a list of devices, will do all of them.
    dtype :
        Which dtype to use. If a list of dtypes, will do all of them.
    """
    commands = []

    # These can be modified if you wish to change them.
    if to_script:
        commands.extend(["#!/usr/bin/env bash", "set -euxo pipefail"])
    cluster_prefix = "PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch/kernels TORCH_HOME=~/.cache/torch MPLCONFIGDIR=~/.cache/matplotlib PLENOPTIC_CACHE_DIR=~/.cache/plenoptic"
    base_out.mkdir(exist_ok=True)

    # Edit the following lines in order to modify what values you will search over. As
    # long as you're using the same type of synthesis object (e.g., Metamer), you
    # shouldn't need to re-generate synthesize.py
    params = OrderedDict()
    params["loss_function"] = ["feature_vis_max_4c_448", "feature_vis_max_4c_448_transformed", "feature_vis_max_4a_476_transformed", "feature_vis_max_4a_476", ]
    params["penalty_function"] = ["penalize_range", ]
    params["initial_image"] = ["randn", "randn_add", "rand_small", "rand", ]
    params["optimizer"] = ["Adam", ]
    params["scheduler"] = ["None", ]
    params["penalty_lambda"] = [0.1, 0.01, 0.001, ]
    params["max_iter"] = [100, ]
    params["stop_criterion"] = [0, ]
    params["stop_iters_to_check"] = [50, ]
    params["seed"] = [0, ]
    params["dtype"] = dtype
    params["device"] = device
    # everything in params needs to be a string
    params = {k: [str(v_) for v_ in v] for k, v in params.items()}
    optimizer_kwargs = {'Adam': ['{"lr": 0.05}', '{"lr": 0.01}', '{"lr": 0.1}']}
    scheduler_kwargs = {'None': ['{}']}

    # The lines from here to the end of the file should probably not be modified
    tmp = []
    for k in params["optimizer"]:
        if k in optimizer_kwargs:
            tmp.extend([f"{k}-{i}" for i in range(len(optimizer_kwargs[k]))])
        else:
            tmp.append(f"{k}-0")
    params["optimizer"] = tmp
    tmp = []
    for k in params["scheduler"]:
        if k in scheduler_kwargs:
            tmp.extend([f"{k}-{i}" for i in range(len(scheduler_kwargs[k]))])
        else:
            tmp.append(f"{k}-0")
    params["scheduler"] = tmp
    keys = list(params.keys())
    keys.append("optimizer_kwargs")
    keys.append("scheduler_kwargs")

    if max_iter is not None:
        params["max_iter"] = [max_iter]

    path_keys = [k for k, v in params.items() if len(v) > 1]
    if any([len(v) > 1 for v in optimizer_kwargs.values()]):
        path_keys.append("optimizer_kwargs")
    if any([len(v) > 1 for v in scheduler_kwargs.values()]):
        path_keys.append("scheduler_kwargs")
    if len(path_keys) == 0:
        raise ValueError("Not searching over anything!")
    
    path_template = "_".join([f'{k}-{{{k}}}' for k in path_keys]) + ".pt"
    args_template = " ".join([f"--{k} '{{{k}}}'" for k in keys])
    print(f"{path_template=}")
    
    for vals in itertools.product(*params.values()):
        vals = OrderedDict({k: v for k, v in zip(keys, vals)})
        path_vals = vals.copy()
        val, kw, path_kw = get_specific_kwargs(optimizer_kwargs, vals['optimizer'])
        vals["optimizer"] = val
        path_vals["optimizer_kwargs"] = path_kw
        vals["optimizer_kwargs"] = kw
        val, kw, path_kw = get_specific_kwargs(scheduler_kwargs, vals['scheduler'])
        vals["scheduler"] = val
        path_vals["scheduler_kwargs"] = path_kw
        vals["scheduler_kwargs"] = kw
        outfile = base_out / path_template.format(**path_vals)
        if outfile.exists():
            warnings.warn(f"{outfile=} exists, skipping...")
            continue
        cmd = "python synthesize.py " + args_template.format(**vals)
        cmd = f"({cluster_prefix} {cmd} --out-path {outfile})"
        if write_to_log:
            cmd += f" &> {outfile.with_suffix('.log')}"
        commands.append(cmd)

    out_file.write_text('\n'.join(commands))
    out_file.chmod(out_file.stat().st_mode | stat.S_IEXEC)

if __name__ == "__main__":
    app()
