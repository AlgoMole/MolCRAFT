from typing import Dict
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import fire

def load_pb(file:str):
    return torch.load(file)['stats']

def get_legend(f:str):
    fn = f.split('.')[0]
    if '_' in fn:
        n, s = fn.split('_')
        fn = f"{n} ({s})"
    return fn

def gather_aio(root:str):
    models = ["AR", "Pocket2Mol", "TargetDiff", "DecompDiff", "MolCRAFT", "Ours"]
    return {
        f : load_pb(osp.join(root, f+".pt"))
        for f in models
    }

class CustomColorMap:
    def __init__(self, colors:list) -> None:
        super().__init__()
        self.colors = colors
    def __call__(self, i):
        return self.colors[i % len(self.colors)]


def plot_percentage_data(data: Dict[str, Dict[str, float]], file:str):
    keys = list(next(iter(data.values())).keys())  # Extract x-axis categories from the first method
    categories = [key for key in keys if key != "valid"]
    x = np.arange(len(categories))  # X-axis positions
    plt.figure(figsize=(9, 6))

    #TODO: color and shape need to be polished
    # color_maps = ["viridis", "plasma", "PiYG", "cividis", "Paired", "custom"]
    # color_maps = ["custom"]
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colors = CustomColorMap([
        # "b", "g", "r", "c", "m", "y", "k"
        'tab:' + c
        for c in ('orange', 'green', 'red', 'blue', 'brown', 'pink')
    ])
    # for cm in color_maps:
    #     plt.clf()
        # if cm == "custom":
        #     colors = custom_color
        # else:
        #     colors = plt.cm.get_cmap(cm, round(len(data)*1.5))  # Use a colormap for distinct colors
    marker_style = 'o'  # Unified marker style
    marker_size = 4     # Size of the markers
    highlight_size = 12  # Size to emphasize data points

    for i, (label, values) in enumerate(data.items()):
        y = [values[category] for category in categories]
        plt.plot(x, y, label=label, color=colors(i), marker=marker_style, markersize=marker_size, linewidth=1.5)
        plt.scatter(x, y, color=colors(i), s=highlight_size, edgecolors='white', zorder=5)  # Highlight data points

    fontdict = {
        "fontsize": 12,
        "fontweight": "bold",
    }

    plt.axvline(x=10.5, color='gray', linestyle='--', linewidth=1.5)
    print(f"[left, right, top, bottom] = {[*plt.xlim(), *plt.ylim()]}")
    plt.text(4, 1.025, "PB-Valid Mol", fontdict, color="gray")
    plt.text(13, 1.025, "PB-Valid Dock", fontdict, color="gray")

    plt.xticks(x, [cate.replace('_', ' ') for cate in categories], rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.yticks(np.linspace(0, 1, 11))  # Set y-axis range from 0 to 1 with 10% intervals
    plt.ylim(0.5, 1.05)  # Explicitly set y-axis limits2
    plt.ylabel("The percentage of molecules\nthat passed the check", fontdict)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()
    # plt.show()
    # save_file = osp.join(osp.dirname(file), f"{cm}_{osp.basename(file)}")
    plt.savefig(file, dpi=900)

def cli(outputs:str, save_file:str):
    data = gather_aio(outputs)
    plot_percentage_data(data, save_file)

if __name__ == "__main__":
    fire.Fire(cli)