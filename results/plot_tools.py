from pathlib import Path
import math
import os

import matplotlib as mpl
import numpy as np


project_dir = Path(__file__).resolve().parent.parent


alg_names = {
    "30v": "BIGUE (RW+CT)",
    "30v_long": "BIGUE thinned",
    "30v_hmc": "Dynamic HMC*",
    "30v_rw": "RW",
    "30v_rw_long": "RW thinned",
}

init_full_names = {
    "random": "Random initialization",
    "mercator": "Mercator initialization",
    "groundtruth": "Ground truth initialization"
}


def pt_to_inch(pt):
    return pt * 0.0138888889


def cm_to_inch(cm):
    return cm * 0.3937007874


def cap_01(value):
    if value > 1:
        return 1
    if value < 0:
        return 0
    return value

def darken_color(color, saturation_correction=1., light_correction=0.7):
    hsv_color = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color))
    hsv_color[2] = cap_01(hsv_color[2]*light_correction)
    hsv_color[1] = cap_01(hsv_color[1]*saturation_correction)
    return mpl.colors.hsv_to_rgb(hsv_color)


# column_width = pt_to_inch(246) # For revtex
column_width = cm_to_inch(9) # For Comm. Phys.

all_colors = mpl.pyplot.rcParams['axes.prop_cycle'].by_key()['color']
alg_colors = {
    "posterior": "#084B83",
    "rw": "#B6443E",
    "hmc": "#F3B47C",
    "mercator": "#74ad08",
    "reference": "#a1a1a1"
}
hist_colors = {
    "mercator": "#cbe898",
    "posterior": "#68aadc",
    "reference": "#a1a1a1"
}
dark_reference = darken_color(alg_colors["reference"], light_correction=0.5)


def figure_dir():
    fig_dir = project_dir.joinpath("figures")
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir

def format_ticks(denom: int):
    if denom <= 0:
        raise ValueError("Invalid denominator for axis formatter.")

    tick_values, tick_labels = [], []
    for num in range(-denom, denom+1):
        tick_values.append(num/denom*np.pi)
        gcd = math.gcd(num, denom)
        tick_labels.append(format_frac(num, abs(num//gcd), denom//gcd))

    return tick_values, tick_labels

def format_frac(sign, num, denom):
    if num == 0:
        return "$0$"
    if num == 1 and denom == 1:
        return f"${'-' if sign<0 else ''}\\pi$"
    return f"${'-' if sign<0 else ''}\\frac{{{num if num != 1 else ''}\\pi}}{{{denom}}}$"
