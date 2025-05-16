import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from plot_tools import figure_dir


def pt_to_inch(pt):
    return pt * 0.0138888889
column_width = pt_to_inch(246)


all_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

midblack ="#3d3d3d"
rcParams["font.family"] = "CMU"

plt.rc('text.latex', preamble=r'\usepackage{amsmath, dsfont}')


def sigmoid(x, b):
    return 1/(1+np.exp(-b*x))

def continuous_abs(x, b):
    return x*(2*sigmoid(x, b)-1)

def distance(x, abs_func):
    return np.pi-abs_func(np.pi-abs_func(x))


fig, ax = plt.subplots(1, 1, figsize=(column_width, 0.6*column_width))

xvalues = np.linspace(-2*np.pi, 2*np.pi, 200)
ax.plot(xvalues, distance(xvalues, np.abs), label="Exact", color=midblack)

ax.set_xlabel(r"$\theta_u-\theta_v$")
ax.set_ylabel(r"Angular separation")


b = 3
ax.plot(xvalues, distance(xvalues, lambda x: continuous_abs(x, b)),
        label=f"Approx.", color=all_colors[2], ls="--")

ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [r"$-2\pi$", r"$-\pi$", "$0$", r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, np.pi/2, np.pi], [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
ax.legend()
fig.tight_layout()


output_dir = figure_dir()
fig.savefig(str(output_dir.joinpath("continuous_approx.pdf")))
plt.show()
