import matplotlib
import matplotlib.pyplot as plt
from torch import arange, relu, tanh
from torch.nn.functional import leaky_relu

matplotlib.rc('text', usetex=True)
# matplotlib.rc('font', **{'family': 'sans-serif'})
params = {'text.latex.preamble': r'''
\usepackage{amsmath}
\newcommand\ddfrac[2]{\frac{\displaystyle #1}{\displaystyle #2}}
'''}
plt.rcParams.update(params)


def plot_fn(x, y, label, filename):
    fig = plt.figure(figsize=(6, 6))
    fig.tight_layout(pad=0.01)

    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y, linewidth=2, color='#76B900', label=label)
    ax.legend(handlelength=0, handletextpad=0, fancybox=True)
    ax.grid(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    if filename:
        fig.savefig(f'graphs/{filename}', bbox_inches='tight', pad_inches=0.01)
    plt.show()


# ReLU
eq = r''' \Large $ \sigma(x) = max(0, x) $ '''
x = arange(-3, 3, 0.01)
y = relu(x)
plot_fn(x, y, eq, 'ReLU.eps')

# LReLU
eq = r''' \Large $ \sigma(x) = \begin{cases} x, & \text{ if } x \geq 0 \\ \text{negative\_slope} \times x, & \text{ otherwise } \end{cases} $ '''
x = arange(-3, 3, 0.01)
y = leaky_relu(x, 0.2)
plot_fn(x, y, eq, 'LReLU.eps')

# Tanh
eq = r''' \Large $ \sigma(x) = \ddfrac{e^{x} - e^{-x}} {e^{x} + e^{-x}} $ '''
x = arange(-3, 3, 0.01)
y = tanh(x)
plot_fn(x, y, eq, 'Tanh.eps')
