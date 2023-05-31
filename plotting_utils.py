import matplotlib.pyplot as plt

def plotter(times, paths, title, x_lbl, y_lbl, alpha, beta):
    plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['figure.dpi'] = 300

    plt.figure(99)
    for x, y in zip(times, paths):
        plt.step(x, y)

    formatted_title = r'$\mathrm{' + title.replace(' ', r'\ ') + r' \alpha = ' + str(alpha) + r', \beta = ' + str(
        beta) + r'}$'

    plt.title(formatted_title, fontsize=14)  # Set title font size with spaces

    plt.xlabel(r'$\mathrm{' + x_lbl + '}$', fontsize=14)  # Re-format x-axis label as LaTeX expression
    plt.ylabel(r'$\mathrm{' + y_lbl + '}$', fontsize=14)  # Re-format y-axis label as LaTeX expression
    plt.grid(True, linewidth=0.5)  # Add grid

    plt.tick_params(axis='x', labelsize=10)  # Set x-axis tick label font size
    plt.tick_params(axis='y', labelsize=10)

    plt.show()
    return plt


def ngplotter(times, paths, title, x_lbl, y_lbl, alpha, beta, mu, sigmasq):
    plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['figure.dpi'] = 300

    plt.figure(99)
    for x, y in zip(times, paths):
        plt.step(x, y)

    formatted_title = r'$\mathrm{' + title.replace(' ', r'\ ') + r', \alpha = ' + str(alpha) + r', \beta = ' + str(beta) + r', \mu = ' + str(mu) + r', \sigma^2 = ' + str(sigmasq) + r'}$'


    plt.title(formatted_title, fontsize=14)  # Set title font size with spaces

    plt.xlabel(r'$\mathrm{' + x_lbl + '}$', fontsize=14)  # Re-format x-axis label as LaTeX expression
    plt.ylabel(r'$\mathrm{' + y_lbl + '}$', fontsize=14)  # Re-format y-axis label as LaTeX expression
    plt.grid(True, linewidth=0.5)  # Add grid

    plt.tick_params(axis='x', labelsize=10)  # Set x-axis tick label font size
    plt.tick_params(axis='y', labelsize=10)

    plt.show()
    return plt

