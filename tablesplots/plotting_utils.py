import matplotlib.pyplot as plt


def save_formatted(fig, ax, settings, save_path, xlabel=None, ylabel=None, title=None):
    """
    Saves (fig, ax) object to save_path with settigns json file
    """
    # labels and title
    if settings["show labels"]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    if settings["show title"]:
        plt.title(title)
    #
    ax.tick_params(
        axis="both", which="major", labelsize=settings["tick labels font size"]
    )
    ax.tick_params(
        axis="both", which="minor", labelsize=2 * settings["tick labels font size"] // 3
    )
    # set height and width
    fig.set_figheight(settings["fig height"])
    fig.set_figwidth(settings["fig width"])
    # set aspect ratio
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(settings["aspect ratio"] * abs((x1 - x0) / (y1 - y0)))
    # save
    plt.savefig(save_path, dpi=settings["dpi"], bbox_inches="tight")
    plt.clf()
