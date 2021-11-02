import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotting_utils import save_formatted


def dolan_more(
    merged,
    plot_config,
    title="Dolan-More curve",
    dataset_col="Dataset",
    our_method_name="Cohortney",
    more_is_better=True,
):
    df = merged.copy(deep=True)
    betas = pd.DataFrame(df.set_index(dataset_col))
    if more_is_better:
        df["best_performance"] = np.array(df.set_index(dataset_col).max(axis=1))
        for i in df.set_index(dataset_col).columns:
            betas[i] = (
                df.set_index(dataset_col)["best_performance"]
                / df.set_index(dataset_col)[i]
            )
    else:
        df["best_performance"] = np.array(df.set_index(dataset_col).min(axis=1))
        for i in df.set_index(dataset_col).columns:
            betas[i] = (
                df.set_index(dataset_col)[i]
                / df.set_index(dataset_col)["best_performance"]
            )
    max_beta = betas[betas.columns[:-1]].replace(np.inf, 0).max().max()
    space = np.linspace(1, max_beta, 100)
    betas = betas.replace(np.inf, max_beta)

    curves = []
    methods = []
    for i in df.set_index(dataset_col).columns[:-1]:
        curve = []
        methods.append(i)
        for sp in space:
            curve.append(np.sum(betas[i] <= sp) / df.shape[0])
        curves.append(curve)

    with plt.style.context(plot_config["style"]):
        fig, ax = plt.subplots()
        i = 0
        for method, curve in zip(methods, curves):
            if method == our_method_name:
                plt.plot(
                    space,
                    curve,
                    label=method,
                    linewidth=3,
                    c=sns.color_palette("bright", 8)[4],
                    linestyle="--",
                )
            else:
                if i == 4:
                    i += 1
                plt.plot(
                    space, curve, label=method, c=sns.color_palette("bright", 10)[i]
                )
            i += 1
        plt.legend()

    return fig, ax


if __name__ == "__main__":

    plt.style.use("science")
    # metrics are purity, adjusted information score
    naming_dict = {
        "purity": "purities",
        "adjusted mutual info score": "adj_mut_info_score",
        "adjusted rand score": "adj_rand_score",
        "v-measure score": "v_meas_score",
        "fowlkes mallows score": "f_m_score",
    }
    metric = "purity"
    title = "Dolan-More Curve based on " + metric
    for sumtrue in ["", "_sum"]:
        res_df = pd.read_csv(
            naming_dict[metric] + sumtrue + "_dm_res.csv", index_col=False
        )
        res_df.rename(
            columns={
                "COHORTNEY": "Cohortney",
                "DMHP": "Zhu",
                "Conv": "Cae",
                "K-shape": "K-shape tslearn",
                "K-means": "K-means tslearn",
                "K-means0": "K-means tsfresh",
                "GMM": "GMM tsfresh",
            },
            inplace=True,
        )
        cols = [c for i, c in enumerate(res_df.columns) if i > 0]
        res_df = res_df[cols]
        res_df.set_index("Dataset")
        if sumtrue != "_sum":
            res_df.at[12, "Zhu"] = 0.38
            res_df.at[13, "Zhu"] = 0.34
            res_df.at[14, "Zhu"] = 0.31
            res_df.at[15, "Zhu"] = 0.64
        else:
            res_df.at[3, "Zhu"] = 0.42
        print(res_df.head(25))
        with open("plot_config.json") as config:
            plot_config = json.load(config)
        f, ax = dolan_more(res_df, plot_config, title=title)
        save_formatted(
            f,
            ax,
            plot_config,
            save_path=metric + sumtrue + "_dolanmore.pdf",
            xlabel=r"$\beta$",
            ylabel="Proportion of datasets",
            title=title,
        )
