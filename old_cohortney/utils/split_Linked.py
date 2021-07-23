import dropbox
import os
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

from data_preprocessor import dropbox_download, stopwatch

DROPBOX_TOKEN = "AS74Amc6RgcAAAAAAAAAAZJXpaexESLjcWQa4NerDECUiuYJ_a1IOrlL7oV1BuhU"

if __name__ == "__main__":

    # download from dropbox
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    dbx_df = dropbox_download(
        dbx, folder="", subfolder="", name="LinkedIn_labelled.csv"
    )
    bk_df = pd.read_csv(BytesIO(dbx_df))
    # local path to input file
    # bk_df = pd.read_csv("../data/booking_challenge_tpp_labeled.csv")
    bk_df = bk_df[
        [   "id",
            "time",
            "event",
            "label",
        ]
    ]
    # rename and recode datetime variable
    bk_df["time"] = (bk_df["time"] - bk_df["time"].min()) 
    # minimum length to filter out sequences
    bk_df["seqlen"] = bk_df.groupby(["id"])["time"].transform("count")
 
    # label encoding categorical variables

    bk_df["event"] = bk_df["event"].astype("category")
    bk_df["event"] = bk_df["event"].cat.codes
    # seqlen stats
    grouped_df = bk_df[["id", "seqlen"]]
    grouped_df = bk_df.groupby("id").agg({"seqlen": "mean"}).reset_index()
    grouped_df = grouped_df[["seqlen"]]
    print("seqlen stats")
    print("mean", grouped_df["seqlen"].mean())
    print(grouped_df.quantile([0.25, 0.5, 0.75]))
    bk_df.drop(columns=["seqlen"], inplace=True)
    # encoding label
    bk_df["label"] = bk_df["label"].astype("category")
    bk_df["label"] = bk_df["label"].cat.codes

    unique_ids = bk_df["id"].unique().tolist()
    gt_clusters = []

    # number of classes
    print("# event =", len(bk_df["event"].unique().tolist()))
 
    # number of labels
    print("# labels =", len(bk_df["label"].unique().tolist()))

    save_path = "data/LinkedIn"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    i = 1
    for id0 in unique_ids:
        curr_df = bk_df[bk_df["id"] == id0].copy()
        gt_clusters.append(int(curr_df["label"].mean()))
        curr_df.drop(columns=["id", "label"], inplace=True)
        curr_df.reset_index(drop=True, inplace=True)
        curr_df.to_csv(os.path.join(save_path, str(i) + ".csv"))
        i += 1

    # saving gt cluster labels
    pd.DataFrame(gt_clusters, columns=["cluster_id"]).to_csv(
        os.path.join(save_path, "clusters.csv")
    )