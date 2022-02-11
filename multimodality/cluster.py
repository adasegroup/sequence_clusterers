from sklearn import cluster
from ast import literal_eval
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import json
import numpy as np


def kmeans_cluster(
    path_to_seq_embed="Amazon.json",
    path_to_text_embed="Amazon_text_f.json",
    num_events=8,
    num=7523,
    path_to_save="Amazon_labels.csv",
):
    X = []
    with open(path_to_seq_embed, "r") as f:
        emb_seq = pd.DataFrame([json.load(f)])
    with open(path_to_text_embed, "r") as f:
        emb_text = pd.DataFrame([json.load(f)])
    for i in range(1, num):
        X.append(
            [*literal_eval(emb_seq[str(i)][0]), *literal_eval(emb_text[str(i)][0])]
        )
    kmeans = cluster.KMeans(n_clusters=num_events)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    clusters = {}
    clusters["event_pred"] = labels
    cl = pd.DataFrame.from_dict(clusters)
    cl.to_csv(path_to_save)


if __name__ == "__main__":
    kmeans_cluster()
