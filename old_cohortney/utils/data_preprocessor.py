import contextlib
import os

import dropbox
import pandas as pd
import time
import torch
import tqdm


def cmp_to_key(mycmp):
    """
    Convert a cmp= function into a key= function
    """

    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def compare(a, b):
    tmp1 = int(a[:-4])
    tmp2 = int(b[:-4])
    return tmp1 - tmp2


def label_dataset(path_to_files, files):
    """
    Does label enconding and replaces dataset
    inputs:
            path_to_files - str, path to dataset
            files - list of str, datapoints file names
    outputs:
            None
    """
    files = sorted(files, key=cmp_to_key(compare))
    evnts = {}
    cur = 0
    for i, f in tqdm.tqdm(enumerate(files)):
        print("File: {}".format(i))
        df = pd.read_csv(path_to_files + "/" + f)
        for i in range(len(df["event"])):
            if df["event"].iloc[i] not in evnts:
                evnts[df["event"].iloc[i]] = cur
                cur += 1
            df["event"].iloc[i] = evnts[df["event"].iloc[i]]
        df["event"] = df["event"].astype(int)
        df.to_csv(path_to_files + "/" + f)


evnts = {}
cur = 0


@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print("Total elapsed time for %s: %.3f" % (message, t1 - t0))


def dropbox_download(dbx, folder, subfolder, name):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    path = "/%s/%s/%s" % (folder, subfolder.replace(os.path.sep, "/"), name)
    while "//" in path:
        path = path.replace("//", "/")
    with stopwatch("download"):
        try:
            md, res = dbx.files_download(path)
        except dropbox.exceptions.HttpError as err:
            print("*** HTTP error", err)
            return None
    data = res.content
    print(len(data), "bytes; md:", md)
    return data


def get_partition(
    df, num_of_steps, num_of_classes, col_to_select: str = None, end_time=None
):
    """
    Transforms dataset into partition
    inputs:
            df - pandas.DataFrame, columns - time and event
            num_of_steps - int, number of steps in partition
            num_of_classes - int, number of event types
            end_time - float, end time or None
    outputs:
            partition - torch.Tensor, size = (num_of_steps, num_of_classes + 1)
    """
    if col_to_select is None:
        df = df.loc[:, ["time", "event"]].copy()
    else:
        df = df[["time", col_to_select]].copy()
        # df = df.loc[:, ['time', col_to_select]].copy()
        df.rename(columns={col_to_select: "event"}, inplace=True)

    df = df.sort_values(by=["time"])
    df = df.reset_index().loc[:, ["time", "event"]].copy()
    # setting end time if None
    if end_time is None:
        end_time = df["time"][len(df["time"]) - 1]

    # preparing output template
    res = torch.zeros(num_of_steps, num_of_classes + 1)

    # finding delta time
    dt = end_time / num_of_steps
    res[:, 0] = end_time / num_of_steps

    # converting time to timestamps
    df["time"] = (df["time"] / dt).astype(int)
    mask = df["time"] == num_of_steps
    df.loc[mask, "time"] -= 1

    # counting points
    df = df.reset_index()
    df = df.groupby(["time", "event"]).count()
    df = df.reset_index()
    df.columns = ["time", "event", "num"]
    try:
        df["event"] = df["event"].astype(int)
    except:
        global evnts
        global cur
        for i in range(len(df["event"])):
            if df["event"].iloc[i] not in evnts:
                evnts[df["event"].iloc[i]] = cur
                cur += 1
            df["event"].iloc[i] = evnts[df["event"].iloc[i]]
        df["event"] = df["event"].astype(int)

    # computing partition
    tmp = torch.Tensor(df.to_numpy()).long()
    res[tmp[:, 0], tmp[:, 1] + 1] = tmp[:, 2].float()

    return res


def get_dataset(path_to_files, n_classes, n_steps, col_to_select=None, n_files=None):
    """
    Reads dataset
    inputs:
            path_to_files - str, path to csv files with dataset
            n_classes - int, number of event types
            n_steps - int, number of steps in partitions
    outputs:
            data - torch.Tensor, size = (N, n_steps, n_classes + 1), dataset
            target - torch.Tensor, size = (N), true labels or None
    """
    # searching for files
    files = os.listdir(path_to_files)
    target = None
    last_event_target = False

    # reading target
    if "clusters.csv" in files:
        files.remove("clusters.csv")
        target = torch.Tensor(
            pd.read_csv(path_to_files + "/clusters.csv")["cluster_id"]
        )
        if n_files is not None:
            target = target[:n_files]
    if "info.json" in files:
        files.remove("info.json")

    # reading data
    files = sorted(files, key=cmp_to_key(compare))
    if n_files is not None:
        files = files[:n_files]
    data = torch.zeros(len(files), n_steps, n_classes + 1)
    for i, f in tqdm.tqdm(enumerate(files)):
        # print('File: {}'.format(f))
        df = pd.read_csv(path_to_files + "/" + f)
        data[i, :, :] = get_partition(df, n_steps, n_classes, col_to_select)

    return data, target