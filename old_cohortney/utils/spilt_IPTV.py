import dropbox
import os
import tarfile
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

import zipfile

from data_preprocessor import dropbox_download, stopwatch

DROPBOX_TOKEN = "AS74Amc6RgcAAAAAAAAAAZJXpaexESLjcWQa4NerDECUiuYJ_a1IOrlL7oV1BuhU"

if __name__ == "__main__":

    # download from dropbox
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    dbx_df = dropbox_download(
        dbx, folder="", subfolder="", name="IPTV_Data.zip"
    )
    zf = zipfile.ZipFile(dbx_df)
    zf.extractall("data/IPTV_unzip")
  
    for file in sorted(os.listdir("data/IPTV_unzip"), key=lambda x: 
                       int(re.sub(fr'.{"txt"}', '', x)) if re.sub(fr'.{"txt"}', '', x).isdigit() 
                       else 0):
        if file.endswith(f'.{ext}') and re.sub(fr'.{"txt"}', '', file).isnumeric():
            df = pd.read_csv(Path(data_dir, file))
            classes = classes.union(set(df[event_col].unique()))
            df["time"] = pd.to_datetime(df["time"])
            df["time"] = (df["time"] - df["time"][0]) / np.timedelta64(1,'D')
            #if maxlen > 0:
                #df = df.iloc[:maxlen]
            save_path = "data/IPTV"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(save_path, file.replace('txt','csv')))
    
