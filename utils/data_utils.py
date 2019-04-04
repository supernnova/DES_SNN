import os
from tqdm import tqdm
import glob
import random
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def process_singlefile_photometry(fname,snn_like_cols):
    """Load photometry and preprocess
    
    Load PHOT and HEAD.FITS, preprocess them, merge and return DataFrame
    
    Arguments:
        fname {[str]} -- filename of the PHOT.FITS file
    
    Returns:
        [pandas.DataFrame] -- photometry with header df
    """
    # get fits file
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()

    if df.MJD.values[-1] == -777.0:
        df = df.drop(df.index[-1])

    # Load the companion HEAD file
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    if snn_like_cols:
        keep_col_header = [
            "SNID",
            "PEAKMJD",
            "HOSTGAL_PHOTOZ",
            "HOSTGAL_PHOTOZ_ERR",
            "HOSTGAL_SPECZ",
            "HOSTGAL_SPECZ_ERR",
            "SNTYPE",
        ]
        df_header = df_header[keep_col_header].copy()
    else:
        keep_col_header = [
            "SNID",
            "PEAKMJD",
            "HOSTGAL_PHOTOZ",
            "HOSTGAL_PHOTOZ_ERR",
            "HOSTGAL_SPECZ",
            "HOSTGAL_SPECZ_ERR",
            "SNTYPE",
        ] + [k for k in df_header.keys() if 'mjd' in k]
        keep_col_header = list(set(keep_col_header))
        df_header = df_header[keep_col_header].copy()
    df_header["SNID"] = df_header["SNID"].astype(np.int32)

    #############################################
    # Compute SNID for df and join with df_header
    #############################################
    arr_ID = np.zeros(len(df), dtype=np.int32)

    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(df["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    df["SNID"] = arr_ID
    df = df.set_index("SNID")
    df_header = df_header.set_index("SNID")

    # join df and header
    df = df.join(df_header).reset_index()
    # filters have a trailing white space which we remove
    df.FLT = df.FLT.apply(lambda x: x.rstrip()).values.astype(str)

    # Drop the delimiter lines
    df = df[df.MJD != -777.000]
    # Reset the index (it is no longer continuous after dropping lines)
    df.reset_index(inplace=True, drop=True)

    return df

def process_photometry(list_files,debug=False,snn_like_cols=False):
    """Preprocess the FITS data

    - Use multiprocessing/threading to speed up data processing
    - Preprocess every FIT file in thi file list

    Args:
        list_files (list): list of photometry files to process

    """

    # Parameters of multiprocessing below
    parallel_fn = partial(process_singlefile_photometry,snn_like_cols=snn_like_cols)
    max_workers = multiprocessing.cpu_count()

    # Split list files in chunks of size 10 or less
    # to get a progress bar and alleviate memory constraints
    num_elem = len(list_files)
    num_chunks = num_elem // 2 + 1
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)

    # Loop over chunks of files
    df_list = []
    if not debug and len(list_files)>1:
            for chunk_idx in tqdm(list_chunks, desc="Process photometry", ncols=100):
                # Process each file in the chunk in parallel
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    start, end = chunk_idx[0], chunk_idx[-1] + 1
                    # Need to cast to list because executor returns an iterator
                    df_list += list(executor.map(parallel_fn, list_files[start:end]))
            df = pd.concat([df_list[i] for i in range(len(df_list))])
    else:
        df = process_singlefile_photometry(list_files[-1])
        df_list = []

    # for i in range(len(list_files)):
    #     print(i)
    #     df = process_singlefile_photometry(list_files[i],snn_like_cols=snn_like_cols)
    #     df_list.append(df)
    # df = pd.concat([df_list[i] for i in range(len(list_files))])

    return df