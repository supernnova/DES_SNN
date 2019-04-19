import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import utils.logging_utils as lu
from functools import partial
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def load_data(raw_dir, dump_dir,redo_photometry = True, debug=False):
    """load and preprocess real and fake data

    Arguments:
        raw_dir {str} -- path to raw data PHOT and HEAD .FITS
        dump_dir {str} -- dump directory

    Keyword Arguments:
        debug {bool} -- optional, load only one file

    Returns:
        PandasDataFrame -- df with real or fake data
    """

    if redo_photometry:
        df_tmp = process_photometry(raw_dir, dump_dir, debug=debug)
        if len(df_tmp)<=5:
            try:
                df = pd.concat([df_tmp[i][0] for i in range(len(df_tmp))])
            except Exception:
                df = pd.concat([df_tmp[i] for i in range(len(df_tmp))])
        else:
            df = df_tmp
    else:
        list_pickle = glob.glob(f"{dump_dir}/*.pickle")
        df_tmp = {}
        for i,fil in enumerate(list_pickle):
            lu.print_green(f"Loading preprocessed data {fil}")
            if debug: print(fil)
            df_tmp[i] = pd.read_pickle(fil)
        df = pd.concat([df_tmp[i] for i in range(len(df_tmp))])

    return df

def process_singlefile_photometry(fname, dump_dir,keep_delim=False):
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
    keep_col_header = [k for k in df_header.keys()]
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
    if not keep_delim:
        df = df[df.MJD != -777.000]
        # Reset the index (it is no longer continuous after dropping lines)
        df.reset_index(inplace=True, drop=True)
    outname = f"{str(Path(fname).stem)}.pickle"
    df.to_pickle(f"{dump_dir}/{outname}")

    return df


def process_photometry(raw_dir, dump_dir, debug=False, keep_delim=False):
    """Preprocess the FITS data

    - multiprocessing ahs fail, to be added later
    - Preprocess every FIT file in thi file list

    Args:
        list_files (list): list of photometry files to process

    Returns:
        df (dictionary of Pandas.DataFrames)

    """
    Path(dump_dir).mkdir(parents=True, exist_ok=True)

    lu.print_green(f"Processing data FITS {raw_dir}")
    list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))

    df ={}

    for i,fil in enumerate(list_files):
        print(fil)
        df[i] = process_singlefile_photometry(
            fil, dump_dir, keep_delim=keep_delim)

    return df

def save_fits(df,fname):
    """Save ata frame in fits table

    Arguments:
        df {pandas.DataFrame} -- data to save
        fname {str} -- outname, must end in .FITS
    """

    outtable = Table.from_pandas(df)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    outtable.write(fname, format='fits')
