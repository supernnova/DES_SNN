import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from astropy.table import Table
import utils.logging_utils as lu
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
        df = pd.concat([df_tmp[i][0] for i in range(len(df_tmp))])
    else:
        for i,fil in enumerate(list_pickle):
            lu.print_green(f"Loading preprocessed data {dump_dir}/preprocessed")
            if debug: print(fil)
            df_tmp = {}
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
    keep_col_header = [
        "SNID",
        "PEAKMJD",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ",
        "HOSTGAL_SPECZ_ERR",
        "SNTYPE",
    ]
    keep_col_header = keep_col_header + \
        [k for k in df_header.keys() if 'mjd' in k or 'fake' in k]
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
    # import ipdb; ipdb.set_trace()
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
    df.to_pickle(f"{dump_dir}/preprocessed/{outname}")

    return df


def process_photometry(raw_dir, dump_dir, debug=False, keep_delim=False):
    """Preprocess the FITS data

    - Use multiprocessing/threading to speed up data processing
    - Preprocess every FIT file in thi file list

    Args:
        list_files (list): list of photometry files to process

    Returns:
        df (dictionary of Pandas.DataFrames)

    """
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    preprocessed_dir = f"{dump_dir}/preprocessed"
    Path(preprocessed_dir).mkdir(parents=True, exist_ok=True)

    lu.print_green(f"Processing data FITS {raw_dir}")
    list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))

    df ={}

    if not debug and len(list_files) > 1:
        # Parameters of multiprocessing below
        parallel_fn = partial(process_singlefile_photometry, dump_dir=dump_dir,
                               keep_delim=keep_delim)
        max_workers = multiprocessing.cpu_count()

        # Split list files in fake chunks
        # to get a progress bar and alleviate memory constraints
        num_elem = len(list_files)
        num_chunks = len(list_files)
        list_chunks = np.array_split(np.arange(num_elem), num_chunks)

        # Loop over chunks of files
        df_list = []

        for i, chunk_idx in enumerate(tqdm(list_chunks, desc="Process photometry", ncols=100)):
            # Process each file in the chunk in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                start, end = chunk_idx[0], chunk_idx[-1] + 1
                # Need to cast to list because executor returns an iterator
                df[i] = list(executor.map(parallel_fn,
                                             list_files[start:end]))

    else:
        if debug: lu.print_yellow(f"Debug mode: single file {list_files[-1]}")
        df[0] = process_singlefile_photometry(
            list_files[-1], dump_dir, keep_delim=keep_delim)

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
