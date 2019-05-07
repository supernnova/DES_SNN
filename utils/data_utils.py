import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from astropy.table import Table
import utils.logging_utils as lu

def read_fits(fname):
    # load photometry
    dat = Table.read(fname, format='fits')
    df_phot = dat.to_pandas()
    # dailsafe
    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    # load header
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    df_header["SNID"] = df_header["SNID"].astype(np.int32)

    # add SNID to phot for skimming
    arr_ID = np.zeros(len(df_phot), dtype=np.int32)
    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(df_phot["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df_phot)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    df_phot["SNID"] = arr_ID

    return df_header, df_phot

def fetch_header_info(path):
    list_head = glob.glob(f"{path}/*HEAD.FITS")
    list_df_head = []
    for fname in list_head:
        dat = Table.read(fname, format='fits')
        list_df_head.append(dat.to_pandas())
    df_head = pd.concat(list_df_head)
    return df_head


def save_fits(df, fname):
    """Save data frame in fits table

    Arguments:
        df {pandas.DataFrame} -- data to save
        fname {str} -- outname, must end in .FITS
    """

    outtable = Table.from_pandas(df)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    outtable.write(fname, format='fits', overwrite=True)


def save_phot_fits(df, fname):
    """
        fname {str} -- outname, including path
    """
    keep_col_phot = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
    df_phot = df[keep_col_phot]
    df_phot = df_phot.loc[df_phot["FLUXCAL"].shift() != df_phot["FLUXCAL"]]

    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    save_fits(df_phot, fname)


def load_bazin_fits(bazin_file):
    fit = pd.read_csv(bazin_file, comment="#",
                      delimiter=" ", skipinitialspace=True)
    fit["SNID"] = fit["CID"].astype(int)

    return fit
