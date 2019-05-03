import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from functools import partial
from astropy.table import Table
import utils.logging_utils as lu
from concurrent.futures import ProcessPoolExecutor


def load_all_data(raw_dir, dump_dir, redo_photometry=True, debug=False, keep_delim=False, preprocess_only=False):
    """load data
    preprocess data if needed, else fetch from pickle

    Arguments:
        raw_dir {str} -- path to raw data PHOT and HEAD .FITS
        dump_dir {str} -- dump directory

    Keyword Arguments:
        redo_photometry {bool} -- optional, preprocess
        debug {bool} -- optional, load only one file
        keep_delim {bool} -- optional, keep -777 for phot splits

    Returns:
        PandasDataFrame -- df with real or fake data
    """

    if redo_photometry:
        lu.print_green(f"Processing data FITS {raw_dir}")

        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))

        # fetch bazin file
        bazin_file = f"{Path(list_files[0]).parent}/DESALL_fake_Bazin_fit.SNANA.TEXT" if "fake" in list_files[0] else f"{Path(list_files[0]).parent}/DESALL_real_Bazin_fit.SNANA.TEXT"
        if Path(bazin_file).exists():
            df_bazin = load_bazin_fits(bazin_file)
        else:
            df_bazin = None
            lu.print_yellow("No Bazin fits available!", bazin_file)

        if debug:
            lu.print_yellow('Debugging mode')
            list_files = list_files[0:1]
        df_tmp = {}
        for i, fil in enumerate(list_files):
            print(fil)
            df_tmp[i] = preprocess_single_fits(
                fil, dump_dir=dump_dir, keep_delim=keep_delim, df_bazin=df_bazin)
        # else:
        #     # partial is bugging
        #     # use partial to speed up
        #     max_workers = multiprocessing.cpu_count()
        #     process_fn = partial(
        #         preprocess_single_fits,
        #         dump_dir=dump_dir, keep_delim=keep_delim, df_bazin=df_bazin
        #     )
        #     list_df = []
        #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #         print(list(executor.map(process_fn, list_files)))
        #         # list_df += list(executor.map(process_fn, list_files))
        #     import ipdb
        #     ipdb.set_trace()
        #     df = pd.concat(list_df)
        #     return df

    else:
        lu.print_green(f"Loading preprocessed data {dump_dir} ")
        list_pickle = glob.glob(f"{dump_dir}/*.pickle")
        df_tmp = {}
        for i, fil in enumerate(list_pickle):
            df_tmp[i] = pd.read_pickle(fil)

    if not preprocess_only:
        df = pd.concat([df_tmp[i] for i in range(len(df_tmp))], sort=False)
        return df


def preprocess_single_fits(fname, dump_dir=None, keep_delim=False, df_bazin=pd.DataFrame()):
    """Load photometry and preprocess

    Load PHOT and HEAD.FITS, and Bazin fits
    preprocess them, merge and return DataFrame
    preprocessing: join photometry and header and eliminate separators
    dump to pickle

    Arguments:
        fname {[str]} -- filename of the PHOT.FITS file
        dump_dir [str] -- path to dump pickled product
        keep_delim [optional, Boolean] -- if photo delimiters are kept

    Returns:
        [pandas.DataFrame] -- photometry with header df
    """
    # get fits file
    dat = Table.read(fname, format='fits')
    df = dat.to_pandas()

    if df.MJD.values[-1] == -777.0:
        df = df.drop(df.index[-1])
    if df.MJD.values[0] == -777.0:
        df = df.drop(df.index[0])

    # Load the companion HEAD file
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    keep_col_header = [k for k in df_header.keys()]
    keep_col_header = list(set(keep_col_header))
    df_header = df_header[keep_col_header].copy()
    try:
        # BEWARE! SNID for fakes is actually NOT the real SNID
        df_header['real_SNID'] = df_header['PRIVATE(DES_snid)'].astype(
            np.int32)
        df_header["SNID"] = df_header["SNID"].astype(np.int32)
    except Exception:
        df_header["SNID"] = df_header["SNID"].astype(np.int32)

    # Bazin fits
    if len(df_bazin) > 1:
        lu.print_green("Adding Bazin features")
        df_header = pd.merge(df_header, df_bazin, on="SNID", how="left")

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

    df = pd.merge(df, df_header, on="SNID", how="left")

    lu.print_green("merged")
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


def save_head_fits(df, fname):
    """
    """
    # header
    keep_col_head = [
        "SNID",
        "PEAKMJD",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ",
        "HOSTGAL_SPECZ_ERR",
        "SNTYPE",
    ]
    keep_col_head = keep_col_head + \
        [k for k in df.keys() if 'mjd' in k or 'fake' in k or "PRIVATE(DES_transient_status)" in k]

    df_head = df[df.MJD != -777.000]
    df_head = df_head[keep_col_head]
    df_head = df_head.drop_duplicates()
    df_head = df_head.reset_index()
    df_head = df_head[keep_col_head]
    save_fits(df_head, fname)


def load_bazin_fits(bazin_file):
    fit = pd.read_csv(bazin_file, comment="#",
                      delimiter=" ", skipinitialspace=True)
    fit["SNID"] = fit["CID"].astype(int)

    return fit
