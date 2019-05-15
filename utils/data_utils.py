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
from collections import OrderedDict

spec_sample_type_dic = OrderedDict({"1": "Ia", "0": "unknown", "2": "SNIax", "3": "SNIa-pec", "20": "SNIIP", "21": "SNIIL", "22": "SNIIn", "29": "SNII",
                                    "32": "SNIb", "33": "SNIc", "39": "SNIbc", "41": "SLSN-I", "42": "SLSN-II", "43": "SLSN-R", "80": "AGN", "81": "galaxy", "98": "None", "99": "pending"})


def spec_type_decoder(typ):
    try:
        spec_sample_type_dic = {"1": "Ia", "0": "unknown", "2": "SNIax", "3": "SNIa-pec", "20": "SNIIP", "21": "SNIIL", "22": "SNIIn", "29": "SNII",
                                "32": "SNIb", "33": "SNIc", "39": "SNIbc", "41": "SLSN-I", "42": "SLSN-II", "43": "SLSN-R", "80": "AGN", "81": "galaxy", "98": "None", "99": "pending"}
        tag = spec_sample_type_dic[str(typ)]
    except Exception:
        tag = f"{typ}"
    return tag


def read_fits(fname):
    # load photometry
    dat = Table.read(fname, format='fits')
    df_phot = dat.to_pandas()
    # failsafe
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
    df_head = pd.concat(list_df_head, sort=True)
    return df_head


def save_fits(df, fname):
    """Save data frame in fits table

    Arguments:
        df {pandas.DataFrame} -- data to save
        fname {str} -- outname, must end in .FITS
    """
    df = df.reset_index()
    outtable = Table.from_pandas(df)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    outtable.write(fname, format='fits', overwrite=True)


def save_phot_fits(df, fname):
    """
        fname {str} -- outname, including path
    """
    keep_col_phot = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
    # eliminate repeated rows (-777.0)
    df_phot = df.copy()
    df_phot = df_phot.loc[df_phot["FLUXCAL"].shift() != df_phot["FLUXCAL"]]

    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    mask_seven = df_phot['MJD'] == -777.0
    df_phot.loc[mask_seven, 'SNID'] = 0

    df_phot = df_phot.reset_index()
    df_phot_saved = df_phot[keep_col_phot]
    save_fits(df_phot_saved, fname)
    return df_phot


def load_bazin_fits(bazin_file):
    fit = pd.read_csv(bazin_file, comment="#",
                      delimiter=" ", skipinitialspace=True)
    fit["SNID"] = fit["CID"].astype(int)

    return fit


def load_predictions_and_info(skim_dir, model_name):
    df_pred_tmp = pd.read_pickle(f"{skim_dir}/models/{model_name}/PRED_{model_name}.pickle")
    # compute predicted target for complete lc classification
    df_pred_tmp["predicted_target"] = (
        df_pred_tmp[[k for k in df_pred_tmp.keys() if "all_class" in k]]
        .idxmax(axis=1)
        .str.strip("all_class")
        .astype(int))

    # add header info
    df_SNinfo = fetch_header_info(skim_dir)
    cols_to_merge = ["SNID"] + [
        k for k in df_SNinfo.keys() if k not in df_pred_tmp.keys()]
    df_pred = df_pred_tmp.merge(
        df_SNinfo[cols_to_merge], how="left", on="SNID")

    return df_pred


def load_fitres(filepath):
    '''Load light curve fitres file
    Arguments:
        filepath (str) -- Lightcurve fit file with path   
    Returns:
        pandas dataframe 
    '''
    list_fitres = glob.glob(f"{filepath}/*FITRES")
    df = pd.read_csv(list_fitres[0], index_col=False,
                     comment='#', delimiter=' ', skipinitialspace=True)
    df['SNID'] = df['CID']

    return df
