import glob
import numpy as np
import pandas as pd
from astropy.table import Table


def load_predictions(fname):
    """Load predictions for a given model
    """
    df_pred = pd.read_pickle(fname)

    # compute predicted target for complete lc classification
    if 'vanilla' in fname:
        df_pred["predicted_target"] = (
            df_pred[[k for k in df_pred.keys() if "all_class" in k]]
            .idxmax(axis=1)
            .str.strip("all_class")
            .astype(int))
    else:
        df_pred["predicted_target"] = (
            df_pred[[k for k in df_pred.keys() if "all_class" in k and "median" in k]]
            .idxmax(axis=1)
            .str.strip("all_class")
            .str.strip("_median")
            .astype(int))

    # reset index
    df_pred.index = np.arange(0,len(df_pred))

    return df_pred


def enrich_predictions(df_pred, path_dtype_data):
    """Enrich predictions with lc info
    """
    # from photometry header
    # get headers
    list_head = glob.glob(f"{path_dtype_data}/*HEAD.FITS")
    list_df_head = []
    for fname in list_head:
        dat = Table.read(fname, format='fits')
        list_df_head.append(dat.to_pandas())
    df_head = pd.concat(list_df_head, sort=True)
    df_head['SNID'] = df_head['SNID'].astype(int)

    # merge w. preds
    cols_to_merge = ["SNID"] + [
        k for k in df_head.keys() if k not in df_pred.keys()]
    df_pred = df_pred.merge(
        df_head[cols_to_merge], how="left", on="SNID")

    df_pred['SNTYPE'] = df_pred['SNTYPE'].astype(int)

    return df_pred

def spec_type_decoder(typ):
    try:
        spec_sample_type_dic = {"1": "Ia", "0": "unknown", "2": "SNIax", "3": "SNIa-pec", "20": "SNIIP", "21": "SNIIL", "22": "SNIIn", "29": "SNII",
                                "32": "SNIb", "33": "SNIc", "39": "SNIbc", "41": "SLSN-I", "42": "SLSN-II", "43": "SLSN-R", "80": "AGN", "81": "galaxy", "98": "None", "99": "pending"}
        tag = spec_sample_type_dic[str(typ)]
    except Exception:
        tag = f"{typ}"
    return tag
