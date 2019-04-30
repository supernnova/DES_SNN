import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu


def preprocess_all_data(path_raw, path_dump, debug=False):
    # preprocess data (join head, phot)
    du.load_all_data(f"{path_raw}/DESALL_forcePhoto_real_snana_fits/", f"{path_dump}/real/", redo_photometry=True, debug=debug, keep_delim=True, preprocess_only=True)
    du.load_all_data(f"{path_raw}/DESALL_forcePhoto_fake_snana_fits/", f"{path_dump}/fake/", redo_photometry=True, debug=debug, keep_delim=True, preprocess_only=True)


def compute_time_cut(df, mask, time_cut):
    """
    time cut 
    """
    if time_cut == "window":
        df['delta_time'] = df['MJD']-df['PRIVATE(DES_mjd_trigger)']
        df['time_cut'] = True
        df.loc[mask, 'time_cut'] = df["delta_time"].apply(lambda x: True if (
            x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False))
        return df
    elif time_cut == "subseason":
        df = compute_delta_time(df)
        df = reformat_subseasons(df)
        import ipdb
        ipdb.set_trace()
        # dummy time cut, since it is already done
        df['time_cut'] = True
        return df
    elif time_cut == None:
        return df


def compute_S_N_cut(df, mask,SN_threshold = None):
    # S/N cut (for limiting magnitudes)
    # loose cut S/N 3
    df['S/N'] = df['FLUXCAL']/df['FLUXCALERR']
    df['S/N_cut'] = True
    if SN_threshold:
        df.loc[mask,
               'S/N_cut'] = df["S/N"].apply(lambda x: True if x > 3 else False)

    return df


def apply_cuts(df, mask):

    df_sel = df[(df['time_cut'] == True) & (df['S/N_cut'] == True)]
    df_sel = df_sel.reset_index()

    return df_sel

def insert_row(idx, df, df_insert):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]

    df = dfA.append(df_insert).append(dfB).reset_index(drop = True)

    return df

def reformat_subseasons(df):
    """ Reformat file with subseasons

    - for each unique SNID, find subseasons
    - split with new SNIDs and -777.0 separators

    (new header will be derived from this file)
    """

    subseason_break_index = df.index[df['delta_time']>100]
    df_insert = df.iloc[0].copy()
    for key in df.keys():
        if key != "FIELD":
            df_insert[key] = -777.0
        else:
            df_insert[key] = "XXXX"
    print(len(df))
    for ind in subseason_break_index:
        df = insert_row(ind-1, df, df_insert)
        import ipdb; ipdb.set_trace()


def compute_delta_time(df):
    """Compute the delta time between two consecutive observations

    Args:
        df (pandas.DataFrame): dataframe holding lightcurve data

    Returns:
        (pandas.DataFrame) dataframe holding lightcurve data with delta_time features
    """
    if df.MJD.values[0] == -777.0:
        df = df.drop(df.index[0])

    df["delta_time"] = df["MJD"].diff()
    # Fill the first row with 0 to replace NaN
    df.delta_time = df.delta_time.fillna(0)
    try:
        IDs = df.SNID.values
    # Deal with the case where lightcrv_ID is the index
    except AttributeError:
        assert df.index.name == "SNID"
        IDs = df.index.values
    # Find idxs of rows where a new light curve start then zero delta_time
    # and where the photo splits are (-777)
    idxs = np.array((np.where(IDs[:-1] != IDs[1:])[0] + 1).tolist() +
                    (np.where(IDs[:-1] != IDs[1:])[0] + 2).tolist())
    arr_delta_time = df.delta_time.values
    arr_delta_time[idxs] = 0
    df["delta_time"] = arr_delta_time

    return df


# def do_classification(skim_dir):
        # evaluate classification
    from collections import OrderedDict
    import SuperNNova.supernnova.conf as conf
    from SuperNNova.supernnova.data import make_dataset
    from SuperNNova.supernnova.visualization import early_prediction
    from SuperNNova.supernnova.validation import validate_rnn, metrics

    for dtype in ["fake", "real"]:
        # get config args
        snn_args = conf.get_args()

        # create database
        snn_args.data = True
        snn_args.data_testing = True
        snn_args.dump_dir = f"{skim_dir}/{dtype}/"
        snn_args.raw_dir = f"{skim_dir}/{dtype}/"
        snn_args.fits_dir = "./"
        snn_args.sntypes = OrderedDict({"0": "Ia"})
        snn_args.model_files = [
            "../SuperNNova_general/trained_models_mutant/vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C/vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C.pt"]
        settings = conf.get_settings(snn_args)

        make_dataset.make_dataset(settings)

        snn_args.validate_rnn = True
        model_settings = conf.get_settings_from_dump(
            settings,
            snn_args.model_files[0],
            override_source_data=settings.override_source_data,
        )
        # fetch predictions
        prediction_file = validate_rnn.get_predictions(
            model_settings, model_file=snn_args.model_files[0]
        )
        # Compute metrics
        metrics.get_metrics_singlemodel(
            model_settings, prediction_file=prediction_file, model_type="rnn"
        )
        # plot lcs
        # model_settings.model_files = snn_args.model_files
        # early_prediction.make_early_prediction(model_settings, nb_lcs=20)


if __name__ == '__main__':

    # settings
    debug = False
    prep = False
    prep_dir = './dumps/preprocessed/'
    skim = True
    time_cut = 'window' #['window','bazin','subseason']
    SN_threshold = 3 #None 

    skim_dir = f"./dumps/{time_cut}_SN{SN_threshold}/"

    # init paths
    path_des_data = os.environ.get("DES_DATA")

    if prep:
        # preprocess ALL data to pickle
        preprocess_all_data(path_des_data, prep_dir, debug=debug)

    # skimming
    if skim:
        for dtype in ["fake"]:#, "real"]:
            list_files = glob.glob(f"{prep_dir}/{dtype}/*.pickle")
            for fname in list_files:
                df = pd.read_pickle(fname)

                # hack to keep the separators
                mask = (df['MJD'] != -777.00)

                # applying cuts
                df = compute_time_cut(df, mask, time_cut)
                df = compute_S_N_cut(df, mask,SN_threshold)

                df = apply_cuts(df, mask)

                # output
                prefix_out = Path(fname).name.split("_")[0]
                # save photometry
                # will need numbering later
                fname = f"{skim_dir}/{dtype}/{prefix_out}_skimmed_{dtype}_PHOT.FITS"
                du.save_phot_fits(df, fname)

                # save header
                fname = f"{skim_dir}/{dtype}/{prefix_out}_skimmed_{dtype}_HEAD.FITS"
                du.save_head_fits(df, fname)
            lu.print_green(f"Skimmed files {dtype}")

    # do_classification(skim_dir)

    """
    To do:
    - plot lcs classified as Ia in real
    - get efficiency for fakes
    - crosscheck qith spec classified SNe in real

    - time cut by subseason
    - time cut by exponential fit (do i need subseason for this?)
    """
