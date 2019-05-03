import os
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copyfile
from functools import partial
import utils.cuts_utils as cu
import utils.data_utils as du
import utils.logging_utils as lu
from utils import visualization_utils as vu


def preprocess_real_and_fake_lcs(path_raw, path_dump, debug=False):
    """ preprocess data (join head, phot, bazin fits)
    """
    for dtype in ["fake", "real"]:
        du.load_all_data(f"{path_raw}/DESALL_forcePhoto_{dtype}_snana_fits/", f"{path_dump}/{dtype}/", redo_photometry=True, debug=debug, keep_delim=True, preprocess_only=True)

def do_skimming(fname, time_cut='window', SN_threshold=None):
    """ Skim data to reduce lcs
    """
    df = pd.read_pickle(fname)
    # hack to keep the separators
    mask = (df['MJD'] != -777.00)

    # applying cuts
    df = cu.compute_time_cut(
        df, mask, time_cut)
    df = cu.compute_S_N_cut(df, mask, SN_threshold)
    df = cu.apply_cuts(df)

    # output
    prefix_out = Path(fname).name.split("_")[0]
    # save photometry
    fname = f"{skim_dir}/{dtype}/{prefix_out}_skimmed_{dtype}_PHOT.FITS"
    du.save_phot_fits(df, fname)
    # save header
    fname = f"{skim_dir}/{dtype}/{prefix_out}_skimmed_{dtype}_HEAD.FITS"
    du.save_head_fits(df, fname)

    # plot some light-curves
    vu.plot_random_lcs(df, f"{skim_dir}/{dtype}/lightcurves/", multiplots=False, nb_lcs=20)


def do_classification(skim_dir):
    """ SNN classification
    """
    from collections import OrderedDict
    import SuperNNova.supernnova.conf as conf
    from SuperNNova.supernnova.data import make_dataset
    from SuperNNova.supernnova.visualization import early_prediction
    from SuperNNova.supernnova.validation import validate_rnn, metrics

    for dtype in ["fake", "real"]:
        lu.print_green(f"___ classifying {dtype} ___")
        # get config args
        snn_args = conf.get_args()

        # create database
        snn_args.data = True
        snn_args.data_testing = True
        snn_args.dump_dir = f"{skim_dir}/{dtype}/"
        snn_args.raw_dir = f"{skim_dir}/{dtype}/"
        snn_args.fits_dir = "./"
        if dtype == "fake":
            snn_args.sntypes = OrderedDict({"0": "Ia"})
        snn_args.model_files = [
            "../SuperNNova_general/trained_models_mutant/vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C/vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C.pt"]
        settings = conf.get_settings(snn_args)

        # make dataset
        make_dataset.make_dataset(settings)

        # classify
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
        model_settings.model_files = snn_args.model_files
        early_prediction.make_early_prediction(model_settings, nb_lcs=20)


if __name__ == '__main__':

    # settings
    debug = True
    prep = True
    skim = True
    time_cut = 'bazin'  # ['trigger','bazin','subseason']
    SN_threshold = None  # [None,3]

    # init paths
    path_des_data = os.environ.get("DES_DATA")
    prep_dir = './dumps/preprocessed/'
    skim_dir = f"./dumps/{time_cut}_SN{SN_threshold}/"

    if prep:
        preprocess_real_and_fake_lcs(path_des_data, prep_dir, debug=debug)

    if skim:
        for dtype in ["fake", "real"]:
            lu.print_green(f"___ skiming {dtype} ___")

            list_files = glob.glob(f"{prep_dir}/{dtype}/*.pickle")
            if debug:
                list_files = list_files[:2]
            for fname in list_files:
                do_skimming(fname, time_cut=time_cut, SN_threshold=SN_threshold)
            # PARTIAL is bugging!
            #     max_workers = multiprocessing.cpu_count()
            #     process_fn = partial(
            #         do_skimming,
            #         time_cut=time_cut, SN_threshold=SN_threshold
            #     )
            #     list_df = []
            #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
            #         list_df += list(executor.map(process_fn, list_files))
            #     import ipdb; ipdb.set_trace()

            lu.print_green(f"Skimmed files {dtype}")

    # SuperNNova
    do_classification(skim_dir)

    """
    To do:
    - plot lcs classified as Ia in real
    - get efficiency for fakes
    - crosscheck qith spec classified SNe in real

    """
