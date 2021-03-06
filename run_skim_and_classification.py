import os
import glob
import datetime
import argparse
import pandas as pd
from pathlib import Path
from shutil import copyfile
from functools import partial
import utils.cuts_utils as cu
import utils.data_utils as du
from astropy.table import Table
import utils.logging_utils as lu
from utils import evaluation_utils as eu
from utils import visualization_utils as vu
from utils.data_utils import spec_sample_type_dic


def do_classification(skim_dir, list_models, sntypes):
    """ SNN classification
    """
    import SuperNNova.supernnova.conf as conf
    from SuperNNova.supernnova.data import make_dataset
    from SuperNNova.supernnova.visualization import early_prediction
    from SuperNNova.supernnova.validation import validate_rnn, metrics

    lu.print_blue(f"Classifying {skim_dir}")
    # get config args
    snn_args = conf.get_args()

    # create database
    snn_args.data = True
    snn_args.data_testing = True
    snn_args.dump_dir = f"{skim_dir}/"
    snn_args.raw_dir = f"{skim_dir}/"
    snn_args.fits_dir = "./"
    snn_args.sntypes = sntypes
    settings = conf.get_settings(snn_args)

    # # make dataset
    # make_dataset.make_dataset(settings)

    for model in list_models:
        # add model file to settings
        snn_args.model_files = model
        settings = conf.get_settings(snn_args)

        model_files = [model]
        # # # classify
        # snn_args.validate_rnn = True
        model_settings = conf.get_settings_from_dump(
            settings,
            snn_args.model_files,
            override_source_data=settings.override_source_data,
        )
        # fetch predictions
        prediction_file = validate_rnn.get_predictions(
            model_settings, model_file=snn_args.model_files
        )
        # Compute metrics
        metrics.get_metrics_singlemodel(
            model_settings, prediction_file=prediction_file, model_type="rnn"
        )
        # plot lcs
        model_settings.model_files = snn_args.model_files
        early_prediction.make_early_prediction(model_settings, nb_lcs=20)

        # evaluate classifications
        df = eu.fetch_prediction_info(settings, model_settings, skim_dir)

        # plots init
        path_plot = f"{snn_args.dump_dir}/figures/"
        eu.plot_efficiency(df, skim_dir, path_plot)

    # "the classified sample"
    eu.pair_plots(df, path_plot)


def classify_data(dump_dir):
    list_models = glob.glob("../SuperNNova_general/trained_models_mutant/*/*.pt")
    # in skim we incorporat ethe name change already
    sntypes = spec_sample_type_dic

    # do SNN classifications
    for timevar in ['trigger', 'bazin', 'clump']:
        do_classification(f"{dump_dir}/{timevar}/", list_models, sntypes)


if __name__ == '__main__':

    # settings
    debug = False
    skim = False
    classify = True

    # init paths
    path_des_data = os.environ.get("DES_DATA")
    central_dump_dir = "./dumps/"

    for dtype in ["fake", "real"]:
        if skim:
            raw_dir = f"{path_des_data}/DESALL_forcePhoto_{dtype}_snana_fits/"
            dump_dir = f"{central_dump_dir}/{dtype}/"

            for timevar in ['trigger', 'bazin', 'clump']:
                if timevar == 'clump':
                    fits_file = f"{Path(raw_dir)}/DESALL_{dtype}_clump.SNANA.TEXT"
                else:
                    fits_file = f"{Path(raw_dir)}/DESALL_{dtype}_Bazin_fit.SNANA.TEXT"

                cu.skim_data(raw_dir, dump_dir, fits_file, timevar, debug=debug)

        if classify:
            dump_dir = f"{central_dump_dir}/{dtype}/"
            classify_data(dump_dir)
