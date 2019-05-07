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
from astropy.table import Table
import utils.logging_utils as lu
from utils import visualization_utils as vu


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

        import ipdb; ipdb.set_trace()
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



# settings
debug = False
prep = True
skim = True

# init paths
path_des_data = os.environ.get("DES_DATA")
dump_dir = "./dumps/"

for dtype in ["fake","real"]:
    raw_dir = f"{path_des_data}/DESALL_forcePhoto_{dtype}_snana_fits/"
    list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))
    # load Bazin
    bazin_file = f"{Path(list_files[0]).parent}/DESALL_{dtype}_Bazin_fit.SNANA.TEXT"
    if Path(bazin_file).exists():
        df_bazin = du.load_bazin_fits(bazin_file)

    for fname in list_files:
        prefix_out = Path(fname).name.split("_")[0]
        dump_prefix = f"{dtype}/{prefix_out}"

        df_header, df_phot = du.read_fits(fname)

        # skimming
        time_cut_type = 'window'
        SN_threshold = None
        timevar = 'trigger'
        cu.apply_cut_save(df_header,df_phot, time_cut_type = time_cut_type, timevar = timevar, SN_threshold= SN_threshold, dump_dir=dump_dir,dump_prefix = dump_prefix)

        # bazin window
        timevar = 'bazin'
        df_header = pd.merge(df_header,df_bazin,on='SNID')

        cu.apply_cut_save(df_header,df_phot, time_cut_type = time_cut_type, timevar = timevar, SN_threshold= SN_threshold, dump_dir=dump_dir,dump_prefix = dump_prefix)

    time_cut_type = 'window'
    SN_threshold = None
    timevar = 'trigger'
    do_classification(f"{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}")
    timevar = 'bazin'
    do_classification(f"{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}")

    # if file exists do no duplicate
    # get efficiency class
    # crosscheck spec real

        # BEWARE classification at is , will create a db for each year. WIll ahve to group this !!!
