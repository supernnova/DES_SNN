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
from collections import OrderedDict
from utils import evaluation_utils as eu
from utils import visualization_utils as vu

spec_sample_type_dic = OrderedDict({"1": "Ia", "0": "unknown", "2": "SNIax", "3": "SNIa-pec", "20": "SNIIP", "21": "SNIIL", "22": "SNIIn", "29": "SNII",
                        "32": "SNIb", "33": "SNIc", "39": "SNIbc", "41": "SLSN-I", "42": "SLSN-II", "43": "SLSN-R", "80": "AGN", "81": "galaxy", "98": "None", "99": "pending"})

model_files = [
        "../SuperNNova_general/trained_models_mutant/vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C/vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C.pt"]

def do_classification(skim_dir,model_files,sntypes):
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
    snn_args.model_files = model_files
    settings = conf.get_settings(snn_args)

    # # make dataset
    make_dataset.make_dataset(settings)

    # # classify
    snn_args.validate_rnn = True
    model_settings = conf.get_settings_from_dump(
        settings,
        snn_args.model_files[0],
        override_source_data=settings.override_source_data,
    )
    # # fetch predictions
    prediction_file = validate_rnn.get_predictions(
        model_settings, model_file=snn_args.model_files[0]
    )
    # # Compute metrics
    metrics.get_metrics_singlemodel(
        model_settings, prediction_file=prediction_file, model_type="rnn"
    )
    # # plot lcs
    model_settings.model_files = snn_args.model_files
    early_prediction.make_early_prediction(model_settings, nb_lcs=20)

    # evaluate classifications
    df = eu.fetch_prediction_info(settings,model_settings,skim_dir)

    # plots init
    path_plot = f"{snn_args.dump_dir}/figures/"
    eu.plot_efficiency(df,skim_dir,path_plot)            
    

def performance_metrics(df, sample_target=0):
    """Get performance metrics
    AUC: only valid for binomial classification, input proba of highest label class.

    Args:
        df (pandas.DataFrame) (str): with columns [target, predicted_target, class1]
        (optional) sample_target (str): for SNIa sample default is target 0

    Returns:
        accuracy, auc, purity, efficiency,truepositivefraction
    """
    from sklearn import metrics
    n_targets = len(np.unique(df["target"]))

    # Accuracy & AUC
    accuracy = metrics.accuracy_score(df["target"], df["predicted_target"])
    accuracy = round(accuracy * 100, 2)
    if n_targets == 2:  # valid for biclass only
        auc = round(metrics.roc_auc_score(df["target"], df["class1"]), 4)
    else:
        auc = 0.0

    SNe_Ia = df[df["target"] == sample_target]
    SNe_CC = df[df["target"] != sample_target]
    TP = len(SNe_Ia[SNe_Ia["predicted_target"] == sample_target])
    FP = len(SNe_CC[SNe_CC["predicted_target"] == sample_target])

    P = len(SNe_Ia)
    N = len(SNe_CC)

    truepositivefraction = P / (P + N)
    purity = round(100 * TP / (TP + FP), 2) if (TP + FP) > 0 else 0
    efficiency = round(100.0 * TP / P, 2) if P > 0 else 0

    return accuracy, auc, purity, efficiency, truepositivefraction

def skim_data(raw_dir,dtype,dump_dir):

        list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))
        # load Bazin
        bazin_file = f"{Path(list_files[0]).parent}/DESALL_{dtype}_Bazin_fit.SNANA.TEXT"
        if Path(bazin_file).exists():
            df_bazin = du.load_bazin_fits(bazin_file)

        for fname in list_files:
            prefix_out = Path(fname).name.split("_")[0]
            dump_prefix = f"{dtype}/{prefix_out}"
            lu.print_blue(f"Processing: {dump_prefix}")

            df_header, df_phot = du.read_fits(fname)

            # skimming
            time_cut_type = 'window'
            timevar = 'trigger'
            SN_threshold = None
            cu.apply_cut_save(df_header, df_phot, time_cut_type=time_cut_type, timevar=timevar,
                              SN_threshold=SN_threshold, dump_dir=dump_dir, dump_prefix=dump_prefix)

            # bazin window
            time_cut_type = 'window'
            timevar = 'bazin'
            SN_threshold = None
            df_header = pd.merge(df_header, df_bazin, on='SNID')
            df_header = df_header[[
                k for k in df_header.keys() if 'Unnamed' not in k]]
            cu.apply_cut_save(df_header, df_phot, time_cut_type=time_cut_type, timevar=timevar,
                              SN_threshold=SN_threshold, dump_dir=dump_dir, dump_prefix=dump_prefix)
            # with S/N
            SN_threshold = 3
            cu.apply_cut_save(df_header, df_phot, time_cut_type=time_cut_type, timevar=timevar,
                              SN_threshold=SN_threshold, dump_dir=dump_dir, dump_prefix=dump_prefix)

def classify_data(dump_dir,dtype):
        # in skim we incorporat ethe name change already
        sntypes = spec_sample_type_dic

        # do SNN classifications
        time_cut_type = 'window'
        SN_threshold = None
        timevar = 'trigger'
        do_classification(f"{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}/{dtype}/",model_files,sntypes)
        timevar = 'bazin'
        do_classification(f"{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}/{dtype}/",model_files,sntypes)

        SN_threshold = 3
        timevar = 'bazin'
        do_classification(f"{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}/{dtype}/",model_files,sntypes)
"""
MAIN
"""

# settings
debug = False
skim = True
classify = True

# init paths
path_des_data = os.environ.get("DES_DATA")
dump_dir = "./dumps/"

for dtype in ["fake", "real"]:
    if skim:
        raw_dir = f"{path_des_data}/DESALL_forcePhoto_{dtype}_snana_fits/"
        skim_data(raw_dir,dtype,dump_dir)

    if classify:
        classify_data(dump_dir,dtype)

# if file exists do no duplicate
# crosscheck spec real
