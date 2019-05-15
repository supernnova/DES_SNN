import os
import numpy as np
import pandas as pd
from pathlib import Path
from random import shuffle
import utils.data_utils as du
import matplotlib.pyplot as plt
import utils.logging_utils as lu
from utils import evaluation_utils as eu
from utils import visualization_utils as vu


def get_photo_sample_w_spec_tags(df_pred, photo_SNIDs=None):

    # define photo samples
    photo_Ia = {}
    photo_nonIa = {}

    photo_tmp = df_pred[df_pred['predicted_target'] == 0].copy()

    if photo_SNIDs:
        photo_Ia['all'] = photo_tmp[photo_tmp['SNID'].isin(photo_SNIDs)]
    else:
        photo_Ia['all'] = photo_tmp
        photo_SNIDs = photo_tmp.SNID.values

    photo_nonIa['all'] = df_pred[~df_pred['SNID'].isin(photo_SNIDs)].copy()

    photo_Ia['spec_Ia'] = photo_Ia['all'][(
        photo_Ia['all']['TYPE'] == 1) | (photo_Ia['all']['TYPE'] == 101)]
    photo_Ia['spec_nonIa'] = photo_Ia['all'][(photo_Ia['all']['TYPE'] != 0) & (
        photo_Ia['all']['TYPE'] != 1) & (photo_Ia['all']['TYPE'] != 101)]
    photo_Ia['all_no_spec_nonIa'] = photo_Ia['all'][(photo_Ia['all']['TYPE'] == 0) | (
        photo_Ia['all']['TYPE'] == 1) | (photo_Ia['all']['TYPE'] == 101)]

    photo_nonIa['spec_Ia'] = photo_nonIa['all'][(
        photo_nonIa['all']['TYPE'] == 1) | (photo_nonIa['all']['TYPE'] == 101)]
    photo_nonIa['spec_nonIa'] = photo_nonIa['all'][(photo_nonIa['all']['TYPE'] != 0) & (
        photo_nonIa['all']['TYPE'] != 1) & (photo_nonIa['all']['TYPE'] != 101)]

    return photo_Ia, photo_nonIa


def get_sample_stats_and_plots(df_pred, photo_Ia, photo_nonIa, skim_dir, model_files=None, out_dir=None, plot=False):

    # inspect sample
    path_plots = f"{skim_dir}/figures/"
    Path(path_plots).mkdir(parents=True, exist_ok=True)
    vars_to_plot = [k for k in ['REDSHIFT_FINAL', 'PRIVATE(DES_numepochs_ml)', 'all_class0', 'PRIVATE(DES_cand_type)',
                                'TYPE', 'PRIVATE(DES_mjd_trigger)', 'PKMJDINI'] if k in photo_Ia['all'].keys()]
    if 'fake' in skim_dir:
        df_dic = {'all_lcs': df_pred, 'photo Ia sample': photo_Ia['all']}
        vars_to_plot += ['PRIVATE(DES_fake_salt2x1)',
                         'PRIVATE(DES_fake_salt2c)']
    else:
        df_dic = {'all_lcs': df_pred, 'photo Ia sample': photo_Ia['all'],
                  'contaminants': photo_Ia['spec_nonIa'], 'photo other but spec Ia ': photo_nonIa['spec_Ia']}
    for var in [k for k in vars_to_plot]:
        vu.plot_superimposed_hist(df_dic, var, nameout=f"{path_plots}/hist_{var}_dist.png", log=True)

    for var in ['FLUXCAL_max', 'SNRMAX1']:
        # photo sample zoom
        vu.plot_superimposed_hist(df_dic, var, nameout=f"{path_plots}/hist_{var}_dist.png", log=True, limits_from_photo_sample=True)

    # Stats
    lu.print_green(cut_type)
    eu.print_percentages(f"photo Ias          ", photo_Ia['all'], df_pred, color='blue')
    if dtype == 'real':
        lu.print_blue(f"      are spec Ias  ", len(photo_Ia['spec_Ia']))
        lu.print_blue(f'      are spec other', len(photo_Ia['spec_nonIa']))
        lu.print_blue(f'               gals ', len(photo_Ia['spec_nonIa'][photo_Ia['spec_nonIa']['TYPE'] == 81]))
        lu.print_red(f"missed Ias          ", len(photo_nonIa['spec_Ia']))

    # dump sample
    if not out_dir:
        out_dir = f"{skim_dir}/sample/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    # filla na
    photo_Ia['all_no_spec_nonIa'] = photo_Ia['all_no_spec_nonIa'].fillna(0)
    photo_Ia['spec_nonIa'] = photo_Ia['spec_nonIa'].fillna(0)
    photo_Ia['all_no_spec_nonIa'][[
        'SNID', 'HOSTGAL_OBJID', 'DEC', 'RA', 'TYPE', 'REDSHIFT_FINAL', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ', 'all_class0', 'c', 'x1']].to_csv(f'{out_dir}/photo_Ia.csv')
    # dump contaminants
    photo_Ia['spec_nonIa'][[
        'SNID', 'HOSTGAL_OBJID', 'DEC', 'RA', 'TYPE', 'REDSHIFT_FINAL', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ', 'all_class0', 'c', 'x1']].to_csv(f'{out_dir}/photo_Ia_spec_contamination.csv')
    # dump missed Ias
    photo_nonIa['spec_Ia'][[
        'SNID', 'HOSTGAL_OBJID', 'DEC', 'RA', 'TYPE', 'REDSHIFT_FINAL', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ', 'all_class0', 'c', 'x1']].to_csv(f'{out_dir}/photo_nonIa_spec_Ia.csv')

    # plot lcs
    if plot:
        vu.plot_early_classification(
            skim_dir, prefix='photo_Ia_', df=photo_Ia['all'], model_files=model_files, out_dir=out_dir)
        vu.plot_early_classification(skim_dir, prefix='photo_Ia_spec_contamination',
                                     df=photo_Ia['spec_nonIa'], model_files=model_files, out_dir=out_dir)
        vu.plot_early_classification(skim_dir, prefix='photo_nonIa_',
                                     df=photo_nonIa['all'], model_files=model_files, out_dir=out_dir)
        vu.plot_early_classification(skim_dir, prefix='photo_nonIa_spec_Ia_',
                                     df=photo_nonIa['spec_Ia'], model_files=model_files, out_dir=out_dir)

        # sample histograms of type
        vu.plot_hist(photo_Ia['all'], 'TYPE', nameout=f"{path_plots}/photo_Ia_hist_type.png", log=True)
        vu.plot_hist(photo_nonIa['all'], 'TYPE', nameout=f"{path_plots}/photo_nonIa_hist_type.png", log=True)

        df_dic = {'photo Ia sample': photo_Ia['all'], 'photo & spec Ia ': photo_Ia['spec_Ia'],
                  'photo other but spec Ia ': photo_nonIa['spec_Ia']}
        for var in ['REDSHIFT_FINAL', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ', 'all_class0']:
            vu.plot_superimposed_hist(df_dic, var, nameout=f"{path_plots}/hist_{var}_dist_spec.png", log=True, only_positive_x=True)
        for var in ['FLUXCAL_max', 'c', 'x1']:
            vu.plot_superimposed_hist(df_dic, var, nameout=f"{path_plots}/hist_{var}_dist_spec.png", log=True, only_positive_x=False, bins=20)


"""
Main
"""

plot = True

path_des_data = os.environ.get("DES_DATA")
model_name = "vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C"
model_files = [
    f"../SuperNNova_general/trained_models_mutant/{model_name}/{model_name}.pt"]

df_pred = {}
photo_Ia = {}
photo_nonIa = {}
for dtype in ["real", "fake"]:
    print()
    lu.print_blue(f'_____STATS FOR {dtype}_____')
    print()
    for cut_type in ['window_bazin_SNNone', 'window_trigger_SNNone']:
        lu.print_green(cut_type)

        skim_dir = f"./dumps/{dtype}/{cut_type}/"

        # fetch predictions
        df_pred[cut_type] = du.load_predictions_and_info(skim_dir, model_name)

        # add salt2 fit parameters
        raw_dir = f"{path_des_data}/DESALL_forcePhoto_{dtype}_snana_fits/"
        saltfit = du.load_fitres(raw_dir)
        saltfit = saltfit[['SNID']+[k for k in saltfit.keys()
                                    if k not in df_pred[cut_type].keys()]]
        df_pred[cut_type] = df_pred[cut_type].merge(saltfit, on='SNID')

        # get photo samples
        photo_Ia[cut_type], photo_nonIa[cut_type] = get_photo_sample_w_spec_tags(
            df_pred[cut_type])

        # stats for sample
        get_sample_stats_and_plots(df_pred[cut_type], photo_Ia[cut_type],
                                   photo_nonIa[cut_type], skim_dir, model_files=model_files, plot=plot)

        # time cut study
        if dtype == 'fake':
            eu.get_time_cut_stats_and_plot(
                df_pred[cut_type], photo_Ia[cut_type]['all'], skim_dir, cut_type, plot=plot, model_files=model_files)

    # get common sample of the two cut types
    lu.print_green("common")
    common_dir = f"./dumps/common/{dtype}"
    Path(skim_dir).mkdir(parents=True, exist_ok=True)
    common_SNIDs = [k for k in photo_Ia['window_bazin_SNNone']['all'].SNID.values if int(
        k) in photo_Ia['window_trigger_SNNone']['all'].SNID.values.astype(int)]
    common_photo_Ia, common_photo_nonIa = get_photo_sample_w_spec_tags(
        df_pred['window_trigger_SNNone'], photo_SNIDs=common_SNIDs)

    # venn
    vu.plot_venn(photo_Ia['window_bazin_SNNone']['all'], photo_Ia['window_trigger_SNNone']['all'], 'SNID', nameout=f"{common_dir}/venn_common_sample.png")

    # stats for sample
    get_sample_stats_and_plots(df_pred['window_trigger_SNNone'], common_photo_Ia,
                               common_photo_nonIa, skim_dir, model_files=model_files, out_dir=common_dir, plot=plot)
