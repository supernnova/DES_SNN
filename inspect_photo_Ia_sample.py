import os
import re
import sys
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from random import shuffle
import utils.data_utils as du
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SuperNNova.supernnova.conf as conf
from utils import visualization_utils as vu
from SuperNNova.supernnova.utils import logging_utils as lu
from SuperNNova.supernnova.utils import training_utils as tu
from SuperNNova.supernnova.utils import data_utils as snn_du
from SuperNNova.supernnova.utils.visualization_utils import FILTER_COLORS, ALL_COLORS, LINE_STYLE


def get_predictions(settings, dict_rnn, X, target, OOD=None):

    list_data = [(X.copy(), target)]

    _, X_tensor, *_ = tu.get_data_batch(list_data, [0], settings, OOD=OOD)

    if settings.use_cuda:
        X_tensor.cuda()

    seq_len = X_tensor.shape[0]
    d_pred = {key: {"prob": []} for key in dict_rnn}

    # Loop over light curve time steps to obtain prediction for each time step
    for i in range(1, seq_len + 1):
        # Slice along the time step dimension
        X_slice = X_tensor[:i]

        # Apply rnn to obtain prediction
        for model_type, rnn in dict_rnn.items():

            n = settings.num_inference_samples if "variational" in model_type else 1
            new_size = (X_slice.size(0), n, X_slice.size(2))

            if "bayesian" in model_type:

                # Loop over num samples to obtain predictions
                list_out = [
                    rnn(X_slice.expand(new_size))
                    for i in range(settings.num_inference_samples)
                ]
                out = torch.cat(list_out, dim=0)
                # Apply softmax to obtain a proba
                pred_proba = nn.functional.softmax(
                    out, dim=-1).data.cpu().numpy()
            else:
                out = rnn(X_slice.expand(new_size))
                # Apply softmax to obtain a proba
                pred_proba = nn.functional.softmax(
                    out, dim=-1).data.cpu().numpy()

            # Add to buffer list
            d_pred[model_type]["prob"].append(pred_proba)

    # Stack
    for key in dict_rnn.keys():
        arr_proba = np.stack(d_pred[key]["prob"], axis=0)
        d_pred[key]["prob"] = arr_proba  # arr_prob is (T, num_samples, 2)
        d_pred[key]["median"] = np.median(arr_proba, axis=1)
        d_pred[key]["perc_16"] = np.percentile(arr_proba, 16, axis=1)
        d_pred[key]["perc_84"] = np.percentile(arr_proba, 84, axis=1)
        d_pred[key]["perc_2"] = np.percentile(arr_proba, 2, axis=1)
        d_pred[key]["perc_98"] = np.percentile(arr_proba, 98, axis=1)

    return d_pred, X_tensor.squeeze().detach().cpu().numpy()


def plot_predictions(
    settings, d_plot, SNID, redshift, target, arr_time, d_pred, OOD, prefix=None, title=None, out_dir =None
):

    plt.figure()
    gs = gridspec.GridSpec(2, 1)
    # Plot the lightcurve
    ax = plt.subplot(gs[0])
    for flt in d_plot.keys():
        flt_time = d_plot[flt]["MJD"]
        # Only plot a time series if it's non empty
        if len(flt_time) > 0:
            flux = d_plot[flt]["FLUXCAL"]
            fluxerr = d_plot[flt]["FLUXCALERR"]
            ax.errorbar(
                flt_time,
                flux,
                yerr=fluxerr,
                fmt="o",
                label=f"Filter {flt}",
                color=FILTER_COLORS[flt],
            )
    ax.set_ylabel("FLUXCAL")
    ylim = ax.get_ylim()
    ax.set_title(title)

    # Plot the classifications
    ax = plt.subplot(gs[1])
    ax.set_ylim(0, 1)

    for idx, key in enumerate(d_pred.keys()):

        for class_prob in range(settings.nb_classes):
            color = ALL_COLORS[class_prob + idx * settings.nb_classes]
            linestyle = LINE_STYLE[class_prob]
            label = snn_du.sntype_decoded(class_prob, settings)

            if len(d_pred) > 1:
                label += f" {key}"

            ax.plot(
                arr_time,
                d_pred[key]["median"][:, class_prob],
                color=color,
                linestyle=linestyle,
                label=label,
            )
            ax.fill_between(
                arr_time,
                d_pred[key]["perc_16"][:, class_prob],
                d_pred[key]["perc_84"][:, class_prob],
                color=color,
                alpha=0.4,
            )
            ax.fill_between(
                arr_time,
                d_pred[key]["perc_2"][:, class_prob],
                d_pred[key]["perc_98"][:, class_prob],
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("classification probability")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if out_dir:
        fig_path = (
            f"{out_dir}/lightcurves/{settings.pytorch_model_name}/{prefix}early_prediction"
        )
        fig_name = (
            f"{settings.pytorch_model_name}_{prefix}class_pred_with_lc_{SNID}.png"
        )
    else:
        if len(settings.model_files) > 1:
            fig_path = f"{settings.figures_dir}/{prefix}multi_model_early_prediction"
            fig_name = f"{prefix}multi_model_{SNID}.png"
        elif len([settings.model_files]) == 1:
            parent_dir = Path(settings.model_files[0]).parent.name
            fig_path = f"{settings.lightcurves_dir}/{parent_dir}/{prefix}early_prediction"
            fig_name = f"{parent_dir}_{prefix}class_pred_with_lc_{SNID}.png"
        else:
            fig_path = (
                f"{settings.lightcurves_dir}/{settings.pytorch_model_name}/{prefix}early_prediction"
            )
            fig_name = (
                f"{settings.pytorch_model_name}_{prefix}class_pred_with_lc_{SNID}.png"
            )
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(fig_path) / fig_name)
    plt.clf()
    plt.close()


def make_early_prediction(settings, nb_lcs=1, do_gifs=False, df=None, prefix=None,out_dir=None):
    """
    """
    available_ids = df.SNID.values.tolist()

    settings.random_length = False
    settings.random_redshift = False

    # Load the test data
    list_data_test = tu.load_HDF5(settings, test=True)

    # Load features list
    file_name = f"{settings.processed_dir}/database.h5"
    with h5py.File(file_name, "r") as hf:
        features = hf["features"][settings.idx_features]

    # Load RNN model
    dict_rnn = {}
    if settings.model_files is None:
        settings.model_files = [f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt"]
    else:
        # check if the model files are there
        tmp_not_found = [
            m for m in settings.model_files if not os.path.exists(m)]
        if len(tmp_not_found) > 0:
            print(lu.str_to_redstr(f"Files not found {tmp_not_found}"))
            tmp_model_files = [
                m for m in settings.model_files if os.path.exists(m)]
            settings.model_files = tmp_model_files

    # Check that the settings match the model file
    base_files = [Path(f).name for f in settings.model_files]
    classes = [int(re.search(r"(?<=CLF\_)\d+(?=\_)", f).group())
               for f in base_files]
    redshifts = [re.search(r"(?<=R\_)[A-Za-z]+(?=\_)", f).group()
                 for f in base_files]

    assert len(set(classes)) == 1, lu.str_to_redstr(
        "Can't provide model files with different number of classes"
    )
    assert len(set(redshifts)) == 1, lu.str_to_redstr(
        "Can't provide model files with different redshifts"
    )

    nb_classes, redshift = classes[0], redshifts[0]
    assert settings.nb_classes == nb_classes, lu.str_to_redstr(
        "Incompatible nb_classes between CLI and model files"
    )
    assert str(settings.redshift) == redshift, lu.str_to_redstr(
        "Incompatible redshift between CLI and model files"
    )

    for model_file in settings.model_files:
        if "variational" in model_file:
            settings.model = "variational"
        if "vanilla" in model_file:
            settings.model = "vanilla"
        if "bayesian" in model_file:
            settings.model = "bayesian"
        rnn = tu.get_model(settings, len(settings.training_features))
        rnn_state = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        rnn.load_state_dict(rnn_state)
        rnn.to(settings.device)
        rnn.eval()
        name = (
            f"{settings.model} photometry"
            if "photometry" in model_file
            else f"{settings.model} salt"
        )
        dict_rnn[name] = rnn

    # load SN info
    SNinfo_df = snn_du.load_HDF5_SNinfo(settings)

    subset_to_plot = []
    for i in range(len(list_data_test)):
        if list_data_test[i][2] == 234413:
            subset_to_plot.append(list_data_test[i])
            if len(subset_to_plot) > 1:
                break
    # # ugly code
    subset_to_plot = []
    # shuffle
    for i in range(len(list_data_test)):
        np.random.shuffle(available_ids)
        if list_data_test[i][2] in available_ids:
            subset_to_plot.append(list_data_test[i])
            if len(subset_to_plot) > nb_lcs or len(subset_to_plot) > len(available_ids):
                break
    for X, target, SNID, _, X_ori in tqdm(subset_to_plot, ncols=100):
        redshift = df[df["SNID"] == SNID]["REDSHIFT_FINAL"].values[0]
        typ = du.spec_type_decoder(df[df["SNID"] == SNID]["TYPE"].values[0])
        trigger_time = df[df["SNID"] == SNID]['PRIVATE(DES_mjd_trigger)'].values[0]
        bazin_time = df[df["SNID"] == SNID]['PKMJDINI'].values[0]
        try:
            fake_time = df[df["SNID"] == SNID]['PRIVATE(DES_fake_peakmjd)'].values[0]
        except Exception:
            fake_time = trigger_time
        title = f" SNTYPE {typ} ID: {SNID}, redshift: {redshift:.3g}, t={int(trigger_time)}, b={int(bazin_time)}, f={int(fake_time)}"

        # Prepare plotting data in a dict
        d_plot = {
            flt: {"FLUXCAL": [], "FLUXCALERR": [], "MJD": []}
            for flt in settings.list_filters
        }
        with torch.no_grad():
            d_pred, X_normed = get_predictions(
                settings, dict_rnn, X, target, OOD=None
            )
        # X here has been normalized. We unnormalize X
        try:
            second = X_normed.shape[1]
        except Exception:
            second = None
        if second:
            X_unnormed = tu.unnormalize_arr(X_normed, settings)
        else:
            continue
        # Check we do recover X_ori when OOD is None
        # check if normalization converges
        # using clipping in case of min<model_min
        X_clip = X_ori.copy()
        X_clip = np.clip(
            X_clip[:, settings.idx_features_to_normalize], settings.arr_norm[:, 0], np.inf)
        X_ori[:, settings.idx_features_to_normalize] = X_clip
        assert np.all(
            np.all(np.isclose(np.ravel(X_ori), np.ravel(X_unnormed), atol=1e-2)))

        # TODO: IMPROVE
        df_temp = pd.DataFrame(data=X_unnormed, columns=features)
        arr_time = np.cumsum(df_temp.delta_time.values)
        df_temp['time'] = arr_time
        # restricting plots to points with smallish errors
        # these points are given to the RNN, this is only visualization
        df_temp = df_temp[(df_temp[f"FLUXCALERR_i"] < 100) & (df_temp[f"FLUXCALERR_g"] < 100) & (df_temp[f"FLUXCALERR_r"] < 100) & (df_temp[f"FLUXCALERR_z"] < 100)]
        for flt in settings.list_filters:
            non_zero = np.where(
                ~np.isclose(df_temp[f"FLUXCAL_{flt}"].values, 0, atol=1E-2)
            )[0]
            d_plot[flt]["FLUXCAL"] = df_temp[f"FLUXCAL_{flt}"].values[non_zero]
            d_plot[flt]["FLUXCALERR"] = df_temp[f"FLUXCALERR_{flt}"].values[
                non_zero
            ]
            d_plot[flt]["MJD"] = arr_time[non_zero]
        plot_predictions(
            settings,
            d_plot,
            SNID,
            redshift,
            target,
            arr_time,
            d_pred,
            None,
            prefix=prefix,
            title=title,
            out_dir=out_dir
        )


    lu.print_green("Finished plotting lightcurves and predictions ")


def plot_hist(df, var, nameout, log=False):
    df = df.fillna(0)
    fig = plt.figure()
    plt.hist(df[var], histtype='step')
    plt.xlabel(var)
    if log:
        plt.yscale("log")
    plt.savefig(nameout)


def plot_superimposed_hist(df_dic, var, nameout=None, log=False):
    fig = plt.figure()
    bins = 10
    for k in df_dic.keys():
        df_dic[k] = df_dic[k].fillna(0)
        n, bins, pathes = plt.hist(
            df_dic[k][var], label=k, histtype='step', bins=bins)
    plt.xlabel(var)
    if log:
        plt.yscale("log")
    plt.legend()
    plt.savefig(nameout)
    plt.close()
    del fig


def plot_venn(df1, df2, colname, nameout='venn.png'):
    from matplotlib_venn import venn2
    fig = plt.figure()
    Path(nameout).parent.mkdir(parents=True, exist_ok=True)
    venn2(
        [
            set(df1[colname].values
                ),
            set(
                df2[colname].values
            )
        ], ('bazin', 'trigger'))
    plt.savefig(nameout)
    plt.close()
    del fig

def print_percentages(what,df_sel,df,color='red'):
            if color == 'red':
                lu.print_red(f"{what} ",f"{len(df_sel)} = {round(100*len(df_sel)/len(df),1)}%")
            elif color == 'blue':
                lu.print_blue(f"{what} ",f"{len(df_sel)} = {round(100*len(df_sel)/len(df),1)}%")

def load_predictions_and_info(skim_dir,model_name):
    df_pred_tmp = pd.read_pickle(f"{skim_dir}/models/{model_name}/PRED_{model_name}.pickle")
    # compute predicted target for complete lc classification
    df_pred_tmp["predicted_target"] = (
        df_pred_tmp[[k for k in df_pred_tmp.keys() if "all_class" in k]]
        .idxmax(axis=1)
        .str.strip("all_class")
        .astype(int))

    # add header info
    df_SNinfo = du.fetch_header_info(skim_dir)
    cols_to_merge = ["SNID"] + [
        k for k in df_SNinfo.keys() if k not in df_pred_tmp.keys()]
    df_pred = df_pred_tmp.merge(df_SNinfo.reset_index()[
        cols_to_merge], how="left", on="SNID")

    return  df_pred

def plot_early_classification(skim_dir,prefix=None,df=None,model_files=None,out_dir=None):
    # plot early class for these only
    snn_args = conf.get_args()
    snn_args.dump_dir = skim_dir
    snn_args.model_files = model_files
    settings = conf.get_settings(snn_args)
    model_settings = conf.get_settings_from_dump(
        settings,
        snn_args.model_files[0]
    )
    model_settings.model_files = snn_args.model_files

    make_early_prediction(model_settings, nb_lcs=50,
                              prefix=prefix, df=df,out_dir=out_dir)

def get_time_cut_stats(df_pred,photo_Ia,skim_dir):
    # get stats for time cuts
    fig = plt.figure()
    photo_Ia['fakemjd'] = photo_Ia['PRIVATE(DES_fake_peakmjd)']
    photo_Ia['trigger'] = photo_Ia['PRIVATE(DES_mjd_trigger)']
    photo_Ia['delta_trigger'] = photo_Ia.apply(lambda row: row.trigger- row.fakemjd,axis=1)
    photo_Ia['delta_bazin'] = photo_Ia.apply(lambda row: row.PKMJDINI - row.fakemjd,axis=1)
    df_pred['fakemjd'] = df_pred['PRIVATE(DES_fake_peakmjd)']
    df_pred['trigger'] = df_pred['PRIVATE(DES_mjd_trigger)']
    df_pred['delta_trigger'] = df_pred.apply(lambda row: row.trigger- row.fakemjd,axis=1)
    df_pred['delta_bazin'] = df_pred.apply(lambda row: row.PKMJDINI - row.fakemjd,axis=1)

    if 'trigger' in cut_type:
        print_percentages(f"Overflows trigger(window  max)",df_pred[abs(df_pred['delta_trigger'])>100],df_pred)
        print_percentages(f"                    photo_Ia  ",photo_Ia[abs(photo_Ia['delta_trigger'])>100],photo_Ia)
        print_percentages(f"Overflows trigger(window .5lc)",df_pred[abs(df_pred['delta_trigger'])>50],df_pred)
        print_percentages(f"                    photo_Ia  ",photo_Ia[abs(photo_Ia['delta_trigger'])>50],photo_Ia)
    else:
        # bazin does not crashes but have more outside window
        print_percentages(f"Overflows bazin (window max)  ",df_pred[abs(df_pred['delta_bazin'])>100],df_pred)
        print_percentages(f"                    photo_Ia  ",photo_Ia[abs(photo_Ia['delta_bazin'])>100],photo_Ia)
        print_percentages(f"Overflows bazin (window .5lc) ",df_pred[abs(df_pred['delta_bazin'])>50],df_pred)
        print_percentages(f"                    photo_Ia  ",photo_Ia[abs(photo_Ia['delta_bazin'])>50],photo_Ia)

        if plot:
            plot_early_classification(skim_dir,model_files=model_files,prefix='OF_bazin_window_',df=photo_Ia[abs(photo_Ia['delta_bazin'])>100])
    
    # plot both possible cuts
    plt.hist(photo_Ia['delta_trigger'],
             histtype='step', label='trigger', bins=1000)
    plt.hist(photo_Ia['delta_bazin'],
             histtype='step', label='bazin', bins=1000)
    plt.xlabel('delta_time')
    plt.xlim(-1000, 1000)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{skim_dir}/figures/phot_Ia_delta_time.png")
    del fig

    fig = plt.figure()
    plt.hist(photo_Ia['delta_trigger'],
             histtype='step', label='trigger', bins=1000)
    plt.hist(photo_Ia['delta_bazin'],
             histtype='step', label='bazin', bins=1000)
    plt.xlabel('delta_time')
    plt.xlim(-100, 100)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{skim_dir}/figures/phot_Ia_delta_time_zoom.png")
    del fig

def get_photo_sample_w_spec_tags(df_pred, photo_SNIDs = None):

    # define photo samples
    photo_Ia = {}
    photo_nonIa ={}

    photo_tmp = df_pred[df_pred['predicted_target'] == 0].copy()

    if photo_SNIDs:
        photo_Ia['all'] = photo_tmp[photo_tmp['SNID'].isin(photo_SNIDs)]
    else:
        photo_Ia['all'] = photo_tmp
        photo_SNIDs = photo_tmp.SNID.values

    photo_nonIa['all'] = df_pred[~df_pred['SNID'].isin(photo_SNIDs)].copy()

    photo_Ia['spec_Ia'] = photo_Ia['all'][(photo_Ia['all']['TYPE']==1) | (photo_Ia['all']['TYPE']==101)]
    photo_Ia['spec_nonIa'] = photo_Ia['all'][(photo_Ia['all']['TYPE']!=0) & (photo_Ia['all']['TYPE']!=1) & (photo_Ia['all']['TYPE']!=101)]
    photo_Ia['all_no_spec_nonIa'] = photo_Ia['all'][(photo_Ia['all']['TYPE']==0) | (photo_Ia['all']['TYPE']==1) | (photo_Ia['all']['TYPE']==101)]

    photo_nonIa['spec_Ia'] = photo_nonIa['all'][(photo_nonIa['all']['TYPE']==1) | (photo_nonIa['all']['TYPE']==101)]
    photo_nonIa['spec_nonIa'] = photo_nonIa['all'][(photo_nonIa['all']['TYPE']!=0) & (photo_nonIa['all']['TYPE']!=1) & (photo_nonIa['all']['TYPE']!=101)]


    return photo_Ia, photo_nonIa

def get_stats_for_sample(df_pred,photo_Ia,photo_nonIa, skim_dir,model_files=None, out_dir =None):

    # inspect sample
    path_plots = f"{skim_dir}/figures/"
    Path(path_plots).mkdir(parents=True, exist_ok=True)
    df_dic = {'phot sample': photo_Ia['all'], 'contaminants': photo_Ia['spec_nonIa']}
    for var in [k for k in ['REDSHIFT_FINAL', 'PRIVATE(DES_numepochs_ml)', 'all_class0', 'PRIVATE(DES_cand_type)', 'TYPE', 'PRIVATE(DES_mjd_trigger)', 'PKMJDINI'] if k in photo_Ia['all'].keys()]:
        plot_superimposed_hist(df_dic, var, nameout=f"{path_plots}/photo_Ia_{var}_dist.png", log=True)

    # Stats
    lu.print_green(cut_type)
    print_percentages(f"photo Ias          ", photo_Ia['all'],df_pred,color='blue')
    if dtype=='real':
        lu.print_blue(f"      are spec Ias  ", len(photo_Ia['spec_Ia']))
        lu.print_blue(f'      are spec other', len(photo_Ia['spec_nonIa']))
        lu.print_blue(f'               gals ', len(photo_Ia['spec_nonIa'][photo_Ia['spec_nonIa']['TYPE']==81]))
        lu.print_red(f"missed Ias          ", len(photo_nonIa['spec_Ia']))
    # plot lcs
    if plot:
        plot_early_classification(skim_dir,prefix='photo_Ia_',df=photo_Ia['all'],model_files=model_files,out_dir=out_dir)
        plot_early_classification(skim_dir,prefix='photo_Ia_contamination',df=photo_Ia['spec_nonIa'],model_files=model_files,out_dir=out_dir)
        plot_early_classification(skim_dir,prefix='photo_nonIa_',df=photo_nonIa['all'],model_files=model_files,out_dir=out_dir)
        
        # sample histograms of type
        plot_hist(photo_Ia['all'], 'TYPE', nameout=f"{path_plots}/photo_Ia_hist_type.png", log=True)
        plot_hist(photo_nonIa['all'], 'TYPE', nameout=f"{path_plots}/photo_nonIa_hist_type.png", log=True)

"""
Main
"""
plot = True
model_name = "vanilla_S_0_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C"
model_files = [
    f"../SuperNNova_general/trained_models_mutant/{model_name}/{model_name}.pt"]

out_dir = {'real':'real_v0','fake':'fake_v0'}

df_pred = {}
photo_Ia = {}
photo_nonIa = {}
for dtype in ["real","fake"]:
    print()
    lu.print_blue(f'_____STATS FOR {dtype}_____')
    print()
    for cut_type in ['window_bazin_SNNone', 'window_trigger_SNNone', 'window_bazin_SN3']:

        skim_dir = f"./dumps/{cut_type}/{dtype}"

        df_pred[cut_type] = load_predictions_and_info(skim_dir, model_name)

        # get photo samples
        photo_Ia[cut_type], photo_nonIa[cut_type] = get_photo_sample_w_spec_tags(df_pred[cut_type])

        # stats for sample
        get_stats_for_sample(df_pred[cut_type],photo_Ia[cut_type],photo_nonIa[cut_type], skim_dir,model_files = model_files)

        # time cut study
        if dtype == 'fake':
            get_time_cut_stats(df_pred[cut_type],photo_Ia[cut_type]['all'],skim_dir)

    # get common sample
    lu.print_green("common")
    common_dir = f"./dumps/common/{dtype}"
    Path(skim_dir).mkdir(parents=True, exist_ok=True)
    common_SNIDs = [k for k in photo_Ia['window_bazin_SNNone']['all'].SNID.values if int(
        k) in photo_Ia['window_trigger_SNNone']['all'].SNID.values.astype(int)]
    common_photo_Ia, common_photo_nonIa = get_photo_sample_w_spec_tags(df_pred['window_trigger_SNNone'], photo_SNIDs=common_SNIDs)

    # plot
    plot_venn(photo_Ia['window_bazin_SNNone']['all'], photo_Ia['window_trigger_SNNone']['all'], 'SNID', nameout=f"{out_dir[dtype]}/venn_common_sample.png")

    # stats for sample
    get_stats_for_sample(df_pred['window_trigger_SNNone'],common_photo_Ia,common_photo_nonIa, skim_dir,model_files = model_files, out_dir=common_dir)

    # dump sample
    common_photo_Ia['all_no_spec_nonIa'][[
        'SNID', 'HOSTGAL_OBJID', 'HOSTGAL2_OBJID', 'DEC', 'RA']].to_csv(f'{out_dir[dtype]}/sample.csv')
