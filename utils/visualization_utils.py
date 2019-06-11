import os
import re
import h5py
import glob
import torch
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from sklearn import metrics
import utils.data_utils as du
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import SuperNNova.supernnova.conf as conf
from SuperNNova.supernnova.utils import training_utils as tu
from SuperNNova.supernnova.utils import logging_utils as lu
from SuperNNova.supernnova.utils import data_utils as snn_du
from SuperNNova.supernnova.utils.visualization_utils import FILTER_COLORS, ALL_COLORS, LINE_STYLE

def inspect_peak(df_real, df_fake, dump_dir, debug=False):
    path_plots = f"{dump_dir}/inspect_peak/"
    os.makedirs(path_plots, exist_ok=True)

    # get attributes by light-curve
    # peaks shouldnt change
    df_real = df_real.groupby('SNID').mean()
    df_fake = df_fake.groupby('SNID').mean()

    # selection
    df_real = df_real[(df_real['PRIVATE(DES_mjd_trigger)']
                       > 0) & (df_real['PEAKMJD'] > 0)]
    df_fake = df_fake[(df_fake['PRIVATE(DES_mjd_trigger)']
                       > 0) & (df_fake['PEAKMJD'] > 0)]

    # SCATTER
    fig = plt.figure()
    plt.scatter(df_real['PRIVATE(DES_mjd_trigger)'],
                df_real['PEAKMJD'], color='blue',label='real')
    plt.scatter(df_fake['PRIVATE(DES_mjd_trigger)'],
                df_fake['PEAKMJD'], color='orange',label='fake')
    plt.legend()
    plt.savefig(f"{path_plots}/scatter_peak_snid_trigger.png")
    del fig

    fig = plt.figure()
    plt.hist(df_fake['PRIVATE(DES_mjd_trigger)']-df_fake['PEAKMJD'],
             color='blue', histtype="step", label='trigger-psnid')
    plt.hist(df_fake['PRIVATE(DES_fake_peakmjd)']-df_fake['PEAKMJD'],
             color='orange', histtype="step", label='sim-psnid')
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"{path_plots}/hist_fake_peak.png")
    del fig


def plot_single_lc(df, sid, ax, plot_peak=True, no_title=False):

    SN = df[df['SNID'] == sid]
    for flt in df['FLT'].unique():
        SN_flt = SN[SN['FLT'] == flt]
        if len(SN_flt) > 1:
            min_time = SN_flt['MJD'].min()
            plt.errorbar(SN_flt['MJD'], SN_flt['FLUXCAL'],
                         yerr=SN_flt['FLUXCALERR'].values, label=flt, fmt='o')
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.xlabel('MJD')
            plt.ylabel('FLUXCAL')
            ax.set_ylim(SN_flt['FLUXCAL'].min(), SN_flt['FLUXCAL'].max())
    if plot_peak:
        if SN["PEAKMJD"].iloc[0] + 5 > SN['MJD'].min():
            ax.plot(
                [SN["PEAKMJD"].iloc[0], SN["PEAKMJD"].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="grey", linestyle="--", label="PSNID peak"
            )
        if SN['PRIVATE(DES_mjd_trigger)'].iloc[0] + 5 > SN['MJD'].min():
            ax.plot(
                [SN['PRIVATE(DES_mjd_trigger)'].iloc[0], SN['PRIVATE(DES_mjd_trigger)'].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="orange", linestyle="--", label="trigger"
            )
        if 'PRIVATE(DES_fake_peakmjd)' in SN.keys():
            ax.plot(
                [SN['PRIVATE(DES_fake_peakmjd)'].iloc[0], SN['PRIVATE(DES_fake_peakmjd)'].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="black", linestyle="--", label="sim"
            )
        if 'PKMJDINI' in SN.keys():
            ax.plot(
                [SN['PKMJDINI'].iloc[0], SN['PKMJDINI'].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="blue", linestyle="--", label="bazin"
            )
        plt.legend()
    try:
        z = str(round(SN['PRIVATE(DES_fake_z)'].iloc[0], 1))
        mag = str(round(SN['SIM_MAGOBS'].iloc[0], 1))

    except Exception:
        z = "None"
        mag = "None"

    if z == "None":
        try:
            z = str(round(SN['HOSTGAL_SPECZ'].iloc[0], 1))
        except Exception:
            a = 0
    if not no_title:
        ax.set_title(f"ID:{sid}, z:{z}, mag:{mag}")

    return ax


def plot_random_lcs(df, path_plots, multiplots=False, nb_lcs=20, plot_peak=True):
    lu.print_green("Plot light-curves")
    # clean directory
    if Path(path_plots).exists():
        shutil.rmtree(path_plots)
    os.makedirs(path_plots, exist_ok=True)
    # randoms Ias
    list_SNIDs = [
        df.iloc[i]['SNID'] for i in sorted(random.sample(range(len(df)), nb_lcs))
    ]
    if multiplots:
        fig = plt.figure()
    for i, sid in enumerate(list_SNIDs):
        if multiplots:
            ax = plt.subplot(3, 3, i+1)
            ax.set_title(f"SNID {sid}")
        else:
            fig, ax = plt.subplots()
        # plot function
        ax = plot_single_lc(df, sid, ax, plot_peak=plot_peak)
        if not multiplots:
            # Tight layout often produces nice results
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.savefig(f"{path_plots}lc_{sid}.png")
    if multiplots:
        # Tight layout often produces nice results
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(f"{path_plots}lc_{sid}.png")

    del fig


def hist_delta_var(df_header, time_cut_type, timevar, dump_dir, dump_prefix, cut_version):

    if timevar == 'trigger':
        timevar_to_cut = 'PRIVATE(DES_mjd_trigger)'
    elif timevar == 'bazin':
        timevar_to_cut = 'PKMJDINI'
    else:
        timevar_to_cut = None

    path_plots = f"{dump_dir}/{cut_version}/{Path(dump_prefix).parent}/figures/"
    os.makedirs(path_plots, exist_ok=True)
    to_plot = df_header[timevar_to_cut]-df_header["PRIVATE(DES_fake_peakmjd)"]
    to_plot = to_plot[abs(to_plot) < 100]
    fig = plt.figure()
    plt.hist(to_plot, histtype="step", label=f"mean {round(to_plot.mean(),1)}")
    plt.plot(
        [to_plot.mean(), to_plot.mean()], [plt.ylim()[0], plt.ylim()[1]], color="black", linestyle="--", label="mean"
    )
    plt.plot(
        [to_plot.std(), to_plot.std()], [plt.ylim()[0], plt.ylim()[1]], color="grey", linestyle="--", label="std"
    )
    plt.xlabel(f"{timevar_to_cut}-fake_peak")
    plt.legend()
    plt.savefig(f"{path_plots}/{Path(dump_prefix).name}_hist_delta_var.png")


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


def plot_early_classification(skim_dir, prefix=None, df=None, model_files=None, out_dir=None):
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
                          prefix=prefix, df=df, out_dir=out_dir)


def plot_hist(df, var, nameout, log=False):
    df = df.fillna(0)
    fig = plt.figure()
    plt.hist(df[var], histtype='step')
    plt.xlabel(var)
    if log:
        plt.yscale("log")
    plt.savefig(nameout)


def plot_superimposed_hist(df_dic, var, nameout=None, log=False, limits_from_photo_sample=False, only_positive_x=False,bins=50):
    fig = plt.figure()
    if limits_from_photo_sample:
        keys_to_plot = ['photo Ia sample'] + \
            [k for k in df_dic.keys() if 'photo Ia sample' not in k]
    else:
        keys_to_plot = df_dic.keys()

    for k in keys_to_plot:
        df_dic[k] = df_dic[k].fillna(0)
        n, bins, pathes = plt.hist(
            df_dic[k][var], label=f"{k} mean:{round(df_dic[k][var].mean(),1)}+-{round(df_dic[k][var].std(),1)}", histtype='step', bins=bins)
    plt.xlabel(var)
    if log:
        plt.yscale("log")
    if limits_from_photo_sample:
        plt.xlim(df_dic['photo Ia sample'][var].min(),
                 df_dic['photo Ia sample'][var].max())
    elif only_positive_x:
        plt.xlim(0,)
    plt.legend()
    plt.savefig(nameout)
    plt.close()
    del fig


def make_early_prediction(settings, nb_lcs=1, do_gifs=False, df=None, prefix=None, out_dir=None):
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
        try:
            trigger_time = df[df["SNID"] ==
                              SNID]['PRIVATE(DES_mjd_trigger)'].values[0]
            bazin_time = df[df["SNID"] == SNID]['PKMJDINI'].values[0]
        except Exception:
            trigger_time = 0
            bazin_time = 0 
        try:
            fake_time = df[df["SNID"] ==
                           SNID]['PRIVATE(DES_fake_peakmjd)'].values[0]
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


def plot_predictions(
    settings, d_plot, SNID, redshift, target, arr_time, d_pred, OOD, prefix=None, title=None, out_dir=None
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
