import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import utils.data_utils as du
import matplotlib.pyplot as plt
from astropy.table import Table
import utils.logging_utils as lu
from matplotlib import pyplot as plt


def inspect_peak(df_real, df_fake, dump_dir, debug=False):
    path_plots = f"{dump_dir}/peak/"
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
                df_real['PEAKMJD'], color='blue')
    plt.scatter(df_fake['PRIVATE(DES_mjd_trigger)'],
                df_fake['PEAKMJD'], color='orange')
    plt.savefig(f"{path_plots}/scatter_peak_snid_trigger.png")
    del fig

    fig = plt.figure()
    plt.hist(df_fake['PRIVATE(DES_mjd_trigger)']-df_fake['PEAKMJD'],
             color='blue', histtype="step", label='trigger-psnid')
    plt.hist(df_fake['PRIVATE(DES_fake_peakmjd)']-df_fake['PEAKMJD'],
             color='orange', histtype="step", label='sim-psnid')
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
        # just in case peak predictions are off
        # if SN["PEAKMJD"].iloc[0] + 5 > SN['MJD'].min():
        #     ax.plot(
        #         [SN["PEAKMJD"].iloc[0], SN["PEAKMJD"].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="grey", linestyle="--", label="PSNID peak"
        #     )
        # if SN['PRIVATE(DES_mjd_trigger)'].iloc[0] + 5 > SN['MJD'].min():
        #     ax.plot(
        #         [SN['PRIVATE(DES_mjd_trigger)'].iloc[0], SN['PRIVATE(DES_mjd_trigger)'].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="orange", linestyle="--", label="trigger"
        #     )
        # if 'PRIVATE(DES_fake_peakmjd)' in SN.keys():
        #     ax.plot(
        #         [SN['PRIVATE(DES_fake_peakmjd)'].iloc[0], SN['PRIVATE(DES_fake_peakmjd)'].iloc[0]], [plt.ylim()[0], plt.ylim()[1]], color="black", linestyle="--", label="sim"
        #     )
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
            a =0
    if not no_title:
        ax.set_title(f"ID:{sid}, z:{z}, mag:{mag}")

    return ax


def plot_random_lcs(df, path_plots, multiplots=False, nb_lcs=20,plot_peak=True):
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
        ax = plot_single_lc(df, sid, ax,plot_peak=plot_peak)
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

def hist_delta_var(df_header, time_cut_type,timevar,dump_dir,dump_prefix,cut_version):

    if timevar=='trigger': 
        timevar_to_cut='PRIVATE(DES_mjd_trigger)'
    elif timevar=='bazin': 
        timevar_to_cut='PKMJDINI'
    else: timevar_to_cut= None

    path_plots= f"{dump_dir}/{cut_version}/{Path(dump_prefix).parent}/figures/"
    os.makedirs(path_plots, exist_ok=True)
    to_plot = df_header[timevar_to_cut]-df_header["PRIVATE(DES_fake_peakmjd)"]
    to_plot = to_plot[abs(to_plot)<100]
    fig = plt.figure()
    plt.hist(to_plot, histtype="step",label=f"mean {round(to_plot.mean(),1)}")
    plt.plot(
              [to_plot.mean(),to_plot.mean()], [plt.ylim()[0], plt.ylim()[1]], color="black", linestyle="--", label="mean"
            )
    plt.plot(
              [to_plot.std(),to_plot.std()], [plt.ylim()[0], plt.ylim()[1]], color="grey", linestyle="--", label="std"
            )
    plt.xlabel(f"{timevar_to_cut}-fake_peak")
    plt.legend()
    plt.savefig(f"{path_plots}/{Path(dump_prefix).name}_hist_delta_var.png")

