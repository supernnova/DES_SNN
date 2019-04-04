import glob,os
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import matplotlib.pyplot as plt
from astropy.table import Table
import utils.logging_utils as lu
from matplotlib import pyplot as plt


path_survey_data = '../DES/data/DESALL_forcePhoto_real_snana_fits/'
path_plots = './dump/lc_visualization/'
os.makedirs(path_plots,exist_ok=True)
path_preprocessed = './dump/preprocessed/'

flt_list=['g','r','i','z']


def plot_random_lcs(df,path_plots, multiplots=False):
    # clean directory
    shutil.rmtree(path_plots)
    os.makedirs(path_plots,exist_ok=True)
    #randoms Ias
    sample_size = 20
    list_SNIDs = [
        df.iloc[i]['SNID'] for i in sorted(random.sample(range(len(df)), sample_size))
    ]
    if multiplots:
        fig = plt.figure()
    for i, sid in enumerate(list_SNIDs):
        SN = df[df['SNID']==sid]
        if multiplots:
            fig, ax=plt.subplot(3,3,i+1)
            ax.set_title(f"SNID {sid}")
        else:
            fig, ax = plt.subplots()
        for flt in flt_list:
            SN_flt= SN[SN['FLT']==flt]
            min_time=SN_flt['MJD'].min()
            plt.errorbar(SN_flt['MJD'],SN_flt['FLUXCAL'],yerr=SN_flt['FLUXCALERR'].values,label=flt,fmt='o')
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.xlabel('MJD')
            plt.ylabel('FLUXCAL')
            ax.set_ylim(SN_flt['FLUXCAL'].min(), SN_flt['FLUXCAL'].max())
        if SN["PEAKMJD"].iloc[0] + 5 > SN['MJD'].min():
            ax.plot(
                    [SN["PEAKMJD"].iloc[0], SN["PEAKMJD"].iloc[0] ], [plt.ylim()[0], plt.ylim()[1]], color="k", linestyle="--"
                )
        if SN['PRIVATE(DES_mjd_trigger)'].iloc[0]+ 5> SN['MJD'].min():
            ax.plot(
                    [SN['PRIVATE(DES_mjd_trigger)'].iloc[0], SN['PRIVATE(DES_mjd_trigger)'].iloc[0] ], [plt.ylim()[0], plt.ylim()[1]], color="orange", linestyle="--"
                )
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

if __name__ == '__main__':
    # Load data
    fname_preprocessed = f"{path_preprocessed}/preprocessed_phot.pickle"
    if Path(fname_preprocessed).is_file():
        lu.print_green("Loading preprocessed data")
        df = pd.read_pickle(fname_preprocessed)
    else:
        lu.print_green("Processing data FITS")
        # Get files
        list_files = glob.glob(os.path.join(path_survey_data, "*PHOT.FITS"))
        os.makedirs(path_preprocessed,exist_ok=True)
        df = du.process_photometry(list_files,debug=False,snn_like_cols=False)
        df.to_pickle(fname_preprocessed)
    # Plot random lightcurves
    lu.print_green("Plot light-curves")
    plot_random_lcs(df,path_plots)
