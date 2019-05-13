import os
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu
import utils.visualization_utils as vu

def compute_time_cut(df_header,df_phot, time_cut_type = None, timevar_to_cut = None):
    # Time cut
    lu.print_green(f"Compute time cut {time_cut_type} with {timevar_to_cut}")
    df_phot['time_cut'] = True
    if time_cut_type == 'window':
        df_info_for_skim = df_header[["SNID",timevar_to_cut]]
        df_phot = pd.merge(df_phot, df_info_for_skim, on="SNID", how="left")
        mask = (df_phot['MJD'] != -777.00)
        df_phot['delta_time'] = df_phot['MJD']-df_phot[timevar_to_cut]
        df_phot['time_cut'] = True
        df_phot.loc[mask, 'time_cut'] = df_phot["delta_time"].apply(lambda x: True if (
            x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False))
        df_phot = df_phot[df_phot['time_cut'] == True]

        ids_to_keep = df_phot["SNID"].unique()
        df_header = df_header[df_header["SNID"].isin(ids_to_keep.tolist())]

    return df_header,df_phot

def compute_S_N_cut(df_header,df_phot, SN_threshold=None):
    # S/N cut (for limiting magnitudes)
    df_phot['S/N'] = df_phot['FLUXCAL']/df_phot['FLUXCALERR']
    df_phot['S/N_cut'] = True
    mask = (df_phot['MJD'] != -777.00)
    if SN_threshold:
        lu.print_green(f"Compute S/N cut {SN_threshold}")
        df_phot.loc[mask,
               'S/N_cut'] = df_phot["S/N"].apply(lambda x: True if x > 3 else False)
        
        df_phot = df_phot[df_phot['S/N_cut'] == True]

        ids_to_keep = df_phot["SNID"].unique()
        df_header = df_header[df_header["SNID"].isin(ids_to_keep.tolist())]

    return df_header,df_phot

def apply_cut_save(df_header,df_phot, time_cut_type = None, timevar = None ,SN_threshold=None, dump_dir=None,dump_prefix = None):
    # init
    if timevar=='trigger': 
        timevar_to_cut='PRIVATE(DES_mjd_trigger)'
    elif timevar=='bazin': 
        timevar_to_cut='PKMJDINI'
    else: timevar_to_cut= None
    cut_version = f"{time_cut_type}_{timevar}_SN{SN_threshold}"

    # apply cuts
    df_header, df_phot = compute_time_cut(df_header,df_phot, time_cut_type = time_cut_type, timevar_to_cut = timevar_to_cut)
    df_header, df_phot = compute_S_N_cut(df_header,df_phot, SN_threshold=None)

    # format sntypes as sim
    if 'fake' in dump_prefix:
        df_header["SNTYPE"] = df_header["SNTYPE"].apply(lambda x: 1 if x==0 else 0)
    else:
        # need to add spec
        df_header["SNTYPE"] = df_header["SNTYPE"].apply(lambda x: 1 if x==1 else 0)

    # save
    df_phot_saved = du.save_phot_fits(df_phot,f'{dump_dir}/{cut_version}/{dump_prefix}_PHOT.FITS')
    df_phot_saved = df_phot_saved[df_phot_saved['SNID']!=0]
    #in order to keep same ordering
    df_phot_for_header = df_phot_saved.loc[df_phot_saved["SNID"].shift() != df_phot_saved["SNID"]]
    df_phot_for_header = df_phot_for_header.reset_index()
    df_header_tosave = df_phot_for_header[['SNID']].merge(df_header,on='SNID')
    du.save_fits(df_header_tosave,f'{dump_dir}/{cut_version}/{dump_prefix}_HEAD.FITS')

    # if fake do histogram with delta_t
    if "PRIVATE(DES_fake_peakmjd)" in df_header.keys():
        vu.hist_delta_var(df_header, time_cut_type,timevar,dump_dir,dump_prefix,cut_version)
    #plot lcs for control
    path_plots = f'{dump_dir}/{cut_version}/{Path(dump_prefix).parent}/skimmed_lightcurves/'
    vu.plot_random_lcs(df_phot, path_plots, multiplots=False, nb_lcs=20,plot_peak=False)



