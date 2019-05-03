import os
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu

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
    if timevar=='trigger': 
        timevar_to_cut='PRIVATE(DES_mjd_trigger)'
    elif timevar=='bazin': 
        timevar_to_cut='PKMJDINI'
    else: timevar_to_cut= None

    df_header, df_phot = compute_time_cut(df_header,df_phot, time_cut_type = time_cut_type, timevar_to_cut = timevar_to_cut)

    compute_S_N_cut(df_header,df_phot, SN_threshold=None)

    du.save_phot_fits(df_phot,f'{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}/{dump_prefix}_PHOT.FITS')
    du.save_fits(df_header,f'{dump_dir}/{time_cut_type}_{timevar}_SN{SN_threshold}/{dump_prefix}_HEAD.FITS')

