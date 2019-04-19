import os
import glob
import shutil
import pandas as pd
from pathlib import Path
from ..utils import data_utils as du
from ..utils import logging_utils as lu
from ..utils import visualization_utils as vu

"""Skimming files for light-curves in a window

Use trigger time to make a window for the lc photometry
"""


def skim_photometry(dtype, raw_dir, dump_dir, debug=False):
    """Skim photometry files

    Eliminating photometric points/events from PHOT, HEAD.FITS

    Arguments:
        dtype {str} -- data type (real, fake)
        raw_dir {str} -- where data is
        dump_dir {str} -- where data should be dumped
        debug {Bool} -- process only one file to test
    """

    df_dic = du.process_photometry(raw_dir, dump_dir, debug=debug, keep_delim=True)
    for i in df_dic.keys():
        try:
            df = df_dic[i][0]
        except Exception:
            df = df_dic[i]
        # apply cuts
        # hack to keep the separators
        mask = (df['MJD'] != -777.00)
        # time window
        # vanilla version
        df['delta_time'] = df['MJD']-df['PRIVATE(DES_mjd_trigger)']
        df['time_window_cut'] = True
        df.loc[mask,'time_window_cut'] = df["delta_time"].apply(lambda x: True if (
            x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False))

        # S/N cut (for limiting magnitudes)
        # loose cut S/N 3
        df['S/N'] = df['FLUXCAL']/df['FLUXCALERR']
        df['S/N_cut'] = True
        df.loc[mask,'S/N_cut'] = df["S/N"].apply(lambda x: True if x>3 else False)

        # cut posible transient_status
        # 0, no season had S/N>5 and good real/bogus score on 2 epochs
        # 1, 2, 4, 8 or 16 (one for each year of the survey)
        # https://cdcvs.fnal.gov/redmine/projects/des-sn/wiki/Transient_Naming
        # AGN should be in all or almost all seasons
        # allowing one season or two consecutive seasons only
        # df['transient_status_cut'] = True
        # too harsh
        df.loc[mask,'transient_status_cut'] = df['PRIVATE(DES_transient_status)'].apply(lambda x: True if int(x) in [1,2,4,8,16,3,6,12,24] else False)
        # df.loc[mask,'transient_status_cut'] = df['PRIVATE(DES_transient_status)'].apply(lambda x: True if int(x) not in [7,11,14,24,21,22,25,26,27,28,19,15] else False)
        df_sel = df[ (df['time_window_cut'] == True) & (df['S/N_cut'] == True ) & (df['transient_status_cut'] == True )]
        # df_sel = df[(df['time_window_cut'] == True) & (df['S/N_cut'] == True )]
        
        df_sel = df_sel.reset_index()
        # need to save only events that have points now
        # photometry
        keep_col_phot = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
        df_phot = df_sel[keep_col_phot]
        df_phot = df_phot.loc[df_phot["FLUXCAL"].shift() != df_phot["FLUXCAL"]]
        du.save_fits(df_phot, f"{dump_dir}/DESALL_forcePhoto_{dtype}_snana_fits/DESY{i+1}_skimmed_{dtype}_PHOT.FITS")
        print('photometry',len(df_phot))

        # header
        keep_col_head = [
            "SNID",
            "PEAKMJD",
            "HOSTGAL_PHOTOZ",
            "HOSTGAL_PHOTOZ_ERR",
            "HOSTGAL_SPECZ",
            "HOSTGAL_SPECZ_ERR",
            "SNTYPE",
        ]
        keep_col_head = keep_col_head + \
            [k for k in df.keys() if 'mjd' in k or 'fake' in k or "PRIVATE(DES_transient_status)" in k]

        df_head = df_sel[df_sel.MJD != -777.000]
        df_head = df_head[keep_col_head]
        df_head = df_head.drop_duplicates()
        df_head = df_head.reset_index()
        df_head = df_head[keep_col_head]
        du.save_fits(df_head, f"{dump_dir}/DESALL_forcePhoto_{dtype}_snana_fits/DESY{i+1}_skimmed_{dtype}_HEAD.FITS")
        lu.print_blue(f"Skimmed files saved {dump_dir}/DESALL_forcePhoto_{dtype}_snana_fits/DESY{i+1}_skimmed_{dtype}_*.FITS")

def skim_fits(raw_dir,dump_dir,debug):

    # reset whatever was processed before
    if Path(dump_dir).exists():
        shutil.rmtree(Path(dump_dir))

    # Porcess both real and fake light-curves
    for dtype in ['real', 'fake']:
        skim_photometry(dtype, f"{raw_dir}/DESALL_forcePhoto_{dtype}_snana_fits/", dump_dir, debug=args.debug)
