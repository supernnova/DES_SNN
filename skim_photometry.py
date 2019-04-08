import os
import glob
import shutil
import argparse
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu
import utils.visualization_utils as vu

"""Skimming files for light-curves in a window

Use trigger time to make a window for the lc photometry
"""


def skim_photometry(dtype, raw_dir, dump_dir, debug=False):
    """Skim photometry files

    Eliminating photoemtric points/events from PHOT, HEAD.FITS

    Arguments:
        dtype {str} -- data type (real, fake)
        raw_dir {str} -- where data is
        dump_dir {str} -- where data should be dumped
        debug {Bool} -- process only one file to test
    """

    df_dic = du.process_photometry(
        raw_dir, dump_dir, debug=debug, keep_delim=True)
    for i in df_dic.keys():
        df = df_dic[i][0]
        # apply cuts
        df['delta_time'] = df['MJD']-df['PRIVATE(DES_mjd_trigger)']
        # hack to keep the separators
        mask = (df['MJD'] != -777.00)
        df_valid = df[mask]
        df['selection'] = True
        df.loc[mask,'selection'] = df["delta_time"].apply(lambda x: True if (
            x > 0 and x < 70) else (True if (x <= 0 and x > -20) else False))
        df_sel = df[df['selection'] == True]
        df_sel = df_sel.reset_index()

        # need to save only events that have points now
        keep_col_phot = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
        df_phot = df_sel[keep_col_phot]
        du.save_fits(df_phot, f"{dump_dir}/DESALL_forcePhoto_{dtype}_snana_fits/DESY{i+1}_skimmed_{dtype}_PHOT.FITS")

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
            [k for k in df.keys() if 'mjd' in k or 'fake' in k]
        df_head = df_sel[df_sel.MJD != -777.000]
        df_head = df_sel[keep_col_head]
        df_head = df_head.drop_duplicates()
        df_head = df_head.reset_index()
        du.save_fits(df_head, f"{dump_dir}/DESALL_forcePhoto_{dtype}_snana_fits/DESY{i+1}_skimmed_{dtype}_HEAD.FITS")
        lu.print_blue(f"Skimmed files saved {dump_dir}/DESALL_forcePhoto_{dtype}_snana_fits/DESY{i+1}_skimmed_{dtype}_*.FITS")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Skimming photometry files (lc window)")

    parser.add_argument("--raw_dir", type=str, default='../DES/data/',
                        help="Directory with raw data")

    parser.add_argument("--dump_dir", type=str, default='./dump_skimming/',
                        help="Directory for processed data")

    parser.add_argument("--debug", action="store_true",
                        help="debug mode")

    args = parser.parse_args()

    raw_dir = args.raw_dir
    dump_dir = args.dump_dir

    # reset whatever was processed before
    if Path(dump_dir).exists():
        shutil.rmtree(Path(dump_dir))

    for dtype in ['real', 'fake']:
        skim_photometry(dtype, f"{raw_dir}/DESALL_forcePhoto_{dtype}_snana_fits/", dump_dir, debug=args.debug)
