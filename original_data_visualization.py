import glob,os
import argparse
import pandas as pd
from utils import data_utils as du
from utils import visualization_utils as vu

"""
Visualize force photometry light-curves
directly from Midway, no preprocessing/skimming
"""

def plot_lcs(df_real, df_fake,dump_dir):
    """Plot real and fake light-curves
    """
    path_plots = f'{dump_dir}/real/'
    vu.plot_random_lcs(df_real,path_plots)
    path_plots = path_plots.replace('real',"fake")
    vu.plot_random_lcs(df_fake,path_plots) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SNIa classification")

    parser.add_argument("--raw_dir", type=str, default='../DES/data/',
                        help="Directory with raw data")

    parser.add_argument("--dump_dir", type=str, default='./dumps/yearly_photometry/',
                        help="Directory for output data")

    parser.add_argument("--debug", action="store_true", 
                        help="debug: lcs from one year only")

    args = parser.parse_args()

    raw_dir = args.raw_dir
    dump_dir = args.dump_dir

    # load data and process if necessary
    df = {}
    for dtype in ["real","fake"]:
        list_files = glob.glob(f"{raw_dir}/DESALL_forcePhoto_{dtype}_snana_fits/*PHOT.FITS")
        if args.debug:
            list_files = list_files[:1]
        for fname in list_files:
            df_header, df_phot = du.read_fits(fname)
            df[dtype] = df_header.merge(df_phot,on="SNID")

    # plot lcs
    plot_lcs(df["real"], df["fake"],dump_dir)

    # statistical inspection
    vu.inspect_peak(df["real"], df["fake"],dump_dir)

