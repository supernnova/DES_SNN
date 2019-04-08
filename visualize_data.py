import glob,os
import argparse
import pandas as pd
import utils.data_utils as du
import utils.visualization_utils as vu


def plot_lcs(df_real, df_fake,dump_dir):
    """Plot real and fake light-curves
    """
    path_plots = f'{dump_dir}/lc_visualization/real/'
    vu.plot_random_lcs(df_real,path_plots)
    path_plots = path_plots.replace('real',"fake")
    vu.plot_random_lcs(df_fake,path_plots) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SNIa classification")

    parser.add_argument("--raw_dir", type=str, default='../DES/data/',
                        help="Directory with raw data")

    parser.add_argument("--dump_dir", type=str, default='./dump_visualize/',
                        help="Directory for output data")

    parser.add_argument("--debug", action="store_true", 
                        help="debug mode")

    args = parser.parse_args()

    raw_dir = args.raw_dir
    dump_dir = args.dump_dir

    # load data and process if necessary
    df = {}
    for dtype in ["real","fake"]:
        df[dtype] = du.load_data(f"{raw_dir}/DESALL_forcePhoto_{dtype}_snana_fits/",dump_dir,debug=args.debug, redo_photometry = True)

    # plot lcs
    plot_lcs(df["real"], df["fake"],dump_dir)

    # statistical inspection
    vu.inspect_peak(df["real"], df["fake"],dump_dir)

