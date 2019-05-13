import os, glob
import argparse
from pathlib import Path
import utils.cuts_utils as cu

if __name__ == '__main__':
    """ Stand alone code that skims data

    Skimming using time window and S/N cut (optional)
    """

    parser = argparse.ArgumentParser(description="Skimming DES data for SuperNNova")

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=f"./raw/",
        help="Default path where raw data is",
    )
    parser.add_argument(
        "--fits_file",
        type=str,
        default=f"./fits/",
        help="Default path where Bazin function fit to data is",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default="./dumps/",
        help="Default path where skimmed data is dumped",
    )

    parser.add_argument(
        "--time_cut_type",
        type=str,
        default="window",
        choices=["window"],
        help="Type of time cut, default window",
    )
    parser.add_argument(
        "--time_var",
        type=str,
        default="trigger",
        choices=["trigger","bazin"],
        help="Variable to use for time cut",
    )
    parser.add_argument(
        "--SN_threshold",
        default=None,
        choices=[None,3],
        help="S/N threshold to use, default None",
    )

    args = parser.parse_args()
    cu.skim_data(args.raw_dir,args.dump_dir, args.fits_file, args.time_cut_type,args.time_var,args.SN_threshold)