import argparse
import os

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
        "--time_var",
        type=str,
        default="clump",
        choices=["clump","trigger", "bazin"],
        help="Computed peak method for time cut",
    )
    parser.add_argument(
        "--done_file",
        type=str,
        default="done_file.txt",
        help="Location of the done file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug: process one file only"
    )
    args = parser.parse_args()

    done_file = args.done_file
    if not done_file.startswith("/"):
        done_file = os.path.join(args.dump_dir, args.done_file)

    try:
        cu.skim_data(args.raw_dir, args.dump_dir, args.fits_file,args.time_var,debug=args.debug)
    except Exception as e:
        with open(done_file, "w") as f:
            f.write("FAILURE")
        raise e
    else:
        with open(done_file, "w") as f:
            f.write("SUCCESS")

