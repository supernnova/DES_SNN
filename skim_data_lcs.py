import os
import glob
import argparse
from astropy.table import Table

def read_fits(fname):
    # remember FITS table are MUCH more than their data
    # there are headers, etc there!

    # load photometry
    dat = Table.read(fname, format='fits')
    df_phot = dat.to_pandas()
    # failsafe
    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    # load header
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_head = header.to_pandas()
    df_head["SNID"] = df_head["SNID"].astype(np.int32)

    # add SNID to phot for skimming
    arr_ID = np.zeros(len(df_phot), dtype=np.int32)
    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(df_phot["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df_phot)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_head.SNID.iloc[counter - 1]
    df_phot["SNID"] = arr_ID

    return df_head, df_phot

def compute_time_cut(df_head, df_phot,timevar_to_cut=None):
    # Time cut
    lu.print_green(f"Compute time cut {timevar_to_cut}")

    df_phot['time_cut'] = True
    df_info_for_skim = df_head[["SNID", timevar_to_cut]]
    df_phot = pd.merge(df_phot, df_info_for_skim, on="SNID", how="left")
    mask = (df_phot['MJD'] != -777.00)
    df_phot['delta_time'] = df_phot['MJD'] - df_phot[timevar_to_cut]
    df_phot['time_cut'] = True
    df_phot.loc[mask, 'time_cut'] = df_phot["delta_time"].apply(
        lambda x: True if (x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False)
    )
    df_phot = df_phot[df_phot['time_cut'] == True]

    ids_to_keep = df_phot["SNID"].unique()
    df_head = df_head[df_head["SNID"].isin(ids_to_keep.tolist())]

    return df_head, df_phot

def decode_time_var(time_var):
    dic_time_var = {'trigger':'PRIVATE(DES_mjd_trigger)', 'bazin':'PKMJDINI', 'clump':'something'}
    try: 
        timevar_to_cut = dic_time_var[time_var]
    else:
        timevar_to_cut = None
    return timevar_to_cut

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
        help="Default path to dump window list",
    )
    parser.add_argument(
        "--time_var",
        type=str,
        default="clump",
        choices=["clump","trigger", "bazin"],
        help="Variable to use for time cut",
    )
    parser.add_argument(
        "--done_file",
        type=str,
        default="done_file.txt",
        help="Location of the done file"
    )
    parser.add_argument(
        "--debug",
        action=store_true,
        default="done_file.txt",
        help="Location of the done file"
    )
    args = parser.parse_args()

    done_file = args.done_file
    if not done_file.startswith("/"):
        done_file = os.path.join(args.dump_dir, args.done_file)
    try:
        # load fits file
        if Path(args.fits_file).exists():
            df_fits = pd.read_csv(fits_file, comment="#",
                          delimiter=" ", skipinitialspace=True)
            df_fits["SNID"] = df_fits["CID"].astype(int)
        else:
            df_fits = None
            lu.print_yellow("No fits file found")

    # list phot files
    list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))
    if debug:
        lu.print_yellow('Debugging mode')
        list_files = list_files[:1]
    lu.print_green(f"Starting data skimming, found {len(list_files)} to operate on")
    # process each phot file
    for fname in list_files:
        df_head, df_phot = read_fits(fname)
        head_keys = df_head.keys().tolist()
        df_head = pd.merge(df_head, df_fits, on='SNID')
        # get window cuts
        timevar_to_cut = decode_time_var(time_var)
        # need to work on this so outputs a list
        df_cuts = compute_time_cut(df_head, df_phot,timevar_to_cut=timevar_to_cut)
        # dump list
        df_cuts.to_csv(outname)

        # dump sntypes in the datafiles
    tmp = list(set(tmp_type_list))
    type_list = [(k, du.spec_type_decoder(k)) for k in tmp]
    with open(f'{dump_dir}/sntypes.json', 'w') as outfile:
        json.dump(type_list, outfile)

    except Exception as e:
        with open(done_file, "w") as f:
            f.write("FAILURE")
        raise e
    else:
        with open(done_file, "w") as f:
            f.write("SUCCESS")

