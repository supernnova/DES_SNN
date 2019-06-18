import os
import glob
import json
import shutil
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu


def compute_time_cut(df_header, df_phot, timevar_to_cut=None):
    # Time cut
    lu.print_green(f"Compute time cut with {timevar_to_cut}")
    df_info_for_skim = df_header[["SNID", timevar_to_cut]]
    df_phot = pd.merge(df_phot, df_info_for_skim, on="SNID", how="left")
    mask = (df_phot['MJD'] != -777.00)
    df_phot['delta_time'] = df_phot['MJD'] - df_phot[timevar_to_cut]
    df_phot['time_cut'] = True
    df_phot.loc[mask, 'time_cut'] = df_phot["delta_time"].apply(
        lambda x: True if (x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False)
    )
    df_phot = df_phot[df_phot['time_cut'] == True]

    ids_to_keep = df_phot["SNID"].unique()
    df_header = df_header[df_header["SNID"].isin(ids_to_keep.tolist())]

    return df_header, df_phot

def decode_timevar(timevar):
    if timevar == 'trigger':
        timevar_to_cut = 'PRIVATE(DES_mjd_trigger)'
    else:
        timevar_to_cut = 'PKMJDINI'
    return timevar_to_cut

def apply_cut_save(df_header_ori, df_phot_ori, timevar=None, dump_dir=None,
                   dump_prefix=None):
    # init
    timevar_to_cut = decode_timevar(timevar)

    # apply cuts
    df_header, df_phot = compute_time_cut(
        df_header_ori, df_phot_ori, timevar_to_cut=timevar_to_cut)
    # save columns of original header
    keys_header = df_header_ori.keys().tolist()
    # add S/N info
    df_phot['S/N'] = df_phot['FLUXCAL'] / df_phot['FLUXCALERR']

    # format sntypes as sim
    if 'fake' in dump_dir:
        df_header["SNTYPE"] = df_header["SNTYPE"].apply(lambda x: 1 if x == 0 else 0).copy()
    else:
        # need to add spec
        df_header["SNTYPE"] = df_header["SNTYPE"].apply(lambda x: 1 if x == 1 else 0).copy()

    # add some extra information abotu the lc
    # select only high S/N values
    df_tmp = df_phot.groupby('SNID').max()
    df_tmp['SNID'] = df_tmp.index
    df_extra_info = df_tmp[['SNID', 'FLUXCAL','S/N']]
    df_extra_info = df_extra_info.rename(columns={'SNID': 'SNID', 'FLUXCAL': 'FLUXCAL_max', 'S/N': 'S/N_max'})
    df_phot = df_phot.merge(df_extra_info, on='SNID')

    # save
    df_phot_saved = du.save_phot_fits(df_phot, f'{dump_dir}/{timevar}/{dump_prefix}_PHOT.FITS')
    df_phot_saved = df_phot_saved[df_phot_saved['SNID'] != 0]
    # in order to keep same ordering
    df_phot_for_header = df_phot_saved.loc[df_phot_saved["SNID"].shift() != df_phot_saved["SNID"]]
    df_phot_for_header = df_phot_for_header.reset_index()
    df_header_tosave = df_phot_for_header[['SNID', 'FLUXCAL_max', 'S/N_max']].merge(df_header, on='SNID')
    df_header_tosave = df_header_tosave[keys_header + ['FLUXCAL_max', 'S/N_max']]
    filename = f"{dump_prefix}_HEAD.FITS"
    du.save_fits(df_header_tosave, f'{dump_dir}/{timevar}/{filename}')

    return df_header_tosave.SNTYPE.unique().tolist(), filename


def skim_data(raw_dir, dump_dir, fits_file, timevar, debug=False):
    """ Skim PHOT and HEAD.FITS
    """
    list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))
    if debug:
        lu.print_yellow('Debugging mode')
        list_files = list_files[:1]
    lu.print_green(f"Starting data skimming, found {len(list_files)} to operate on")

    # load Bazin
    df_fits = None
    if Path(fits_file).exists():
        df_fits = du.load_fits(fits_file)

    tmp_type_list = []
    filenames = []
    # skim each FITS file
    for fname in list_files:
        # fetch data year as prefix
        dump_prefix = Path(fname).name.split("_")[0]
        lu.print_blue(f"Processing: {dump_prefix}")

        df_header, df_phot = du.read_fits(fname)
        if df_fits is not None:
            df_header = pd.merge(df_header, df_fits, on='SNID')
        df_header = df_header[[k for k in df_header.keys() if 'Unnamed' not in k]]
        # apply cuts
        unique_types, filename = apply_cut_save(df_header, df_phot, timevar=timevar, dump_dir=dump_dir, dump_prefix=dump_prefix)
        tmp_type_list += unique_types
        filenames.append(filename)

    # # Copy *all* auxiliary files
    # aux_files = [f for f in os.listdir(raw_dir) if not f.endswith("FITS")]
    # for f in aux_files:
    #     shutil.copy(os.path.join(raw_dir, f),  f'{dump_dir}/{timevar}/')

    # Also need to update the .LIST file
    # start = os.path.basename(raw_dir)
    # list_file = f"{dump_dir}/{timevar}/{start}.LIST"
    # with open(list_file, "w") as f:
    #     for file in filenames:
    #         f.write(f"{file}\n")

    # # Save out the types
    # tmp = list(set(tmp_type_list))
    # type_list = [(k, du.spec_type_decoder(k)) for k in tmp]
    # with open(f'{dump_dir}/sntypes.json', 'w') as outfile:
    #     json.dump(type_list, outfile)

