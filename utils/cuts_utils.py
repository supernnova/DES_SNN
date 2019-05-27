import os
import glob
import json
import shutil

import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu


def compute_time_cut(df_header, df_phot, time_cut_type=None, timevar_to_cut=None):
    # Time cut
    lu.print_green(f"Compute time cut {time_cut_type} with {timevar_to_cut}")
    df_phot['time_cut'] = True
    if time_cut_type == 'window':
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


def compute_S_N_cut(df_header, df_phot, SN_threshold=None):
    # S/N cut (for limiting magnitudes)
    df_phot['S/N'] = df_phot['FLUXCAL'] / df_phot['FLUXCALERR']
    df_phot['S/N_cut'] = True
    mask = (df_phot['MJD'] != -777.00)
    if SN_threshold:
        lu.print_green(f"Compute S/N cut {SN_threshold}")
        df_phot.loc[mask,
                    'S/N_cut'] = df_phot["S/N"].apply(lambda x: True if x > SN_threshold else False)

        df_phot = df_phot[df_phot['S/N_cut'] == True]

        ids_to_keep = df_phot["SNID"].unique()
        df_header = df_header[df_header["SNID"].isin(ids_to_keep.tolist())]

    return df_header, df_phot


def apply_cut_save(df_header_ori, df_phot_ori, time_cut_type=None, timevar=None, SN_threshold=None, dump_dir=None,
                   dump_prefix=None, cut_version=None):
    # init
    if timevar == 'trigger':
        timevar_to_cut = 'PRIVATE(DES_mjd_trigger)'
    elif timevar == 'bazin':
        timevar_to_cut = 'PKMJDINI'
    else:
        timevar_to_cut = None

    if cut_version is None:
        cut_version = f"{time_cut_type}_{timevar}_SN{SN_threshold}"

    # apply cuts
    df_header, df_phot = compute_time_cut(
        df_header_ori, df_phot_ori, time_cut_type=time_cut_type, timevar_to_cut=timevar_to_cut)
    # save columns of original header
    keys_header = df_header_ori.keys().tolist()
    # compute S/N cut if any
    df_header, df_phot = compute_S_N_cut(df_header, df_phot, SN_threshold=None)

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
    df_extra_info = df_tmp[['SNID', 'FLUXCAL', 'S/N']]
    df_extra_info = df_extra_info.rename(columns={'SNID': 'SNID', 'FLUXCAL': 'FLUXCAL_max', 'S/N': 'S/N_max'})
    df_phot = df_phot.merge(df_extra_info, on='SNID')

    # save
    df_phot_saved = du.save_phot_fits(df_phot, f'{dump_dir}/{cut_version}/{dump_prefix}_PHOT.FITS')
    df_phot_saved = df_phot_saved[df_phot_saved['SNID'] != 0]
    # in order to keep same ordering
    df_phot_for_header = df_phot_saved.loc[df_phot_saved["SNID"].shift() != df_phot_saved["SNID"]]
    df_phot_for_header = df_phot_for_header.reset_index()
    df_header_tosave = df_phot_for_header[['SNID', 'FLUXCAL_max', 'S/N_max']].merge(df_header, on='SNID')
    df_header_tosave = df_header_tosave[keys_header + ['FLUXCAL_max', 'S/N_max']]
    filename = f"{dump_prefix}_HEAD.FITS"
    du.save_fits(df_header_tosave, f'{dump_dir}/{cut_version}/{filename}')

    return df_header_tosave.SNTYPE.unique().tolist(), filename


def skim_data(raw_dir, dump_dir, bazin_file, time_cut_type, timevar, SN_threshold, cut_version=None, debug=False):
    """ Skim PHOT and HEAD.FITS
    """
    list_files = glob.glob(os.path.join(f"{raw_dir}", "*PHOT.FITS"))
    if debug:
        lu.print_yellow('Debugging mode')
        list_files = list_files
    lu.print_green(f"Starting data skimming, found {len(list_files)} to operate on")

    # load Bazin
    df_bazin = None
    if Path(bazin_file).exists():
        df_bazin = du.load_bazin_fits(bazin_file)

    tmp_type_list = []
    filenames = []
    # skim each FITS file
    for fname in list_files:
        # fetch data year as prefix
        dump_prefix = Path(fname).name.split("_")[0]
        lu.print_blue(f"Processing: {dump_prefix}")

        df_header, df_phot = du.read_fits(fname)
        if df_bazin is not None:
            df_header = pd.merge(df_header, df_bazin, on='SNID')
        df_header = df_header[[k for k in df_header.keys() if 'Unnamed' not in k]]
        # apply cuts
        unique_types, filename = apply_cut_save(df_header, df_phot, time_cut_type=time_cut_type, timevar=timevar, cut_version=cut_version,
                       SN_threshold=SN_threshold, dump_dir=dump_dir, dump_prefix=dump_prefix)
        tmp_type_list += unique_types
        filenames.append(filename)

    # Copy *all* auxiliary files
    aux_files = [f for f in os.listdir(raw_dir) if not f.endswith("FITS")]
    for f in aux_files:
        shutil.copy(os.path.join(raw_dir, f),  f'{dump_dir}/{cut_version}/')

    # Also need to update the .LIST file
    start = os.path.basename(raw_dir)
    list_file = f"{dump_dir}/{cut_version}/{start}.LIST"
    with open(list_file, "w") as f:
        for file in filenames:
            f.write(f"{file}\n")

    # Save out the types
    tmp = list(set(tmp_type_list))
    type_list = [(k, du.spec_type_decoder(k)) for k in tmp]
    with open(f'{dump_dir}/sntypes.json', 'w') as outfile:
        json.dump(type_list, outfile)

