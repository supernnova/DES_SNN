import os
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import utils.data_utils as du
import utils.logging_utils as lu

def compute_time_cut(df, mask, time_cut):
    """ LC time skimming
    Central for time cuts

    Args:
        df
        mask
        time_cut
        path_prep [str(optional)]

    """

    lu.print_green("Compute time cut")
    if time_cut == "trigger":
        df['delta_time'] = df['MJD']-df['PRIVATE(DES_mjd_trigger)']
        df['time_cut'] = True
        df.loc[mask, 'time_cut'] = df["delta_time"].apply(lambda x: True if (
            x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False))
        return df
    elif time_cut == "subseason":
        lu.print_red("Implementation waaaaay to slow")
        raise AttributeError
        df = compute_delta_time(df)
        df = reformat_subseasons(df)
        # dummy time cut, since it is already done
        df['time_cut'] = True
        return df
    elif time_cut == "bazin":
        #renaming for SNN to plot the fake peak
        df['delta_time'] = df['MJD']-df['PKMJDINI']
        df['time_cut'] = True
        df.loc[mask, 'time_cut'] = df["delta_time"].apply(lambda x: True if (
            x > 0 and x < 70) else (True if (x <= 0 and x > -30) else False))
        return df
    elif time_cut == None:
        return df


def compute_S_N_cut(df, mask, SN_threshold=None):
    # S/N cut (for limiting magnitudes)
    lu.print_green("Compute S/N cut")
    df['S/N'] = df['FLUXCAL']/df['FLUXCALERR']
    df['S/N_cut'] = True
    if SN_threshold:
        df.loc[mask,
               'S/N_cut'] = df["S/N"].apply(lambda x: True if x > 3 else False)

    return df


def apply_cuts(df):

    df_sel = df[(df['time_cut'] == True) & (df['S/N_cut'] == True)]
    df_sel = df_sel.reset_index()

    return df_sel

def insert_row(idx, df, df_insert):
    # Option 2 to insert row
    # slow but the best for the moment
    df.index = df.index[:idx].append(df.index[idx:]+1)
    df.loc[idx+1] = df_insert
    return df

def insert_row0(idx, df, df_insert):
    # Option 0 to insert row
    # slow
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]
    df = dfA.append(df_insert).append(dfB).reset_index(drop=True)

    return df

def insert_row1(row_number, df, row_value):
    # Option 1 to insert row
    # slow 
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df


def reformat_subseasons(df):
    """ Reformat file with subseasons

    - for each unique SNID, find subseasons
    - split with new SNIDs and -777.0 separators

    (new header will be derived from this file)

    !!! This is quite slow, need to speed it up
    """

    lu.print_green("Reformat subseasons")
    subseason_break_index = df.index[df['delta_time'] > 100]

    # separator row to insert
    df_insert = df.iloc[0].copy()
    for key in df.keys():
        if key != "FIELD" and key != "FLT":
            df_insert[key] = -777.0
        else:
            df_insert[key] = "XXXX"

    lu.print_green("Change SNIDs")
    # reformat as if it was a new light-curve
    for i, ind in enumerate(subseason_break_index):
        # save for use
        if i != 0:
            before = subseason_break_index[i-1]
            sid_before = df.iloc[before]['SNID']
        else:
            before = 0
            sid_before = df.iloc[ind]['SNID']
        sid_now = df.loc[ind]['SNID']
        # add subseason tag for SNID.subseason
        if int(sid_before) != int(sid_now) or i == 0:
            counter = 0.1
            set_index = df.iloc[ind]["SNID"] + counter
        else:
            counter += 0.1
            set_index = df.iloc[before]["SNID"] + counter
        df.at[before:ind-1, "SNID"] = set_index

    lu.print_green("Adding -777")
    # reversing order for row insert (to conserve sequence)
    for ind in reversed(subseason_break_index):
        # set delta_time to zero
        df.at[ind, 'delta_time'] = 0
        # add -777.0 separator
        df = insert_row(ind-1, df, df_insert)
    lo.print_green("Sort")
    df.sort_index(inpalce=True)

    return df


def compute_delta_time(df):
    """Compute the delta time between two consecutive observations

    Args:
        df (pandas.DataFrame): dataframe holding lightcurve data

    Returns:
        (pandas.DataFrame) dataframe holding lightcurve data with delta_time features
    """
    lu.print_green("Compute delta time")

    if df.MJD.values[0] == -777.0:
        df = df.drop(df.index[0])

    df["delta_time"] = df["MJD"].diff()
    # Fill the first row with 0 to replace NaN
    df.delta_time = df.delta_time.fillna(0)
    try:
        IDs = df.SNID.values
    # Deal with the case where lightcrv_ID is the index
    except AttributeError:
        assert df.index.name == "SNID"
        IDs = df.index.values
    # Find idxs of rows where a new light curve start then zero delta_time
    # and where the photo splits are (-777)
    idxs = np.array((np.where(IDs[:-1] != IDs[1:])[0] + 1).tolist() +
                    (np.where(IDs[:-1] != IDs[1:])[0] + 2).tolist())
    arr_delta_time = df.delta_time.values
    arr_delta_time[idxs] = 0
    df["delta_time"] = arr_delta_time

    return df

