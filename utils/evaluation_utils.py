import numpy as np
import pandas as pd
import utils.data_utils as du
import matplotlib.pylab as plt
import utils.logging_utils as lu
from utils import visualization_utils as vu


def fetch_prediction_info(settings, model_settings, skim_dir):

    # fetch prediction
    pred_dir = f"{settings.dump_dir}/models/{model_settings.pytorch_model_name}"
    prediction_file = f"{pred_dir}//PRED_{model_settings.pytorch_model_name}.pickle"
    df = pd.read_pickle(prediction_file)

    # add header info
    df_SNinfo = du.fetch_header_info(skim_dir)
    cols_to_merge = [
        k for k in df_SNinfo.keys() if k not in df.keys()] + ["SNID"]
    df = df.merge(df_SNinfo.reset_index()[
                  cols_to_merge], how="left", on="SNID")

    # compute predicted target for complete lc classification
    df["predicted_target"] = (
        df[[k for k in df.keys() if "all_class" in k]]
        .idxmax(axis=1)
        .str.strip("all_class")
        .astype(int))

    return df


def plot_efficiency(df, skim_dir, path_plot):

    # fake efficiency
    if "PRIVATE(DES_fake_z)" in df.keys():
        # bin in redshift
        df["bin"], sliced_bins = pd.cut(
            df["PRIVATE(DES_fake_z)"], 10, retbins=True)
        sorted_df = df.sort_values("PRIVATE(DES_fake_z)")
        bin_list = sorted_df["bin"].unique()
        list_efficiency = []
        list_accuracy = []
        list_redshift = []
        for i, mybin in enumerate(bin_list):
            selected = sorted_df[sorted_df["bin"] == mybin]
            dic_metric = {}
            for key, value in zip(
                ["accuracy", "auc", "purity", "efficiency", "truepositivefraction"],
                performance_metrics(selected),
            ):
                dic_metric[key] = value
            list_efficiency.append(dic_metric["efficiency"])
            list_accuracy.append(dic_metric["accuracy"])
            list_redshift.append(selected["PRIVATE(DES_fake_z)"].mean())
        fig = plt.figure()
        plt.scatter(list_redshift, list_efficiency)
        plt.xlabel("PRIVATE(DES_fake_z)")
        plt.ylabel("efficiency")
        plt.savefig(f"{path_plot}/efficiency.png")


def performance_metrics(df, sample_target=0):
    """Get performance metrics
    AUC: only valid for binomial classification, input proba of highest label class.

    Args:
        df (pandas.DataFrame) (str): with columns [target, predicted_target, class1]
        (optional) sample_target (str): for SNIa sample default is target 0

    Returns:
        accuracy, auc, purity, efficiency,truepositivefraction
    """
    from sklearn import metrics
    n_targets = len(np.unique(df["target"]))

    # Accuracy & AUC
    accuracy = metrics.accuracy_score(df["target"], df["predicted_target"])
    accuracy = round(accuracy * 100, 2)
    if n_targets == 2:  # valid for biclass only
        auc = round(metrics.roc_auc_score(df["target"], df["class1"]), 4)
    else:
        auc = 0.0

    SNe_Ia = df[df["target"] == sample_target]
    SNe_CC = df[df["target"] != sample_target]
    TP = len(SNe_Ia[SNe_Ia["predicted_target"] == sample_target])
    FP = len(SNe_CC[SNe_CC["predicted_target"] == sample_target])

    P = len(SNe_Ia)
    N = len(SNe_CC)

    truepositivefraction = P / (P + N)
    purity = round(100 * TP / (TP + FP), 2) if (TP + FP) > 0 else 0
    efficiency = round(100.0 * TP / P, 2) if P > 0 else 0

    return accuracy, auc, purity, efficiency, truepositivefraction


def pair_plots(df, path_plot):

    import seaborn as sns
    if 'TYPE' in df.keys():
        df['SNTYPE'] = 'TYPE'
    host_plots = sns.pairplot(df[['HOSTGAL_CONFUSION', 'HOSTGAL_LOGMASS', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_z', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ',
                                  'HOSTGAL_SPECZ_ERR', 'SNTYPE']], hue="SNTYPE")
    host_plots.savefig(f"{path_plot}/pair_host.png")

    other_plots = sns.pairplot(df[['PRIVATE(DES_numepochs)', 'PRIVATE(DES_numepochs_ml)',
                                   'PRIVATE(DES_transient_status)', 'REDSHIFT_FINAL', 'SNTYPE', 'predicted_target']], hue="SNTYPE")
    other_plots.savefig(f"{path_plot}/pair_other.png")


def print_percentages(what, df_sel, df, color='red'):
    if color == 'red':
        lu.print_red(f"{what} ", f"{len(df_sel)} = {round(100*len(df_sel)/len(df),1)}%")
    elif color == 'blue':
        lu.print_blue(f"{what} ", f"{len(df_sel)} = {round(100*len(df_sel)/len(df),1)}%")


def get_time_cut_stats_and_plot(df_pred, photo_Ia, skim_dir, cut_type, plot=False, model_files=None):
    # get stats for time cuts
    fig = plt.figure()
    photo_Ia['fakemjd'] = photo_Ia['PRIVATE(DES_fake_peakmjd)']
    photo_Ia['trigger'] = photo_Ia['PRIVATE(DES_mjd_trigger)']
    photo_Ia['delta_trigger'] = photo_Ia.apply(
        lambda row: row.trigger - row.fakemjd, axis=1)
    photo_Ia['delta_bazin'] = photo_Ia.apply(
        lambda row: row.PKMJDINI - row.fakemjd, axis=1)
    df_pred['fakemjd'] = df_pred['PRIVATE(DES_fake_peakmjd)']
    df_pred['trigger'] = df_pred['PRIVATE(DES_mjd_trigger)']
    df_pred['delta_trigger'] = df_pred.apply(
        lambda row: row.trigger - row.fakemjd, axis=1)
    df_pred['delta_bazin'] = df_pred.apply(
        lambda row: row.PKMJDINI - row.fakemjd, axis=1)

    if 'trigger' in cut_type:
        print_percentages(f"Overflows trigger(window  max)", df_pred[abs(df_pred['delta_trigger']) > 100], df_pred)
        print_percentages(f"                    photo_Ia  ", photo_Ia[abs(photo_Ia['delta_trigger']) > 100], photo_Ia)
        print_percentages(f"Overflows trigger(window .5lc)", df_pred[abs(df_pred['delta_trigger']) > 50], df_pred)
        print_percentages(f"                    photo_Ia  ", photo_Ia[abs(photo_Ia['delta_trigger']) > 50], photo_Ia)
    else:
        # bazin does not crashes but have more outside window
        print_percentages(f"Overflows bazin (window max)  ", df_pred[abs(df_pred['delta_bazin']) > 100], df_pred)
        print_percentages(f"                    photo_Ia  ", photo_Ia[abs(photo_Ia['delta_bazin']) > 100], photo_Ia)
        print_percentages(f"Overflows bazin (window .5lc) ", df_pred[abs(df_pred['delta_bazin']) > 50], df_pred)
        print_percentages(f"                    photo_Ia  ", photo_Ia[abs(photo_Ia['delta_bazin']) > 50], photo_Ia)

    if plot:
        vu.plot_early_classification(
            skim_dir, model_files=model_files, prefix='OF_bazin_window_', df=photo_Ia[abs(photo_Ia['delta_bazin']) > 100])

    # plot both possible cuts
    plt.hist(photo_Ia['delta_trigger'],
             histtype='step', label='trigger', bins=1000)
    plt.hist(photo_Ia['delta_bazin'],
             histtype='step', label='bazin', bins=1000)
    plt.xlabel('delta_time')
    plt.xlim(-1000, 1000)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{skim_dir}/figures/phot_Ia_delta_time.png")
    del fig

    fig = plt.figure()
    plt.hist(photo_Ia['delta_trigger'],
             histtype='step', label='trigger', bins=1000)
    plt.hist(photo_Ia['delta_bazin'],
             histtype='step', label='bazin', bins=1000)
    plt.xlabel('delta_time')
    plt.xlim(-100, 100)
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{skim_dir}/figures/phot_Ia_delta_time_zoom.png")
    del fig
