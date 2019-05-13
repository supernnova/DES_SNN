import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import utils.data_utils as du


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
                                  'HOSTGAL_SPECZ_ERR','SNTYPE']], hue="SNTYPE")
    host_plots.savefig(f"{path_plot}/pair_host.png")

    other_plots = sns.pairplot(df[['PRIVATE(DES_numepochs)', 'PRIVATE(DES_numepochs_ml)',
                                   'PRIVATE(DES_transient_status)', 'REDSHIFT_FINAL', 'SNTYPE', 'predicted_target']],hue="SNTYPE")
    other_plots.savefig(f"{path_plot}/pair_other.png")
