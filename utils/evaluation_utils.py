import pandas as pd
import matplotlib.pylab as plt
import utils.data_utils as du

def fetch_prediction_info(settings,model_settings,skim_dir):

    # fetch prediction
    pred_dir = f"{settings.dump_dir}/models/{model_settings.pytorch_model_name}"
    prediction_file = f"{pred_dir}//PRED_{model_settings.pytorch_model_name}.pickle"
    df = pd.read_pickle(prediction_file)

    #add header info
    df_SNinfo = du.fetch_header_info(skim_dir)
    cols_to_merge = ["SNID","REDSHIFT_FINAL", "SNTYPE"]
    if "PRIVATE(DES_fake_z)" in df_SNinfo.keys():
        cols_to_merge += ["PRIVATE(DES_fake_z)"]
    df = df.merge(df_SNinfo.reset_index()[cols_to_merge], how="left", on="SNID")

    # compute predicted target for complete lc classification
    df["predicted_target"] = (
        df[[k for k in df.keys() if "all_class" in k]]
        .idxmax(axis=1)
        .str.strip("all_class")
        .astype(int))

    return df

def plot_efficiency(df,skim_dir,path_plot):
    
    # fake efficiency
    if "PRIVATE(DES_fake_z)" in df.keys():
        # bin in redshift
        df["bin"], sliced_bins = pd.cut(df["PRIVATE(DES_fake_z)"], 10, retbins=True)
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
        plt.scatter(list_redshift,list_efficiency)
        plt.xlabel("PRIVATE(DES_fake_z)")
        plt.ylabel("efficiency")
        plt.savefig(f"{path_plot}/efficiency.png")