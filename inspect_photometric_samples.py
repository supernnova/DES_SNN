import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import utils.logging_utils as lu
import utils.new_data_utils as du
from utils import visualization_utils as vu

"""
Photometric samples
"""
# Init
path_des_data = os.environ.get("DES_DATA")

# Load predictions for all data available
dic_pred = {}
for dtype in ["real", "fake"]:
    lu.print_green(f'____{dtype}____')
    path_predictions = f"./dumps/{dtype}/clump/"
    # available models
    list_predictions = glob.glob(f"{path_predictions}/models/vanilla*/PRED*.pickle")
    list_predictions += glob.glob(f"{path_predictions}/models/*/PRED*_aggregated.pickle")
    # common samples
    path_common = f"./dumps/{dtype}/clump/common_sample"
    path_common_plots = f"{path_common}/plots/"
    Path(path_common_plots).mkdir(parents=True, exist_ok=True)

    dic_pred[dtype] = {}

    for fname_preds in list_predictions:
        # init
        name_model = str(Path(fname_preds).parent).split('/')[-1]#Path(fname_preds).parent.stem
        path_dtype_data = f"{path_des_data}/DESALL_forcePhoto_{dtype}_snana_fits/"
        out_dir = f"{path_predictions}/sample/{name_model}/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plots_dir = f"{path_predictions}/sample/{name_model}/plots/"
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

        # load preds & enrich
        df_pred = du.load_predictions(fname_preds)
        df_pred = du.enrich_predictions(df_pred, path_dtype_data)

        # save the preds
        dic_pred[dtype][name_model] = df_pred

        # Select a default "photometric" sample
        photo_sample = df_pred[df_pred['predicted_target'] == 0]
        dic_pred[dtype][name_model]['photo_sample'] = np.array([df_pred['predicted_target'] == 0])[0]

        # save photo sample
        cols_to_save = ['SNID','HOSTGAL_OBJID', 'DEC', 'RA', 'SNTYPE', 'REDSHIFT_FINAL', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ']
        cols_to_save += ['all_class0'] if 'vanilla' in name_model else ['all_class0_median','all_class0_std']
        photo_sample[cols_to_save].to_csv(f"{out_dir}/photo_sample.csv")
        lu.print_blue(name_model)
        print(f'photo sample {len(photo_sample)} representing {int(len(photo_sample)/len(df_pred)*100)}%')

        # metrics
        if dtype == 'real':
            # Ias
            dic_pred[dtype][name_model]['photo_spec_Ia'] = np.array([(df_pred['predicted_target'] == 0) & ((df_pred['SNTYPE'] == 1) | (
                df_pred['SNTYPE'] == 101))])[0]
            spec_Ia = df_pred[(df_pred['SNTYPE'] == 1) |
                              (df_pred['SNTYPE'] == 101)]
            print('spec Ia', len(dic_pred[dtype][name_model][dic_pred[dtype][name_model]['photo_spec_Ia']==True]), f"from {len(spec_Ia)}")
            # non Ias
            dic_pred[dtype][name_model]['photo_spec_nonIa'] = np.array([(df_pred['predicted_target'] == 0) & (df_pred['SNTYPE'] != 0) & (df_pred['SNTYPE'] != 1) & (
                df_pred['SNTYPE'] != 101)])[0]
            spec_non_Ia = df_pred[(df_pred['SNTYPE'] != 0) & (df_pred['SNTYPE'] != 1) &
                              (df_pred['SNTYPE'] != 101)]
            print('spec non Ia', len(dic_pred[dtype][name_model][dic_pred[dtype][name_model]['photo_spec_nonIa']==True]), f"from {len(spec_non_Ia)}")

        # correlation with c,x1, peakmag
        # get salt info (which is not available for all photo Ias)
        # from salt fit
        list_fitres = glob.glob(f"{path_dtype_data}/*FITRES")
        saltfit = pd.read_csv(list_fitres[0], index_col=False,
                         comment='#', delimiter=' ', skipinitialspace=True)
        saltfit['SNID'] = saltfit['CID']
        saltfit = saltfit[['SNID']+[k for k in saltfit.keys()
                                            if k not in photo_sample.keys()]]
        df_preds_salt = dic_pred[dtype][name_model].merge(saltfit, on='SNID')
        df_photo_salt = df_preds_salt[df_preds_salt['photo_sample']]
        if dtype == 'real':
            df_specIa_salt = df_preds_salt[df_preds_salt['photo_spec_Ia']]

        # plot color, strech and mag distributions
        for var in ['c','x1','mB','REDSHIFT_FINAL']:
            fig = plt.figure()
            plt.hist(df_preds_salt[var],color='black',histtype='step',label='all',bins=20)
            plt.hist(df_photo_salt[var],color='orange',histtype='step',label='photo Ias',bins=20)
            if dtype == 'real':
                plt.hist(df_specIa_salt[var],color='green',histtype='step',label='spec Ias',bins=20)
            plt.legend()
            plt.yscale('log')
            plt.xlabel(var)
            plt.savefig(f"{plots_dir}/hist_{var}.png")
            del fig

        # plot lcs
        model_files = [f"../SuperNNova_general/trained_models_mutant/{name_model}/{name_model}.pt"]
        photo_sample['TYPE'] = photo_sample['SNTYPE']
        vu.plot_early_classification(path_predictions, prefix='photo_Ia_',
                                     df=photo_sample, model_files=model_files, out_dir=path_predictions)



    # crossmatch between different models samples
    from matplotlib_venn import venn2
    for m1 in list_predictions:
        for m2 in [p for p in list_predictions if m1!=p]:
            name_m1 = Path(m1).parent.stem
            name_m2 = Path(m2).parent.stem
            out_name_m1 = f"{Path(m1).parent.stem.split('_')[0]}_{Path(m1).parent.stem.split('_')[6]}"
            out_name_m2 = f"{Path(m2).parent.stem.split('_')[0]}_{Path(m2).parent.stem.split('_')[6]}"

            photo_m1 = dic_pred['real'][name_m1][dic_pred['real'][name_m1]['photo_sample']]
            photo_m2 = dic_pred['real'][name_m2][dic_pred['real'][name_m2]['photo_sample']]

            fig = plt.figure()
            venn2(
                [
                    set(photo_m1['SNID'].values
                        ),
                    set(
                        photo_m2['SNID'].values
                    )
                ], (out_name_m1,out_name_m2))
            plt.savefig(f"{path_common_plots}/venn_{out_name_m1}_{out_name_m2}.png")
            plt.close()
            del fig


            if 'zpho' not in name_m1 and 'zpho' not in name_m2 and name_m1!=name_m2:
                common = photo_m1[photo_m1['SNID'].isin(photo_m2['SNID'].tolist())]
                cols_to_save = ['SNID','HOSTGAL_OBJID', 'DEC', 'RA', 'SNTYPE', 'REDSHIFT_FINAL', 'HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ']
                photo_sample[cols_to_save].to_csv(f"{path_common}/photo_sample_{out_name_m1}_{out_name_m2}.csv")

