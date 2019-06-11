import sys
import glob,os
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table
from matplotlib import pyplot as plt
sys.path.append("../")
import utils.data_utils as du
import utils.visualization_utils as vu

def lcplot_stats(prefix,df_sel,model_name,plot=True, save=False):
	if plot:
		vu.plot_early_classification(data_dir, prefix=prefix, df=df_sel, model_files=[f"{model_dump_dir}/{model_name}/{model_name}.pt"], out_dir=f"../dumps/real/clump/")
	# some stats
	print(prefix,len(sample))
	print('spec Ia',len(sample[(sample['SNTYPE']==1) | (sample['SNTYPE']==101)]))
	print('spec nonIa',len(sample[ (sample['SNTYPE']!=1) & (sample['SNTYPE']!=101) & (sample['SNTYPE']!=0) & (sample['SNTYPE']!=80) ]))
	print('AGN',len(sample[sample['SNTYPE']==80]))
	print('_____')

	if save:
		save_sample_csv = f"{data_dir}/sample/{model_name}/SN_{prefix}.csv"
		Path(save_sample_csv).parent.mkdir(parents=True, exist_ok=True)
		sample[['SNID']].to_csv(f'{save_sample_csv}')

# model_list = [f"variational_S_0_CLF_{i}_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.01_128_True_mean_C_WD_1e-07" for i in [2,3,7]]
model_list = [f"variational_S_0_CLF_{i}_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.01_128_True_mean_C_WD_1e-07" for i in [7]]
model_dump_dir = "../../SuperNNova_general/trained_models_mutant/"
data_dir = "../dumps/real/clump"
pred_dir = f'{data_dir}/models/'

for model_name in model_list:
	pred_file = f'{pred_dir}/{model_name}/PRED_{model_name}_aggregated.pickle'
	df_pred = pd.read_pickle(pred_file)

	# merge with header info
	# get header info to see corrrelations
	path_des_data = os.environ.get("DES_DATA")
	raw_dir = f"{path_des_data}/DESALL_forcePhoto_real_snana_fits/"
	list_files = glob.glob(os.path.join(f"{raw_dir}", "*HEAD.FITS"))
	list_df_h = []
	for f in list_files:
	    header = Table.read(f, format="fits")
	    df_header = header.to_pandas()
	    df_header["SNID"] = df_header["SNID"].astype(np.int32)
	    list_df_h.append(df_header)
	df_head = pd.concat(list_df_h, sort=True)

	df_pred = df_pred.merge(df_head,on='SNID')
	df_pred['TYPE']=df_pred['SNTYPE']
	df_pred["predicted_target"] = (
	        df_pred[[k for k in df_pred.keys() if "all_class" in k and "median" in k]]
	        .idxmax(axis=1)
	        .str.strip("all_class").str.strip("_median")
	        .astype(int))
	df_pred["std_probs"] = df_pred[[k for k in df_pred.keys() if "all_class" in k and "median" in k]].std(axis=1)

	# for i in df_pred["predicted_target"].unique():
	# 	df_sel = df_pred[ (df_pred[f"all_class{i}_std"]>0.5) & (df_pred[f"all_class{i}_median"]<0.5)]
	# 	vu.plot_early_classification(data_dir, prefix=f"class{i}_lowprob_highstd05", df=df_sel, model_files=[f"{model_dump_dir}/{model_name}/{model_name}.pt"], out_dir=f"../dumps/real/clump/")

	# for i in df_pred["predicted_target"].unique():
	# 	df_sel = df_pred[ (df_pred[f"all_class{i}_std"]<0.3) & (df_pred[f"all_class{i}_median"]>0.7)]
	# 	print('selected as high prob>0.7 and low uncertainty <0.3',len(df_sel),'from',len(df_pred))
	# 	vu.plot_early_classification(data_dir, prefix=f"class{i}_highprob07_lowstd03", df=df_sel, model_files=[f"{model_dump_dir}/{model_name}/{model_name}.pt"], out_dir=f"../dumps/real/clump/")

	if 'CLF_7' in model_name:

		# # v0
		# # no IIn, restrict IIps to low-uncertainty
		# sample = df_pred[ (df_pred['predicted_target']!=2 ) ]
		# sample = sample[ (sample['predicted_target']!=1) | ((sample['predicted_target']==1) & (sample[f'all_class1_std']<0.1)) ]
		# df_sel = sample
		# prefix = "sample_v0_"
		# lcplot_stats(prefix,df_sel,model_name)

		# # v1
		# # high-prob and low-unc
		sub_sample_list = []
		for i in range(0,7):
			sub_sample = df_pred[ (df_pred['predicted_target']==i) &  (df_pred[f"all_class{i}_std"]<0.2) &  (df_pred[f"all_class{i}_median"]>.7)]
			sub_sample_list.append(sub_sample)
		sample = pd.concat(sub_sample_list, sort=True)
		lcplot_stats("sample_v1_",sample,model_name,plot=False)

		# v2
		# no IIn, restrict uncertainty
		sub_sample_list = []
		for i in range(0,7):
			sub_sample = df_pred[ (df_pred['predicted_target']==i) &  (df_pred[f"all_class{i}_std"]<0.2) ]
			sub_sample_list.append(sub_sample)
		sample = pd.concat(sub_sample_list, sort=True)
		sample = sample[ (sample['predicted_target']!=2 ) ]
		lcplot_stats("sample_v2_",sample,model_name,plot=False)

		# v3
		# no IIn, restrict uncertainty, prob>0.5
		sub_sample_list = []
		for i in range(0,7):
			sub_sample = df_pred[ (df_pred['predicted_target']==i) &  (df_pred[f"all_class{i}_std"]<0.2) & (df_pred[f"all_class{i}_median"]>.5)]
			sub_sample_list.append(sub_sample)
		sample = pd.concat(sub_sample_list, sort=True)
		sample = sample[ (sample['predicted_target']!=2 ) ]
		lcplot_stats("sample_v3_",sample,model_name,plot=False,save=True)

		# v4
		# no IIns, restrict uncertainty, no probabilities all concentrated in the bottom
		sub_sample_list = []
		for i in range(0,7):
			sub_sample = df_pred[ (df_pred['predicted_target']==i) &  (df_pred[f"all_class{i}_std"]<0.2) & (df_pred[f"std_probs"]>.3)]
			sub_sample_list.append(sub_sample)
		sample = pd.concat(sub_sample_list, sort=True)
		sample = sample[ (sample['predicted_target']!=2 ) ]
		lcplot_stats("sample_v4_",sample,model_name,plot=False)

		# v5
		# no IIns, no probabilities all concentrated in the bottom
		sample = df_pred[ (df_pred['predicted_target']!=2 ) & (df_pred[f"std_probs"]>.3) ]
		lcplot_stats("sample_v5_",sample,model_name,plot=False)

		# v6
		# no probabilities all concentrated in the bottom
		sample = df_pred[ (df_pred[f"std_probs"]>.3) ]
		lcplot_stats("sample_v6_",sample,model_name,plot=False)

		# v7
		# loose no probabilities all concentrated in the bottom
		# restric IIp
		sample = df_pred[ (df_pred[f"std_probs"]>.1) ]
		sample = sample[ (sample['predicted_target']!=1) | ((sample['predicted_target']==1) & (sample[f'all_class1_std']<0.1)) ]
		lcplot_stats("sample_v7_",sample,model_name,plot=False)


		# v8
		# looser no probabilities all concentrated in the bottom
		# hard restric IIp
		sample = df_pred[ (df_pred[f"std_probs"]>.05) ]
		sample = sample[ (sample['predicted_target']!=1) | ((sample['predicted_target']==1) & (sample[f'all_class1_std']<0.1) & (sample[f'all_class1_median']<0.8)) ]
		lcplot_stats("sample_v8_",sample,model_name,plot=False)


		# v9
		# looser no probabilities all concentrated in the bottom
		# hard restric IIp, no IIns
		# harder cut on std
		sub_sample_list = []
		for i in range(0,7):
			sub_sample = df_pred[ (df_pred['predicted_target']==i) &  (df_pred[f"all_class{i}_std"]<0.2) & (df_pred[f"std_probs"]>.3)]
			sub_sample_list.append(sub_sample)
		sample = pd.concat(sub_sample_list, sort=True)
		sample = sample[ (sample[f"std_probs"]>.05) ]
		sample = sample[ (sample['predicted_target']!=1) | ((sample['predicted_target']==1) & (sample[f'all_class1_std']<0.1) & (sample[f'all_class1_median']<0.8)) ]
		sample = sample[sample['predicted_target']!=2]
		lcplot_stats("sample_v9_",sample,model_name,plot=False,save=True)
		
		