import sys
import glob,os
import pandas as pd
from matplotlib import pyplot as plt
sys.path.append("../")
import utils.data_utils as du

# Init paths
path_raw = os.environ.get("DES_DATA")
path_dump = './dump/'

# load data and process if necessary
df_real = du.load_data(f"{path_raw}DESALL_forcePhoto_real_snana_fits/",f"{path_dump}/real/",redo_photometry = True,debug=False)
df_fake = du.load_data(f"{path_raw}/DESALL_forcePhoto_fake_snana_fits/",f"{path_dump}/fake/",redo_photometry = True,debug=False)

import ipdb; ipdb.set_trace()