import os
from preprocessing import skim_photometry as sk

if __name__ == '__main__':
	# Init paths
	path_des_data = os.environ.get("DES_DATA")
	raw_dir = path_des_data

	# Skim DES data
	# sk.skim_fits(raw_dir,dump_dir,debug)
