# %% Imports
import xplor

# %% make the tbl files
# xplor.functions.make_15_N_table('data/spre_and_relaxation_data_k6_k29/relaxation_file_ub2_k6.txt', out_file='/home/kevin/git/xplor_functions/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')
xplor.functions.write_argparse_lines_from_yaml()

# %%
xplor.functions.call_xplor_with_yaml('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb', from_tmp=True)
