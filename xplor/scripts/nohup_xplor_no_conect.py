#!/usr/bin/env python

import xplor

xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/', suffix='_df_no_conect.csv', write_csv=True, fix_isopeptides=False)
