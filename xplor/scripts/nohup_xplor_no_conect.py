#!/usr/bin/env python

import xplor

xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], df_outdir=f"{Path(xplor.__file__).parent.parent}/data/values_from_every_frame/from_package_all/", suffix='_df_no_conect.csv', write_csv=True, fix_isopeptides=False, subsample=1)
