#!/usr/bin/env python

import xplor

xplor.functions.parallel_xplor(['k6', 'k11', 'k33'], df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/', suffix='_df.csv', parallel=False)
