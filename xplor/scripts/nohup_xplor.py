import xplor

xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], df_outdir=f"{Path(xplor.__file__).parent.parent}/data/values_from_every_frame/from_package_with_conect/", suffix='_df.csv', parallel=True)
