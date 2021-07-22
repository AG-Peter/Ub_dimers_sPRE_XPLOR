#!/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor

import os, sys, re, glob, os, shutil, json, argparse
import datetime
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def test_main():
    greeting = ("Running in testing mode. I will now proceed to print outputs similar to a working script."
               " This is how to check for correct unraveling of outputs: The psol values are np.random.random()."
               " The rrp600 values are np.random.random() * 10 and the rrp800 np.random.random() * 10 + 10."
               " All values are created with a seed of 1."
               f" Current python executable is {os.path.dirname(sys.executable)}\n\n")
    
    greeting += """
   rrp600 : potential restraining NH relaxation rates ratio
  Assuming tumbling in water at 300 K with hydration layer 2.8 [A]
  Accepted 71 data points recorded at 600.0 MHz
  Initial Energy=4155.38 for initial Scale=1.00 


   rrp800 : potential restraining NH relaxation rates ratio
  Assuming tumbling in water at 300 K with hydration layer 2.8 [A]
  Accepted 71 data points recorded at 800.0 MHz
  Initial Energy=13009.65 for initial Scale=1.00 

SurfTessellation::retessellateIfNecessary: retessellating
findImportantAtoms: done
    """
    
    ubq_prolines = [19, 37, 38]
    np.random.seed(1)
    out = []
    
    # relax 600
    fake_values = np.random.random(76) * 10
    for res_id, fake_value in zip(range(1, 76), fake_values):
        if res_id not in ubq_prolines:
            out.append(['rrp600', res_id, fake_value])
            
    # relax 800
    fake_values = np.random.random(76) * 10 + 10
    for res_id, fake_value in zip(range(1, 76), fake_values):
        if res_id not in ubq_prolines:
            out.append(['rrp800', res_id, fake_value])
            
            
    # relax 600
    fake_values = np.random.random(76)
    for res_id, fake_value in zip(range(1, 76), fake_values):
        if res_id not in ubq_prolines:
            out.append(['psol', res_id, fake_value])
    
    print(greeting)
    print(out)
    

def main(**kwargs):
    print(os.path.dirname(sys.executable))

    for key, val in kwargs.items():
        print(key, val)

    if kwargs['testing']:
        test_main()
        sys.exit(0)
        return
    import protocol
    protocol.loadPDB(kwargs['pdb'], deleteUnknownAtoms=True)
    protocol.initParams('protein')
    
    if not 'psol_call_parameters_restraints' in kwargs and not 'rrp600_call_parameters_restraints' in kwargs and not 'rrp800_call_parameters_restraints' in kwargs:
        raise Exception("Provide ad least one .tbl file")
    
    from diffPotTools import readInRelaxData
    from relaxRatioPotTools import create_RelaxRatioPot
    
    out = []
    
    if 'rrp600_call_parameters_restraints' in kwargs:
        relax_data_in = readInRelaxData(relax_600_tbl, pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
        rrp600 = create_RelaxRatioPot('rrp600', data_in=relax_data_in, freq=600, temperature=300)

        for r in rrp600.restraints():
            res_id = r.name().split()[1]
            value = r.calcd()
            out.append(['rrp600', res_id, value])
            
    if 'rrp800_call_parameters_restraints' in kwargs:
        relax_data_in = readInRelaxData(relax_800_tbl, pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
        rrp800 = create_RelaxRatioPot('rrp800', data_in=relax_data_in, freq=800, temperature=300)

        for r in rrp800.restraints():
            res_id = r.name().split()[1]
            value = r.calcd()
            out.append(['rrp800', res_id, value])

    print(kwargs['psol_call_parameters_restraints'])
    if 'psol_call_parameters_restraints' in kwargs:
        from psolPotTools import create_PSolPot
        psol = create_PSolPot(name='psol', file=kwargs['psol_call_parameters_restraints'])
        # psol = create_PSolPot(name=kwargs['psol_call_parameters_name'],
        #                       file=kwargs['psol_call_parameters_restraints'],
        #                       tauc=kwargs['psol_call_parameters_tauc'],
        #                       probeR=kwargs['psol_call_parameters_probeR'],
        #                       probeC=kwargs['psol_call_parameters_probeC'],
        #                       fixTauc=kwargs['psol_call_parameters_fixTauc'],
        #                       eSpinQuantumNumber=kwargs['psol_call_parameters_eSpinQuantumNumber'],
        #                       domainSelection=kwargs['psol_call_parameters_domainSelection'])

        # prefix = 'psol_set_parameters_'
        # for key, value in kwargs.items():
        #     if key.startswith(prefix):
        #         key = 'set' + key.replace(prefix, '')
        #         if isinstance(value, bool) and value:
        #             pass
        #             # getattr(psol, key)()
        #         else:
        #             getattr(psol, key)(value)
        # getattr(psol, 'setFreqI')(600.0)
        # print(psol.freqI())
        # raise SystemExit(0)

        # psol.setScale(0.002)

        radius = 3.5
        # psol options
        psol.setRmin(0.8)
        #psol.setRadMax(30.0)
        #psol.setThkIni(1.0)
        #psol.setThkFac(1.4)
        #psol.setTcType("fix")
        psol.setTauC(0.2)
        #psol.setPconc(4.0)
        psol.setSqn(3.5)
        psol.setTargetType('correlation')
        psol.setRho0(0.24)
        psol.setFreqI(600)
        psol.tessellation().setVerbose(True)
        psol.setThreshold(0)
        psol.setProbeRadius(radius)
        psol.setRadiusOffset(radius)

        print(psol.restraints())
        for r in psol.restraints():
            res_id = r.name().split()[2]
            value = r.calcd()
            out.append(['psol', res_id, value])
        
    print(out)
    

def entrypoint():
    parser = argparse.ArgumentParser(description="Get restraints from single file")
    parser.add_argument('-pdb', metavar='<string>', required=True, type=str, help="The pdb file to run the sim on.")
    parser.add_argument("-psol_call_parameters_name", required=False, type=str, default="psol", help="""This is the name of the potential term assigned to this PSolPot object
    It can contain any string and can be used to
    """)
    parser.add_argument("-psol_call_parameters_restraints", required=False, type=str, help="""The location of the spre_tble file that will be passed to XPLOR.
    The spre table file needs to be formatted as such:
    f"assign (resid {resSeq:<2} and name HN)  {sPRE:5.3f} {err:5.3f}"
    So for example: For Ubiquitin the first three lines of that table look like this:
    assign (resid 2  and name HN)   5.510   0.711
    assign (resid 3  and name HN)   1.223   1.816
    assign (resid 4  and name HN)   4.381   0.402
    """)
    parser.add_argument("-psol_call_parameters_tauc", required=False, type=float, default=0.2,
                        help="""correlation time""")
    parser.add_argument("-psol_call_parameters_probeR", required=False, type=float, default=3.5,
                        help="""radius of probe molecule""")
    parser.add_argument("-psol_call_parameters_probeC", required=False, type=float, default=4.0,
                        help="""probe concentration - units?""")
    parser.add_argument("-psol_call_parameters_fixTauc", required=False, type=str2bool, default=True,
                        help="""whether to fix the value of tauc, or to let it float""")
    parser.add_argument("-psol_call_parameters_eSpinQuantumNumber", required=False, type=float, default=3.5,
                        help="""electron sping quantum number""")
    parser.add_argument("-psol_call_parameters_domainSelection", required=False, type=str,
                        default="known and not pseudo", help="""atoms to use in surface calculation""")
    parser.add_argument("-psol_set_parameters_RadiusNoise", required=False, type=float, default=10e-8, help="""small value added to each atomic radius to try to avoid
    numerical instabilities. Further, if a bad tessellation
    is detected, a different random value between
    0 and radiusNoise is used. [10^{-8}]
    """)
    parser.add_argument("-psol_set_parameters_ProbeRadius", required=False, type=float, default=3.5,
                        help="""Solute radius, used to generate tessellated surface for surface integration. [3.5 angstrom].""")
    parser.add_argument("-psol_set_parameters_RadiusOffset", required=False, type=float, default=3.5,
                        help="""Amount added to VdW radii given in radii for use in generating the tessellated surface  [3.5 angstrom].""")
    parser.add_argument("-psol_set_parameters_Rmin", required=False, type=float, default=0.8, help="""min distance from surface to nuclear center. The
    value actually used depends on physicalAtoms(). If the
    later is True the actual rmin used is the larger of rmin and
    """)
    parser.add_argument("-psol_set_parameters_PhysicalAtoms", required=False, type=str2bool, default=True,
                        help="""a boolean value. Please see the docs for the rmin accessor. [True]""")
    parser.add_argument("-psol_set_parameters_CenterOffset", required=False, type=float, default=0.0,
                        help="""center offset of solute - for off-center paramagnetic center [0. angstrom]""")
    parser.add_argument("-psol_set_parameters_TargetType", required=False, type=str, default="correlation", help="""Whether energy term targets differences in observed and
    calculated solvent PRE, or the correlation of observed to
    calculated: "gamma" or "correlation". ["correlation"]
    """)
    parser.add_argument("-psol_set_parameters_PotType", required=False, type=str, default="hard",
                        help="""energy form for the energy term: "soft" or "hard" ["hard].""")
    parser.add_argument("-psol_set_parameters_RlxType", required=False, type=str, default="r2dd", help="""specification of relaxation type: "r2dd" or "r1dd".
    This is used in computation of the constant prefactor,
    and has no effect for correlation target type.  ["r2dd"]
    """)
    parser.add_argument("-psol_set_parameters_FunType", required=False, type=str, default="harmonic", help="""takes a value of "square" or "harmonic". The former value
    results in zero energy in a range of obs()+/-err().
    ["harmonic"]
    """)
    parser.add_argument("-psol_set_parameters_SclType", required=False, type=str, default="const", help="""controls how the restraint energy is scaled: "none" for no
    scaling (weight=1), "sigma" for 1/err^2 scaling, and
    obs/obs_max/err^2 for "obsig" ["const"]
    """)
    parser.add_argument("-psol_set_parameters_ShowAllRestraints", required=False, type=str2bool, default=False,
                        help="""If set to True, the `showViolations()` method will show all restraints an not just the ones, violating temrs.""")
    parser.add_argument("-psol_set_parameters_HardExp", required=False, type=int, default=2,
                        help="""exponential for hard potType [2].""")
    parser.add_argument("-psol_set_parameters_TauC", required=False, type=float, default=0.2,
                        help="""Correlation Time""")
    parser.add_argument("-psol_set_parameters_FreqI", required=False, type=float, default=600,
                        help="""nuclear frequency  [MHz]""")
    parser.add_argument("-psol_set_parameters_Sqn", required=False, type=float, default=3.5,
                        help="""electron spin quantum number""")
    parser.add_argument("-psol_set_parameters_GammaI", required=False, type=float, default=1e7,
                        help="""gyromagnetic ratio [10^7 Ts-1]""")
    parser.add_argument("-psol_set_parameters_Rho0", required=False, type=float, default=4.0,
                        help="""conc. of paramagnetic solute [mM]""")
    parser.add_argument("-psol_set_parameters_FixTauc", required=False, type=str2bool, default=True,
                        help="""If True, fix tau_c. If False, parameters for tau_c optimization should be specified - this feature is not well tested.""")
    parser.add_argument("-psol_set_parameters_Verbose", required=False, type=str2bool, default=True,
                        help="""set to True for verbose output [True].""")

    parser.add_argument('-testing', required=False, action='store_true', help="Does not run XPLOR but uses np.random with seed 1 to produce predictable results.")
    args = vars(parser.parse_args())
    main(**args)

if __name__ == '__main__':
    entrypoint()