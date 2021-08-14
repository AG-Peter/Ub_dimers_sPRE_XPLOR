#!/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor

import os, sys, re, glob, os, shutil, json, argparse
import datetime
import numpy as np

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        # self.print_help()
        sys.exit(2)

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

    if kwargs['testing']:
        test_main()
        sys.exit(0)
        return

    # load pdb
    import protocol
    if kwargs['struct_loading_method'] == 'loadPDB':
        protocol.loadPDB(kwargs['pdb'], deleteUnknownAtoms=True)
        protocol.initParams('protein')
    elif kwargs['struct_loading_method'] == 'initStruct':
        protocol.initStruct(kwargs['psf'])
        protocol.initCoords(kwargs['pdb'])
        protocol.initParams('protein')
    else:
        print(f"Unknown option for -struct_loading_method. Only 'loadPDB' and 'initStruct' are allowed. Uou provided {kwargs['struct_loading_method']}")
        sys.exit(2)

    
    if not 'psol_call_parameters_restraints' in kwargs and not 'rrp600_call_parameters_restraints' in kwargs and not 'rrp800_call_parameters_restraints' in kwargs:
        raise Exception("Provide ad least one .tbl file")
    
    from diffPotTools import readInRelaxData
    from relaxRatioPotTools import create_RelaxRatioPot
    
    out = []

    for mhz in ['600', '800']:
        if f'rrp{mhz}_call_parameters_restraints' in kwargs:
            relax_data_in = readInRelaxData(kwargs[f'rrp{mhz}_call_parameters_restraints'], pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
            rrp = create_RelaxRatioPot(name=kwargs[f'rrp{mhz}_call_parameters_name'],
                                       data_in=relax_data_in,
                                       freq=kwargs[f'rrp{mhz}_call_parameters_freq'],
                                       sel=kwargs[f'rrp{mhz}_call_parameters_sel'],
                                       inc_sel=kwargs[f'rrp{mhz}_call_parameters_inc_sel'],
                                       temperature=kwargs[f'rrp{mhz}_call_parameters_temperature'],
                                       optTemp=kwargs[f'rrp{mhz}_call_parameters_optTemp'],
                                       addAtoms=kwargs[f'rrp{mhz}_call_parameters_addAtoms'],
                                       mass=kwargs[f'rrp{mhz}_call_parameters_mass'],
                                       bond_type=kwargs[f'rrp{mhz}_call_parameters_bond_type'],
                                       sigmaFactor=kwargs[f'rrp{mhz}_call_parameters_sigmaFactor'],
                                       CSA=kwargs[f'rrp{mhz}_call_parameters_CSA'])

            prefix = f'{mhz}_set_parameters_'
            for key, value in kwargs.items():
                if key.startswith(prefix):
                    key = 'set' + key.replace(prefix, '')
                    if isinstance(value, bool) and value:
                        if key == 'setVerbose':
                            getattr(rrp, key)(True)
                        # getattr(rrp, key)()
                    else:
                        getattr(rrp, key)(value)

            for r in rrp.restraints():
                res_id = r.name().split()[1]
                value = r.calcd()
                out.append([f'rrp{mhz}', res_id, value])


    print(kwargs['psol_call_parameters_restraints'])
    if 'psol_call_parameters_restraints' in kwargs:
        from psolPotTools import create_PSolPot

        # create psol with call parameters
        psol = create_PSolPot(name=kwargs['psol_call_parameters_name'],
                              file=kwargs['psol_call_parameters_restraints'],
                              tauc=kwargs['psol_call_parameters_tauc'],
                              probeR=kwargs['psol_call_parameters_probeR'],
                              probeC=kwargs['psol_call_parameters_probeC'],
                              fixTauc=kwargs['psol_call_parameters_fixTauc'],
                              eSpinQuantumNumber=kwargs['psol_call_parameters_eSpinQuantumNumber'],
                              domainSelection=kwargs['psol_call_parameters_domainSelection'])

        # set parameters
        prefix = 'psol_set_parameters_'
        for key, value in kwargs.items():
            if key.startswith(prefix):
                key = 'set' + key.replace(prefix, '')
                if isinstance(value, bool) and value:
                    if key == 'setVerbose':
                        getattr(psol, key)(True)
                    # getattr(psol, key)() # currently not working
                else:
                    getattr(psol, key)(value)

        # get restraints
        for r in psol.restraints():
            res_id = r.name().split()[2]
            value = r.calcd()
            out.append(['psol', res_id, value])
        
    print(out)
    

def entrypoint():
    parser = MyParser(description="Get restraints from single file")
    parser.add_argument("-struct_loading_method", required=False, type=str, default="loadPDB",
                        help="""Sets the method with which the pdb/psf should be loaded.""")
    parser.add_argument('-pdb', metavar='<string>', required=True, type=str, help="The pdb file to run the sim on.")
    parser.add_argument('-psf', metavar='<string>', required=False, type=str, help="The psf file to run the sim on.")
    parser.add_argument('-testing', required=False, action='store_true',
                        help="Does not run XPLOR but uses np.random with seed 1 to produce predictable results.")
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
    parser.add_argument("-psol_call_parameters_probeC", required=False, type=float, default=0.24,
                        help="""probe concentration - units?""")
    parser.add_argument("-psol_call_parameters_fixTauc", required=False, type=str2bool, default="True",
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
    parser.add_argument("-psol_set_parameters_PhysicalAtoms", required=False, type=str2bool, default="True",
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
    parser.add_argument("-psol_set_parameters_ShowAllRestraints", required=False, type=str2bool, default="False",
                        help="""If set to True, the `showViolations()` method will show all restraints an not just the ones, violating temrs.""")
    parser.add_argument("-psol_set_parameters_HardExp", required=False, type=int, default=2,
                        help="""exponential for hard potType [2].""")
    parser.add_argument("-psol_set_parameters_TauC", required=False, type=float, default=0.2,
                        help="""Correlation Time""")
    parser.add_argument("-psol_set_parameters_FreqI", required=False, type=float, default=600,
                        help="""nuclear frequency  [MHz]""")
    parser.add_argument("-psol_set_parameters_Sqn", required=False, type=float, default=3.5,
                        help="""electron spin quantum number""")
    parser.add_argument("-psol_set_parameters_GammaI", required=False, type=float, default=26.752196,
                        help="""gyromagnetic ratio [10^7 Ts-1] of hydrogen.""")
    parser.add_argument("-psol_set_parameters_Rho0", required=False, type=float, default=0.24,
                        help="""conc. of paramagnetic solute [mM]""")
    parser.add_argument("-psol_set_parameters_FixTauc", required=False, type=str2bool, default=True,
                        help="""If True, fix tau_c. If False, parameters for tau_c optimization should be specified - this feature is not well tested.""")
    parser.add_argument("-psol_set_parameters_Verbose", required=False, type=str2bool, default="True",
                        help="""set to True for verbose output [True].""")
    parser.add_argument("-rrp600_call_parameters_name", required=False, type=str, default="rrp600", help="""This is the name of the potential term assigned to this PSolPot object
    It can contain any string and can be used to
    """)
    parser.add_argument("-rrp600_call_parameters_restraints", required=False, type=str, help="""The location of the spre_tble file that will be passed to XPLOR.
    The spre table file needs to be formatted as such:
    int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err
    So for example: For Ubiquitin the first three lines of that table look like this:
    2 1.401544 0.009619 12.445858 0.009619 0.779673 0.025396
    3 1.439783 0.016894 11.625145 0.016894 0.7907 0.032749
    4 1.442767 0.014602 12.005927 0.014602 0.759039 0.033966
    5 1.345446 0.011759 11.814099 0.011759 0.742896 0.028782
    """)
    parser.add_argument("-rrp600_call_parameters_freq", required=False, type=float, default=600,
                        help="""spectrometer frequency in MHz.""")
    parser.add_argument("-rrp600_call_parameters_sel", required=False, type=str, default="known and (not PSEUDO)", help="""an atomSel.AtomSel specifying atoms to define
    the molecular surface. This argument is not used
    the link_to argument is specified.
    """)
    parser.add_argument("-rrp600_call_parameters_inc_sel", required=False, type=str, default="known",
                        help="""an atomSel.AtomSel specifying the relaxation data to use for energy calculations, by atom.""")
    parser.add_argument("-rrp600_call_parameters_temperature", required=False, type=float, default=293,
                        help="""the nominal experimental temperature in Kelvins.""")
    parser.add_argument("-rrp600_call_parameters_optTemp", required=False, type=str2bool, default="False",
                        help="""whether or not to perform temperature optimization. [default: False]""")
    parser.add_argument("-rrp600_call_parameters_addAtoms", required=False, type=str2bool, default="False",
                        help="""synonym for optTemp - deprecated.""")
    parser.add_argument("-rrp600_call_parameters_mass", required=False, type=float, default=1000,
                        help="""mass of the pseudo atoms used for temperature optimization.""")
    parser.add_argument("-rrp600_call_parameters_bond_type", required=False, type=str, default="NH",
                        help="""the nuclei involved in relaxation. Currently only 'NH' is supported.""")
    parser.add_argument("-rrp600_call_parameters_sigmaFactor", required=False, type=float, default=10000000,
                        help="""value for the sigmaFactor of the RelaxRatioPot object.""")
    parser.add_argument("-rrp600_call_parameters_CSA", required=False, type=float, default=-160,
                        help="""value of Chemical Shielding tensor Anisotropy, in ppm""")
    parser.add_argument("-rrp600_set_parameters_ShowAllRestraints", required=False, type=str2bool, default="True", help="""boolean which changes the behavior of the
    violations printout. If this parameter is
    set to True,  all restraints are printed.
    Violated restraints are indicated by an
    asterisk in the first column.
    """)
    parser.add_argument("-rrp600_set_parameters_DiffTmpF", required=False, type=float, default=80315402327.6, help="""factor which accounts for viscosity and
    temperature of solvent, in units of 1/s.
    Default value corresponds to water at 293 K.
    See definition below.
    """)
    parser.add_argument("-rrp600_set_parameters_DiffShell", required=False, type=float, default=2.8, help="""hydration layer thickness, in Angstroms.
    Note: changes the shell thickness in
    surfTessellation object where the potential
    is linked to. Thus, if changed for the same
    surfTessellation object.
    """)
    parser.add_argument("-rrp600_set_parameters_MedianTmp", required=False, type=float, default=293,
                        help="""the nominal temperature.""")
    parser.add_argument("-rrp600_set_parameters_GyroR", required=False, type=float, default=9.869,
                        help="""value of ratio between gyro magnetic ratios of interacting nuclei.""")
    parser.add_argument("-rrp600_set_parameters_DipC2", required=False, type=float, default=1.1429, help="""[J^2/(s^2 T^2 m^6)]*[MHz^(-4)]. strength of the
    dipolar coupling interaction. Units of
    inverse nanoseconds.
    For a pair of a pair of NH nuclei
     dipC2=(mu_0*g_n*g_h*h_bar/(8*pi*r_NH^3))^2
    where
    magnetic permittivity of vacuum:
         mu_0  =  4*pi*1e-7       [H/m]
    hydrogen gyro magnetic ratio:
         g_n   =  2*pi*42.576*1e6 [Hz/T]
    15N gyro magnetic ratio:
         g_h   = -2*pi*4.3156*1e6 [Hz/T]
    Planck constant over 2*pi
         h_bar =  1.05457*1e-34   [J s]
    effective nuclei pair distance
         r_NH  =  1.0420 [A]
         (this is the value obtained from
          neutron diffraction scaled to
          account for librations)
    """)
    parser.add_argument("-rrp600_set_parameters_DiffRrmsd", required=False, type=float, default=0.5, help="""the value of rmsd, in Angstroms, separating
    current structure and the one that was used for
    tessellation of the protein surface last
    time which triggers new retessellation
    of the protein surface.
    """)
    parser.add_argument("-rrp600_set_parameters_DiffRstep", required=False, type=float, default=30, help="""the number of derivative calculation
    events which trigger a new retessellation of
    the protein surface.
    """)
    parser.add_argument("-rrp600_set_parameters_TempRange", required=False, type=float, default=0,
                        help="""the range of temperatures, in Kelvin, for temperature optimization.""")
    parser.add_argument("-rrp600_set_parameters_Freq", required=False, type=float, default=600,
                        help="""spectrometer frequency, in MHz.""")
    parser.add_argument("-rrp600_set_parameters_SigmaFactor", required=False, type=float, default=10000000,
                        help="""factor used in the computation of the cutoff value. See below.""")
    parser.add_argument("-rrp600_set_parameters_SoftExp", required=False, type=float, default=8.0,
                        help="""softExp parameter for asymptotic potential (see below).""")
    parser.add_argument("-rrp600_set_parameters_AsymSlope", required=False, type=float, default = 0.0, help = """asympSlope parameter for asymptotic potential (see below).""")
    parser.add_argument("-rrp600_set_parameters_Verbose", required=False, type=str2bool, default="True",
                        help="""verbose operation.""")
    parser.add_argument("-rrp800_call_parameters_name", required=False, type=str, default="rrp800", help="""This is the name of the potential term assigned to this PSolPot object
    It can contain any string and can be used to
    """)
    parser.add_argument("-rrp800_call_parameters_restraints", required=False, type=str, help="""The location of the spre_tble file that will be passed to XPLOR.
    The spre table file needs to be formatted as such:
    int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err
    So for example: For Ubiquitin the first three lines of that table look like this:
    2 1.401544 0.009619 12.445858 0.009619 0.779673 0.025396
    3 1.439783 0.016894 11.625145 0.016894 0.7907 0.032749
    4 1.442767 0.014602 12.005927 0.014602 0.759039 0.033966
    5 1.345446 0.011759 11.814099 0.011759 0.742896 0.028782
    """)
    parser.add_argument("-rrp800_call_parameters_freq", required=False, type=float, default=800,
                        help="""spectrometer frequency in MHz.""")
    parser.add_argument("-rrp800_call_parameters_sel", required=False, type=str, default="known and (not PSEUDO)", help="""an atomSel.AtomSel specifying atoms to define
    the molecular surface. This argument is not used
    the link_to argument is specified.
    """)
    parser.add_argument("-rrp800_call_parameters_inc_sel", required=False, type=str, default="known",
                        help="""an atomSel.AtomSel specifying the relaxation data to use for energy calculations, by atom.""")
    parser.add_argument("-rrp800_call_parameters_temperature", required=False, type=float, default=293,
                        help="""the nominal experimental temperature in Kelvins.""")
    parser.add_argument("-rrp800_call_parameters_optTemp", required=False, type=str2bool, default="False",
                        help="""whether or not to perform temperature optimization. [default: False]""")
    parser.add_argument("-rrp800_call_parameters_addAtoms", required=False, type=str2bool, default="False",
                        help="""synonym for optTemp - deprecated.""")
    parser.add_argument("-rrp800_call_parameters_mass", required=False, type=float, default=1000,
                        help="""mass of the pseudo atoms used for temperature optimization.""")
    parser.add_argument("-rrp800_call_parameters_bond_type", required=False, type=str, default="NH",
                        help="""the nuclei involved in relaxation. Currently only 'NH' is supported.""")
    parser.add_argument("-rrp800_call_parameters_sigmaFactor", required=False, type=float, default=10000000,
                        help="""value for the sigmaFactor of the RelaxRatioPot object.""")
    parser.add_argument("-rrp800_call_parameters_CSA", required=False, type=float, default=-160,
                        help="""value of Chemical Shielding tensor Anisotropy, in ppm""")
    parser.add_argument("-rrp800_set_parameters_ShowAllRestraints", required=False, type=str2bool, default="True", help="""boolean which changes the behavior of the
    violations printout. If this parameter is
    set to True,  all restraints are printed.
    Violated restraints are indicated by an
    asterisk in the first column.
    """)
    parser.add_argument("-rrp800_set_parameters_DiffTmpF", required=False, type=float, default=80315402327.6, help="""factor which accounts for viscosity and
    temperature of solvent, in units of 1/s.
    Default value corresponds to water at 293 K.
    See definition below.
    """)
    parser.add_argument("-rrp800_set_parameters_DiffShell", required=False, type=float, default=2.8, help="""hydration layer thickness, in Angstroms.
    Note: changes the shell thickness in
    surfTessellation object where the potential
    is linked to. Thus, if changed for the same
    surfTessellation object.
    """)
    parser.add_argument("-rrp800_set_parameters_MedianTmp", required=False, type=float, default=293,
                        help="""the nominal temperature.""")
    parser.add_argument("-rrp800_set_parameters_GyroR", required=False, type=float, default=9.869,
                        help="""value of ratio between gyro magnetic ratios of interacting nuclei.""")
    parser.add_argument("-rrp800_set_parameters_DipC2", required=False, type=float, default=1.1429, help="""[J^2/(s^2 T^2 m^6)]*[MHz^(-4)]. strength of the
    dipolar coupling interaction. Units of
    inverse nanoseconds.
    For a pair of a pair of NH nuclei
     dipC2=(mu_0*g_n*g_h*h_bar/(8*pi*r_NH^3))^2
    where
    magnetic permittivity of vacuum:
         mu_0  =  4*pi*1e-7       [H/m]
    hydrogen gyro magnetic ratio:
         g_n   =  2*pi*42.576*1e6 [Hz/T]
    15N gyro magnetic ratio:
         g_h   = -2*pi*4.3156*1e6 [Hz/T]
    Planck constant over 2*pi
         h_bar =  1.05457*1e-34   [J s]
    effective nuclei pair distance
         r_NH  =  1.0420 [A]
         (this is the value obtained from
          neutron diffraction scaled to
          account for librations)
    """)
    parser.add_argument("-rrp800_set_parameters_DiffRrmsd", required=False, type=float, default=0.5, help="""the value of rmsd, in Angstroms, separating
    current structure and the one that was used for
    tessellation of the protein surface last
    time which triggers new retessellation
    of the protein surface.
    """)
    parser.add_argument("-rrp800_set_parameters_DiffRstep", required=False, type=float, default=30, help="""the number of derivative calculation
    events which trigger a new retessellation of
    the protein surface.
    """)
    parser.add_argument("-rrp800_set_parameters_TempRange", required=False, type=float, default=0,
                        help="""the range of temperatures, in Kelvin, for temperature optimization.""")
    parser.add_argument("-rrp800_set_parameters_Freq", required=False, type=float, default=800,
                        help="""spectrometer frequency, in MHz.""")
    parser.add_argument("-rrp800_set_parameters_SigmaFactor", required=False, type=float, default=10000000,
                        help="""factor used in the computation of the cutoff value. See below.""")
    parser.add_argument("-rrp800_set_parameters_SoftExp", required=False, type=float, default=8.0,
                        help="""softExp parameter for asymptotic potential (see below).""")
    parser.add_argument("-rrp800_set_parameters_AsymSlope", required=False, type=float, default = 0.0, help = """asympSlope parameter for asymptotic potential (see below).""")
    parser.add_argument("-rrp800_set_parameters_Verbose", required=False, type=str2bool, default="True",
                        help="""verbose operation.""")
    args = vars(parser.parse_args())
    main(**args)

if __name__ == '__main__':
    entrypoint()