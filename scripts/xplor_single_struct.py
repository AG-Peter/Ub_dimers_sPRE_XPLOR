#!/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor

import os, sys, re, glob, os, shutil, json, argparse
import datetime
import numpy as np

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
    

def main(pdb, spre_tbl, relax_600_tbl, relax_800_tbl, testing=False):
    if testing:
        test_main()
        sys.exit(0)
        return
    import protocol
    protocol.loadPDB(pdb, deleteUnknownAtoms=True)
    protocol.initParams('protein')
    
    if not spre_tbl and not relax_600_tbl and not relax_800_tbl:
        raise Exception("Provide ad least one .tbl file")
    
    from diffPotTools import readInRelaxData
    from relaxRatioPotTools import create_RelaxRatioPot
    
    out = []
    
    if relax_600_tbl:
        relax_data_in = readInRelaxData(relax_600_tbl, pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
        rrp600 = create_RelaxRatioPot('rrp600', data_in=relax_data_in, freq=600, temperature=300)

        for r in rrp600.restraints():
            res_id = r.name().split()[1]
            value = r.calcd()
            out.append(['rrp600', res_id, value])
            
    if relax_800_tbl:
        relax_data_in = readInRelaxData(relax_800_tbl, pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
        rrp800 = create_RelaxRatioPot('rrp800', data_in=relax_data_in, freq=800, temperature=300)

        for r in rrp800.restraints():
            res_id = r.name().split()[1]
            value = r.calcd()
            out.append(['rrp800', res_id, value])
            
    if spre_tbl:
        from psolPotTools import create_PSolPot
        psol = create_PSolPot("psol", file=spre_tbl)
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

        for r in psol.restraints():
            res_id = r.name().split()[2]
            value = r.calcd()
            out.append(['psol', res_id, value])
        
    print(out)
    

def entrypoint():
    parser = argparse.ArgumentParser(description="Get restraints from single file")
    parser.add_argument('-pdb', metavar='<string>', required=True, type=str, help="The pdb file to run the sim on.")
    parser.add_argument('-spre_tbl', metavar='<string>', required=False, type=str, default='', help="Path to the restraints.tbl file for PSolPot.")
    parser.add_argument('-relax_600_tbl', metavar='<string>', required=False, type=str, default='', help="Path to the restraints.tbl file dor RelaxRatioPot with 600 MHz frequency.")
    parser.add_argument('-relax_800_tbl', metavar='<string>', required=False, type=str, default='', help="Path to the restraints.tbl file dor RelaxRatioPot with 800 MHz frequency.")
    parser.add_argument('-testing', required=False, action='store_true', help="Does not run XPLOR but uses np.random with seed 1 to produce predictable results.")
    args = vars(parser.parse_args())
    main(**args)

if __name__ == '__main__':
    entrypoint()