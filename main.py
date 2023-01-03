import sys
from input_proc import proc_inputfile
from materials import mat_dict
import initialize
import dynamics
import os
from shutil import copyfile
import time
import numpy as np
import debug

#np.random.seed(1234)

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print('Usage: python main.py [opts] inputfile')
        sys.exit()
    start_time = time.time()
    inputfile = args[-1]
    opt = args[0]
    #inputfile = './inputfile'
    sim = proc_inputfile(inputfile)
    sim.printInfo()
    mat = mat_dict[sim.material]
    mat.printInfo()
    if not(os.path.exists('./inits/')):
        os.mkdir('./inits')
    mat = initialize.initialize(mat, sim)
    sim = initialize.initialize_sim(sim)
    if sim.basis == 'kspace':
        sim = initialize.init_kspace_basis(sim, mat)
        sim = initialize.dynamics_init_kspace(sim)
        copyfile(inputfile, sim.calc_dir + '/inputfile')
        sim = dynamics.dynamics_kspace(sim, mat)
    if sim.basis == 'exciton':
        sim = initialize.init_exciton_basis(sim, mat)
        if opt != 'init':
            sim = initialize.dynamics_init_exciton(sim)
            if sim.trials != 0:
                copyfile(inputfile, sim.calc_dir + '/inputfile')
                sim = dynamics.dynamics_exciton(sim, mat)
    #debug.test(mat, sim)
    stop_time = time.time()
    print('Elapsed Time: ', np.round(stop_time - start_time, 3))