class simulation:
    def __init__(self, material, res, rad, e_model, ph_model, c_spin_ind, valley_list, temp, debug, v_spin_ind, basis, N_cutoff, nprocs):
        self.material = material
        self.res = res
        self.rad = rad
        self.temp = temp
        self.e_model = e_model
        if e_model == 'tb':
            self.c_ind = [1]
            self.v_ind = [0]
        self.ph_model = ph_model
        if ph_model == '1opt':
            self.phonon_modes = [0]
            self.phonon_couples = ['DP']
        self.c_spin_ind = c_spin_ind
        self.v_spin_ind = v_spin_ind
        self.debug = debug
        self.valley_list = valley_list
        self.basis = basis
        self.nprocs = nprocs
        self.init_dir = './inits/'+material+'_'+str(res)+'_'+e_model+'_'+ph_model
        valley_names = str(valley_list).replace('[', '').replace(']', '').replace("'", '').replace(' ', '_').replace(
            ',', '')
        #cspin = str(self.c_spin_ind).replace('[', '').replace(']', '').replace("'", '').replace(' ', '_').replace(',', '')
        #vspin = str(self.v_spin_ind).replace('[', '').replace(']', '').replace("'", '').replace(' ', '_').replace(',', '')
        if self.basis == 'kspace':
            self.N_cutoff = 0
            self.calc_dir = str(self.material) + '_' + str(self.res) + '_' + str(self.e_model) + '_' + str(
                self.ph_model) + '_rad_' + str(self.rad) + '_' + str(
                valley_names) + '_' + str(self.basis)
        if self.basis == 'exciton':
            self.N_cutoff = N_cutoff
            self.calc_dir = str(self.material) + '_' + str(self.res) + '_' + str(self.e_model) + '_' + str(
                self.ph_model) + '_rad_' + str(self.rad) + '_' + str(
                valley_names) + '_' + str(self.basis) + '_' + str(self.N_cutoff)

        self.exciton_dir = self.init_dir + '/rad_' + str(self.rad) + '_' + str(self.e_model) + '_' + str(
            self.ph_model) + '_' + str(valley_names)  # + '_' + str(self.N_cutoff)
    def printInfo(self):
        print("#### SIMULATION INFO ####")
        print('Material: ',self.material)
        print('resolution: ', self.res)
        print('truncation radius (2pi/a): ', self.rad)
        print('basis: ', self.basis)
        if self.basis == 'exciton':
            print('N cutoff: ', self.N_cutoff)
        print('e model: ', self.e_model)
        print('ph model: ', self.ph_model)
        print('temperature: ', self.temp)
        print('valleys: ', self.valley_list)
        print('e spin index: ', self.c_spin_ind)
        print('h spin index: ', self.v_spin_ind)
        print('init dir: ', self.init_dir)
        print('calc dir: ', self.calc_dir)
        print('debug: ', self.debug)
        print('nprocs: ', self.nprocs)
        print("#########################")
        return

def proc_inputfile(inputfile):
    with open(inputfile) as f:
        for line in f:
            exec(str(line),locals())
    sim = simulation(material=locals()['material'],\
                     res=locals()['res'],\
                     rad=locals()['rad'],\
                     e_model=locals()['e_model'],\
                     ph_model=locals()['ph_model'],\
                     c_spin_ind=locals()['c_spin_ind'],\
                     valley_list=locals()['valley_list'],\
                     temp=locals()['temp'],\
                     debug=locals()['debug'],\
                     v_spin_ind=locals()['v_spin_ind'],\
                     basis=locals()['basis'],\
                     N_cutoff=locals()['N_cutoff'],\
                     nprocs=locals()['nprocs'])
    return sim