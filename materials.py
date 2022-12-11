import numpy as np

class material:
    def __init__(self,material,lattice_const, lambda_c, lambda_v, w, g_e_DP0, g_h_DP0, chi_2D):
        self.material = material
        self.lattice_const = lattice_const
        self.lambda_c = lambda_c
        self.lambda_v = lambda_v
        self.w = w
        self.g_e_DP0 = g_e_DP0
        self.g_h_DP0 = g_h_DP0
        self.lattice_vec_1 = 2 * np.pi / np.sqrt(3.0) * np.array([np.sqrt(3.0), -1.0])
        self.lattice_vec_2 = 4 * np.pi / np.sqrt(3.0) * np.array([0.0, 1.0])
        self.Brillouin_zone_area = np.cross(self.lattice_vec_1, self.lattice_vec_2)/ self.lattice_const ** 2
        self.chi_2D = chi_2D
    def printInfo(self):
        print("####  MATERIAL INFO  ####")
        print("lattice constant (bohr): " + str(self.lattice_const))
        print("lambda_c (eV): " + str(self.lambda_c))
        print("lambda_v (eV): " + str(self.lambda_v))
        print("Opt. Freq. Gamma (w) (therm): " + str(self.w))
        print("Def. Pot. 0th e-ph (therm): ", self.g_e_DP0)
        print("Def. Pot. 0th h-ph (therm): ", self.g_h_DP0)
        print("#########################")
        return

######
# lattice_const in units of Bohr
# lambda_c and lambda_v in units of eV
# w in units of thermal quantum (25 meV) therm_eV = 25.2488e-3  # eV thermal quantum at 293K
# g_e_DP0 in units of res * thermal quantum, need to divide by res in code !!!
MoS2 = material(material='MoS2',lattice_const=3.193/0.5292, lambda_c= -3.0e-3, lambda_v=148e-3, w=2.004, g_e_DP0=3.732, g_h_DP0=2.959, chi_2D=6.60/0.5292)
MoSe2 = material(material='MoSe2',lattice_const=3.313/0.5292, lambda_c=-31.0e-3, lambda_v = 184e-3, w=1.3598, g_e_DP0=3.189, g_h_DP0=3.005, chi_2D=8.23/0.5292)
WS2 = material(material='WS2',lattice_const=3.197/0.5292, lambda_c=26.0e-3, lambda_v=430e-3, w=1.85355, g_e_DP0=1.648, g_h_DP0=1.222, chi_2D=6.03/0.5292)
WSe2 = material(material='WSe2',lattice_const=3.310/0.5292, lambda_c=36.0e-3, lambda_v=466e-3, w=1.2159, g_e_DP0=1.285, g_h_DP0=1.732, chi_2D=7.18/0.5292)

mat_dict = {"MoS2": MoS2, "MoSe2": MoSe2, "WS2": WS2, "WSe2": WSe2}