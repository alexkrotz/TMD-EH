import numpy as np
import matplotlib.pyplot as plt


def spin_diag(mat, sim):
    evals_out = np.zeros(len(mat))
    evecs_out = np.zeros(np.shape(mat), dtype=complex)
    num_blocks = len(sim.c_spin_ind) * len(sim.v_spin_ind)
    ran = np.arange(len(sim.ke_id))
    e1h1_ind = ran[(sim.se_id == 1) * (sim.sh_id == 1)]
    e0h1_ind = ran[(sim.se_id == 0) * (sim.sh_id == 1)]
    e1h0_ind = ran[(sim.se_id == 1) * (sim.sh_id == 0)]
    e0h0_ind = ran[(sim.se_id == 0) * (sim.sh_id == 0)]
    e1h1_mat = mat[e1h1_ind, :][:, e1h1_ind]
    e0h1_mat = mat[e0h1_ind, :][:, e0h1_ind]
    e1h0_mat = mat[e1h0_ind, :][:, e1h0_ind]
    e0h0_mat = mat[e0h0_ind, :][:, e0h0_ind]
    e1h1_evals, e1h1_evecs = np.linalg.eigh(e1h1_mat)
    e0h1_evals, e0h1_evecs = np.linalg.eigh(e0h1_mat)
    e1h0_evals, e1h0_evecs = np.linalg.eigh(e1h0_mat)
    e0h0_evals, e0h0_evecs = np.linalg.eigh(e0h0_mat)
    zero_mat = np.zeros_like(e1h1_evecs)
    #evecs_out = np.block([[e1h1_evecs, zero_mat, zero_mat, zero_mat], \
    #                      [zero_mat, e0h1_evecs, zero_mat, zero_mat], \
    #                      [zero_mat, zero_mat, e1h0_evecs, zero_mat], \
    #                      [zero_mat, zero_mat, zero_mat, e0h0_evecs]])
    evecs_out = (1/2) * np.block([[e1h1_evecs, e1h1_evecs, e1h1_evecs, e1h1_evecs],\
                                  [e1h0_evecs, -1.0*e1h0_evecs, e1h0_evecs, -1.0*e1h0_evecs],\
                                  [e0h1_evecs, e0h1_evecs, -1.0*e0h1_evecs, -1.0*e0h1_evecs],\
                                  [e0h0_evecs, -1.0*e0h0_evecs, -1.0*e0h0_evecs, e0h0_evecs]])
    evals_out = np.concatenate((e1h1_evals, e0h1_evals, e1h0_evals, e0h0_evals))

    return evals_out, evecs_out

    return