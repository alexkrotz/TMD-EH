import grid
import numpy as np
import time
import plots
import os
import sys
import itertools
from tqdm import tqdm
import constants
import operators
import ray
from numba import jit
import scipy.sparse as sparse
def initialize(mat, sim):
    print('Starting grid initialization...')
    start_time = time.time()
    if not(os.path.exists(sim.init_dir)):
        os.mkdir(sim.init_dir)

    k = grid.Monkhorst_Pack(sim.res)
    mat.d_k = np.linalg.norm(mat.lattice_vec_1) / sim.res / mat.lattice_const
    mat.int_fac = mat.Brillouin_zone_area / (sim.res ** 2) / (2 * np.pi) ** 2
    mat.g_e_DP0 = mat.g_e_DP0/sim.res
    mat.g_h_DP0 = mat.g_h_DP0/sim.res
    plots.kgrid_plot(k,sim)
    num_of_states = sim.res ** 2
    print('# of points in full grid: ', len(k))
    K_points_order = np.array([1, 3, 5, 0, 2, 4])
    K_points = 4.0 * np.pi / 3.0 * np.column_stack(
        (np.cos(K_points_order / 3. * np.pi), np.sin(K_points_order / 3. * np.pi)))
    K_points_tau = np.power(-1, K_points_order)
    found = 0
    for K in K_points:
        found += np.sum(np.all(np.round(k, 6) == np.round(K, 6), axis=1))
    if found == 0:
        print('Grid does not contain K points')
        print('Stopping calculation')
        sys.exit()
    else:
        print('Grid has K points')

    if sim.e_model == 'tb':
        from twobandmodel import two_band_noCurve, two_band_noCurve_new, two_band_noCurve_noBerry, two_band
        num_bands = 2
        num_spins = 2
        v_inds = [0]
        c_inds = [1]
        q, tau = grid.k_to_q(k, K_points, K_points_tau)
        eigvals = np.zeros((num_spins, len(k), num_bands))
        eigvecs = np.zeros((num_spins, len(k), num_bands, num_bands), dtype=complex)
        eigvals[0, :, 1], eigvals[0, :, 0], eigvecs[0, :, :, 1], eigvecs[0, :, :, 0] \
            = two_band(q, tau, -1, mat)
        eigvals[1, :, 1], eigvals[1, :, 0], eigvecs[1, :, :, 1], eigvecs[1, :, :, 0] \
            = two_band(q, tau, +1, mat)
        plots.tb_psi_plot(k, eigvecs, sim)


    if sim.ph_model == '1opt':
        num_phonon_modes = 1 # one optical mode
        num_spins = 2 #
        w_array = np.zeros((num_of_states, num_phonon_modes))
        w_array[:, 0] = mat.w

        g_array_DP = np.zeros((num_of_states, num_phonon_modes, num_of_states,\
                               num_bands, num_bands, num_spins, num_spins))
        g_array_DP[:, 0, :, c_inds, :] = mat.g_e_DP0 # scattering between conduction band states
        g_array_DP[:, 0, :, v_inds, :] = mat.g_h_DP0 # scattering between valence band states
    np.save(sim.init_dir + '/k.npy', k)
    np.save(sim.init_dir + '/w.npy', w_array)
    np.save(sim.init_dir + '/g_DP.npy', g_array_DP)
    np.save(sim.init_dir + '/bloch_bands.npy', eigvals)
    np.save(sim.init_dir + '/bloch_vecs.npy', eigvecs)
    stop_time = time.time()
    print('took', np.round(stop_time - start_time, 3), 'seconds')
    print('Finished grid initialization.')
    return mat

def get_bse_eigs(sim, mat):
    sim.b_se = (2 * sim.b_se_id - 1) / 2
    sim.b_sh = (2 * sim.b_sh_id - 1) / 2
    b_se = sim.b_se
    b_sh = sim.b_sh
    b_tau_e = sim.b_tau_e
    b_tau_h = sim.b_tau_h
    umk_point_list = sim.umk_point_list
    dk_id = sim.dk_id
    exciton_dir = sim.exciton_dir
    num_blocks = len(umk_point_list)
    block_dims = np.zeros(len(umk_point_list), dtype=int)
    for b_id in range(len(umk_point_list)):
        block_dims[b_id] = np.sum(dk_id == b_id)
    max_block_dim = np.max(block_dims)
    print('Looking for saved eigenstates...')
    if not (sim.debug) and os.path.exists(exciton_dir + '/eval_block_list.npy') and os.path.exists(
            exciton_dir + '/evec_block_list.npy') and os.path.exists(exciton_dir + '/dk_id_sort.npy'):
        print('Found saved eigenstates')
        eval_block_list = np.load(exciton_dir + '/eval_block_list.npy')
        evec_block_list = np.load(exciton_dir + '/evec_block_list.npy')
        dk_id_sort = np.load(exciton_dir + '/dk_id_sort.npy')
    else:
        print('Diagonalizing BSE Hamiltonian...')
        start_time = time.time()
        evec_block_list = np.zeros((max_block_dim, len(dk_id)), dtype=complex)  ## for storing eigenvectors
        eval_block_list = np.zeros((len(dk_id)))  ## for storing eigenvalues
        e_spin_block_list = np.zeros((len(dk_id)))
        h_spin_block_list = np.zeros((len(dk_id)))
        e_tau_block_list = np.zeros((len(dk_id)))
        h_tau_block_list = np.zeros((len(dk_id)))
        for dk_n in tqdm(range(len(umk_point_list))):
            bmat = operators.H_BSE_block(dk_n, sim, mat)  # H_bse_block(dk_n, block_inds, kappa_inds, params, sys_lists)
            se_block = b_se[dk_id == dk_n]
            sh_block = b_sh[dk_id == dk_n]
            te_block = b_tau_e[dk_id == dk_n]
            th_block = b_tau_h[dk_id == dk_n]

            evals, evecs = np.linalg.eigh(bmat)
            # e_spin = np.real(np.sum(np.conj(evecs)*np.einsum('i,ij->ji',se_block,evecs),axis=1))
            e_spin = np.real(np.sum(np.conj(evecs) * se_block[:, np.newaxis] * evecs, axis=0))
            e_tau = np.real(np.sum(np.conj(evecs) * te_block[:, np.newaxis] * evecs, axis=0))
            h_tau = np.real(np.sum(np.conj(evecs) * th_block[:, np.newaxis] * evecs, axis=0))
            # h_spin = np.real(np.sum(np.conj(evecs)*np.einsum('i,ij->ji',sh_block,evecs),axis=1))
            h_spin = np.real(np.sum(np.conj(evecs) * sh_block[:, np.newaxis] * evecs, axis=0))
            e_spin_block_list[dk_id == dk_n] = e_spin
            h_spin_block_list[dk_id == dk_n] = h_spin
            e_tau_block_list[dk_id == dk_n] = e_tau
            h_tau_block_list[dk_id == dk_n] = h_tau
            eval_block_list[dk_id == dk_n] = evals
            evec_block_list[:np.sum(dk_id == dk_n)][:, dk_id == dk_n] = evecs

        end_time = time.time()
        print("...took", np.round(end_time - start_time, 3), "seconds.")
        idx = eval_block_list.argsort()

        e_spin_block_list = e_spin_block_list[idx]
        h_spin_block_list = h_spin_block_list[idx]
        e_tau_block_list = e_tau_block_list[idx]
        h_tau_block_list = h_tau_block_list[idx]
        eval_block_list = eval_block_list[idx]
        evec_block_list = evec_block_list[:, idx]
        dk_id_sort = dk_id[idx]
        num_exciton_states = len(eval_block_list)
        np.save(exciton_dir + '/e_spin_block_list', e_spin_block_list)
        np.save(exciton_dir + '/h_spin_block_list', h_spin_block_list)
        np.save(exciton_dir + '/e_tau_block_list', e_tau_block_list)
        np.save(exciton_dir + '/h_tau_block_list', h_tau_block_list)
        np.save(exciton_dir + '/eval_block_list', eval_block_list)
        np.save(exciton_dir + '/evec_block_list', evec_block_list)
        np.save(exciton_dir + '/dk_id_sort', dk_id_sort)

    sim.eval_block_list = eval_block_list
    sim.evec_block_list = evec_block_list
    sim.dk_id_sort = dk_id_sort
    sim.dk_id = dk_id
    sim.block_dims = block_dims

    return sim

def initialize_sim(sim):
    print('Starting simulation initialization')
    start_time = time.time()
    if not(os.path.exists(sim.exciton_dir)):
        os.mkdir(sim.exciton_dir)
    q_trunc = sim.rad * 2 * np.pi

    K1_points_order = np.array(
        [2, 4, 6])  # K1 points are even numbered, tau at K1 is + 1, optical ground state for Espin = + 1
    K2_points_order = np.array([1, 3, 5])  # K2 points are even numbere,  tau at K2 is -1 for Espin = +1
    K_points_order = np.array([])
    for valley in sim.valley_list:
        if valley == 'K1':
            K_points_order = np.append(K_points_order, K1_points_order)
        if valley == 'K2':
            K_points_order = np.append(K_points_order, K2_points_order)
    K_points = 4.0 * np.pi / 3.0 * np.column_stack(
        (np.cos(K_points_order / 3. * np.pi), np.sin(K_points_order / 3. * np.pi)))
    K_points_only = np.copy(K_points)
    K_points_tau = np.power(-1, K_points_order)
    centers = np.sum(
        np.array(tuple(itertools.product(np.outer([-1, 0, 1], grid.lattice_vec_1), np.outer([-1, 0, 1], grid.lattice_vec_2)))),
        axis=1)
    k = np.load(sim.init_dir + '/k.npy')
    k_full = np.copy(k)
    w_list = np.load(sim.init_dir + '/w.npy')
    g_list_DP = np.load(sim.init_dir + '/g_DP.npy')
    bloch_bands_full = np.load(sim.init_dir + '/bloch_bands.npy')
    bloch_vecs_full = np.load(sim.init_dir + '/bloch_vecs.npy')
    q, tau = grid.k_to_q(k, K_points, K_points_tau)
    k_index = np.arange(0, len(k), dtype=int)
    if sim.rad != 0.0:
        trunc_mask = (np.apply_along_axis(np.linalg.norm, 1, q) < q_trunc)
        num_of_states = np.sum(trunc_mask)
        k = k[trunc_mask]
        q = q[trunc_mask]
        tau = tau[trunc_mask]
        bvec_list = bloch_vecs_full[:, trunc_mask, :, :]
        bands_list = bloch_bands_full[:, trunc_mask, :]
    else:
        trunc_mask = np.ones(len(k_index), dtype=int)
        bvec_list = bloch_vecs_full
        bands_list = bloch_bands_full
        num_of_states = len(k)

    num_exciton_states = len(sim.c_spin_ind) * len(sim.c_ind) * num_of_states * len(sim.v_spin_ind) * len(sim.v_ind) * num_of_states
    e_ids = np.array(list(itertools.product(sim.c_spin_ind, sim.c_ind, np.arange(num_of_states))))
    h_ids = np.array(list(itertools.product(sim.v_spin_ind, sim.v_ind, np.arange(num_of_states))))
    full_ids = np.array(list(itertools.product(e_ids, h_ids))).reshape((num_exciton_states, 6))
    se_id = full_ids[:, 0]
    c_band_id = full_ids[:, 1]
    ke_id = full_ids[:, 2]
    sh_id = full_ids[:, 3]
    v_band_id = full_ids[:, 4]
    kh_id = full_ids[:, 5]
    tau_e = tau[ke_id]
    tau_h = tau[kh_id]

    umk_point_mat = np.zeros((num_of_states, num_of_states, 2))
    for k1 in tqdm(range(len(k))):
        umk_point_mat[k1] = np.round(grid.k_Umklapp2(k[k1] - k, centers), 7) # wraps kappa points to first brillouin zone
    umk_point_list_full = grid.adjust_grid(umk_point_mat.reshape((-num_of_states ** 2, 2)),k_full) # wrapps kappa points to the monkhorst grid
    umk_point_mat = umk_point_list_full.reshape((num_of_states, num_of_states, 2))
    umk_point_list = np.unique(umk_point_list_full, axis=0)  # unique list of kappa points

    # find the index of each kappa point in the reduced (unique) list of kappa points
    umk_ind_mat = np.zeros((num_of_states, num_of_states), dtype=int)
    for k1 in tqdm(range(num_of_states)):
        umk_ind_mat[k1] = grid.find_index(umk_point_mat[k1], umk_point_list)
    # find the index of each unique kappa point in the original k grid
    kappa_point_inds = grid.find_index(umk_point_list, k_full).astype(int)
    tot_bands = np.concatenate((sim.v_ind, sim.c_ind))
    w_list = w_list[kappa_point_inds, :]
    g_list_DP = g_list_DP[kappa_point_inds, :, :, :, :][:, np.unique(sim.phonon_modes), :, :, :][:, :, trunc_mask, :, :][:, :, :, tot_bands, :][:,:,:, :, tot_bands]
    print('# of q-points:', len(umk_point_list))
    if os.path.exists(sim.exciton_dir + '/kappa_umk_point_mat.npy'):
        kappa_umk_point_mat = np.load(sim.exciton_dir + '/kappa_umk_point_mat.npy')
    else:
        kappa_umk_point_mat = np.zeros((len(umk_point_list), len(umk_point_list), 2))
        for k1n in tqdm(range(len(umk_point_list))):
            kappa_umk_point_mat[k1n] = grid.adjust_grid(np.round(grid.k_Umklapp2(umk_point_list[k1n] - umk_point_list, centers), 7),
                                               k_full)
        np.save(sim.exciton_dir + '/kappa_umk_point_mat.npy', kappa_umk_point_mat)
    if os.path.exists(sim.exciton_dir + '/kappa_umk_ind_mat.npy'):
        kappa_umk_ind_mat = np.load(sim.exciton_dir + '/kappa_umk_ind_mat.npy')
    else:
        kappa_umk_ind_mat = np.zeros((len(umk_point_list), len(umk_point_list)), dtype=int)
        for k1n in tqdm(range(len(kappa_umk_point_mat))):
            kappa_umk_ind_mat[k1n] = grid.find_index_trunc(kappa_umk_point_mat[k1n], umk_point_list)
        np.save(sim.exciton_dir + '/kappa_umk_ind_mat.npy', kappa_umk_ind_mat)
    # full_umk_mat is a kappa matrix of the full dimension
    full_umk_mat = umk_ind_mat
    for n in range(len(sim.c_ind) * len(sim.c_spin_ind) - 1):
        full_umk_mat = np.vstack((full_umk_mat, umk_ind_mat))
    v_full_umk_mat = full_umk_mat
    for n in range(len(sim.v_ind) * len(sim.v_spin_ind) - 1):
        full_umk_mat = np.hstack((full_umk_mat, v_full_umk_mat))

    dk_ind_mat = umk_ind_mat
    dk_point_mat = umk_point_list
    # dk_sum is a list that contains the number of terms (along a side) of each block
    dk_sum = np.zeros(len(umk_point_list), dtype=int)
    for k_n in range(len(umk_point_list)):
        dk_sum[k_n] = np.sum(full_umk_mat == k_n)
    # dk_id is an index for each block
    dk_id = np.zeros(np.sum(dk_sum), dtype=int)
    for k_s in range(len(dk_sum)):
        dk_id[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = np.zeros(dk_sum[k_s]) + k_s
    # be_id is an index for electronic states in each block where the index goes over the bands
    be_id = np.zeros(np.sum(dk_sum), dtype=int)
    be_id_1b = np.zeros(np.sum(dk_sum), dtype=int)
    b_c_band_id = np.zeros(np.sum(dk_sum), dtype=int)
    b_se_id = np.zeros(np.sum(dk_sum), dtype=int)
    for k_s in range(len(dk_sum)):
        be_id_1b[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = e_ids[np.where(full_umk_mat == k_s)[0], 2]
        b_c_band_id[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = e_ids[np.where(full_umk_mat == k_s)[0], 1]
        b_se_id[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = e_ids[np.where(full_umk_mat == k_s)[0], 0]
    bh_id = np.zeros(len(be_id), dtype=int)
    bh_id_1b = np.zeros(np.sum(dk_sum), dtype=int)
    b_v_band_id = np.zeros(np.sum(dk_sum), dtype=int)
    b_sh_id = np.zeros(np.sum(dk_sum), dtype=int)
    for k_s in range(len(dk_sum)):
        bh_id_1b[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = h_ids[np.where(full_umk_mat == k_s)[1], 2]
        b_v_band_id[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = h_ids[np.where(full_umk_mat == k_s)[1], 1]
        b_sh_id[np.sum(dk_sum[:k_s]):np.sum(dk_sum[:k_s + 1])] = h_ids[np.where(full_umk_mat == k_s)[1], 0]

    block_id_list = np.array([b_se_id, b_c_band_id, be_id_1b, b_sh_id, b_v_band_id, bh_id_1b]).transpose()

    # get singlet states from block_id_list

    # Here we only do the singlet states, triplet states can't couple without spin-orbit.e
    singlet_states = np.where(full_ids[:, 0] == full_ids[:, 3])  # full_ids[:,3])
    block_singlet_states = np.where(block_id_list[:, 0] == block_id_list[:, 3])  # block_id_list[:,3])

    ke_id = ke_id[singlet_states]
    c_band_id = c_band_id[singlet_states]
    se_id = se_id[singlet_states]
    kh_id = kh_id[singlet_states]
    v_band_id = v_band_id[singlet_states]
    sh_id = sh_id[singlet_states]
    tau_e = tau_e[singlet_states]
    tau_h = tau_h[singlet_states]
    dk_id = dk_id[block_singlet_states]
    be_id_1b = be_id_1b[block_singlet_states]
    bh_id_1b = bh_id_1b[block_singlet_states]
    b_c_band_id = b_c_band_id[block_singlet_states]
    b_v_band_id = b_v_band_id[block_singlet_states]
    b_se_id = b_se_id[block_singlet_states]
    b_sh_id = b_sh_id[block_singlet_states]

    block_id_list = np.array([b_se_id, b_c_band_id, be_id_1b, b_sh_id, b_v_band_id, bh_id_1b]).transpose()

    full_ids = full_ids[singlet_states]
    num_exciton_states = len(full_ids)
    ran = np.arange(num_exciton_states, dtype=int)
    block_sort = np.zeros((num_exciton_states), dtype=int)
    block_unsort = np.zeros((num_exciton_states), dtype=int)
    for b_n in range(len(block_id_list)):
        k_n = ran[np.all(full_ids == block_id_list[b_n], axis=1)]
        block_sort[b_n] = k_n
        block_unsort[k_n] = b_n

    b_tau_e = tau_e[block_sort]
    b_tau_h = tau_h[block_sort]


    sim.w_list = w_list
    sim.g_list_DP = g_list_DP
    sim.bands_list = bands_list
    sim.bvec_list = bvec_list
    sim.ke_id = ke_id
    sim.kh_id = kh_id
    sim.se_id = se_id
    sim.sh_id = sh_id
    sim.c_band_id = c_band_id
    sim.v_band_id = v_band_id
    sim.be_id_1b = be_id_1b
    sim.bh_id_1b = bh_id_1b
    sim.b_c_band_id = b_c_band_id
    sim.b_v_band_id = b_v_band_id
    sim.b_se_id = b_se_id
    sim.b_sh_id = b_sh_id
    sim.b_tau_e = b_tau_e
    sim.b_tau_h = b_tau_h
    sim.block_sort = block_sort
    sim.block_unsort = block_unsort
    sim.dk_id = dk_id
    sim.umk_ind_mat = umk_ind_mat
    sim.umk_point_list_MK= umk_point_list
    sim.umk_point_mat = grid.k_Umklapp(umk_point_list_full, centers).reshape(np.shape(umk_point_mat))#umk_point_mat
    sim.umk_point_list = grid.k_Umklapp(umk_point_list, centers) # wraps kappa points back to first BZ instead of Monkhorst
    sim.num_of_states = num_of_states
    sim.num_exciton_states = num_exciton_states
    sim.kappa_umk_ind_mat = kappa_umk_ind_mat



    stop_time = time.time()
    print('took', np.round(stop_time - start_time, 3), 'seconds')
    print('Finished simulation initialization')




    return sim

def get_Gab_inds_sparse(sim):
    ke_id = sim.ke_id
    se_id = sim.se_id
    c_band_id = sim.c_band_id
    kh_id = sim.kh_id
    v_band_id = sim.v_band_id
    sh_id = sim.sh_id
    bvec_list = sim.bvec_list
    block_unsort = sim.block_unsort
    num_exciton_states = len(ke_id)
    ran = np.arange(0, num_exciton_states)
    num_e_vals = 0
    num_h_vals = 0
    for n in range(0,num_exciton_states):
        se1 = se_id[n]
        ke1 = ke_id[n]
        cb1 = c_band_id[n]
        sh1 = sh_id[n]
        kh1 = kh_id[n]
        vb1 = v_band_id[n]
        #j_vals_e = ran[(kh_id[ran] == kh1)]  # find values where hole has remained the same
        #j_vals_h = ran[(ke_id[ran] == ke1)]  # find values where elec has remained the same
        j_vals_e = ran[((kh_id[ran] == kh1).astype(int) * (v_band_id[ran] == vb1).astype(int) * \
                        (sh_id[ran] == sh1).astype(int) * (se_id[ran] == se1).astype(int)).astype(bool)]  # find values where hole has remained the same
        j_vals_h = ran[((ke_id[ran] == ke1).astype(int) * (c_band_id[ran] == cb1).astype(int) * \
                        (se_id[ran] == se1).astype(int) * (sh_id[ran] == sh1)).astype(bool)]  # find values where elec has remained the same
        num_e_vals += len(j_vals_e)
        num_h_vals += len(j_vals_h)
    col_inds_e = np.zeros((num_e_vals),dtype=int)
    row_inds_e = np.zeros((num_e_vals),dtype=int)
    ke_mat_mels = np.ones((num_e_vals))
    bvec_e_mels = np.zeros((num_e_vals),dtype=complex)
    col_inds_h = np.zeros((num_h_vals),dtype=int)
    row_inds_h = np.zeros((num_h_vals),dtype=int)
    kh_mat_mels = np.ones((num_h_vals))
    bvec_h_mels = np.zeros((num_h_vals),dtype=complex)
    num_e_vals = 0
    num_h_vals = 0
    for n in range(0,num_exciton_states):
        se1 = se_id[n]
        ke1 = ke_id[n]
        cb1 = c_band_id[n]
        sh1 = sh_id[n]
        kh1 = kh_id[n]
        vb1 = v_band_id[n]
        #j_vals_e = ran[(kh_id[ran] == kh1)]  # find values where hole has remained the same
        #j_vals_h = ran[(ke_id[ran] == ke1)]  # find values where elec has remained the same
        j_vals_e = ran[((kh_id[ran] == kh1).astype(int) * (v_band_id[ran] == vb1).astype(int) * \
                        (sh_id[ran] == sh1).astype(int) * (se_id[ran] == se1).astype(int)).astype(bool)]  # find values where hole has remained the same
        j_vals_h = ran[((ke_id[ran] == ke1).astype(int) * (c_band_id[ran] == cb1).astype(int) * \
                        (se_id[ran] == se1).astype(int) * (sh_id[ran] == sh1)).astype(bool)]  # find values where elec has remained the same
        e_vals = len(j_vals_e)
        h_vals = len(j_vals_h)
        if e_vals > 0:
            col_inds_e[num_e_vals:(num_e_vals+e_vals)] = j_vals_e
            row_inds_e[num_e_vals:(num_e_vals+e_vals)] = n
            bvec_e_mels[num_e_vals:(num_e_vals+e_vals)] = operators.vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                 bvec_list[se_id[j_vals_e], ke_id[j_vals_e], :, c_band_id[j_vals_e]])
        if h_vals > 0:
            col_inds_h[num_h_vals:(num_h_vals+h_vals)] = n
            row_inds_h[num_h_vals:(num_h_vals+h_vals)] = j_vals_h
            bvec_h_mels[num_h_vals:(num_h_vals+h_vals)] = operators.vec_single_x_array(bvec_list[sh1, kh1, :, vb1].conj(),
                                                 bvec_list[sh_id[j_vals_h], kh_id[j_vals_h], :, v_band_id[j_vals_h]])
        num_e_vals += e_vals
        num_h_vals += h_vals
    bvec_e_block = sparse.coo_matrix((bvec_e_mels, (block_unsort[row_inds_e], block_unsort[col_inds_e])),\
                               shape=(num_exciton_states, num_exciton_states), dtype=complex)
    bvec_h_block = sparse.coo_matrix((bvec_h_mels, (block_unsort[row_inds_h], block_unsort[col_inds_h])),
                               shape=(num_exciton_states, num_exciton_states), dtype=complex)
    ke_mat_block = sparse.coo_matrix((ke_mat_mels, (block_unsort[row_inds_e], block_unsort[col_inds_e])),
                               shape=(num_exciton_states, num_exciton_states), dtype=int)
    kh_mat_block = sparse.coo_matrix((kh_mat_mels, (block_unsort[row_inds_h], block_unsort[col_inds_h])),
                               shape=(num_exciton_states, num_exciton_states), dtype=int)

    return ke_mat_block, kh_mat_block, bvec_h_block, bvec_e_block


def init_kspace_basis(sim, mat):
    if os.path.exists(sim.init_dir):
        print('Initializing ', sim.init_dir, ' for kspace basis.')
    else:
        print('init_dir not found')
    if not(os.path.exists(sim.exciton_dir)):
        os.mkdir(sim.exciton_dir)
    sim = get_bse_eigs(sim, mat)
    return sim

def init_exciton_basis(sim, mat):
    if os.path.exists(sim.init_dir):
        print('Initializing ',sim.init_dir,' for BSE basis.')
    else:
        print('init_dir not found')
    if not(os.path.exists(sim.exciton_dir)):
        os.mkdir(sim.exciton_dir)

    sim = get_bse_eigs(sim, mat)
    eval_block_list = sim.eval_block_list
    evec_block_list = sim.evec_block_list
    dk_id = sim.dk_id
    dk_id_sort = sim.dk_id_sort
    block_dims = sim.block_dims
    umk_point_list = sim.umk_point_list
    exciton_dir = sim.exciton_dir
    N_cutoff = sim.N_cutoff
    if N_cutoff == 0:
        E_cutoff =  0#eval_block_list[-1] - eval_block_list[N_cutoff]
    else:
        E_cutoff = eval_block_list[N_cutoff] - eval_block_list[0]
    E_cutoff_eV = E_cutoff * constants.therm_Ha / constants.eV_to_Hartree
    E_cutoff_therm = E_cutoff#E_cutoff_eV * eV_to_Hartree / therm_Ha
    print('Energy Cutoff: ', E_cutoff_therm, '(293 K)')
    print('Energy Cutoff: ', E_cutoff_eV, 'eV')
    if E_cutoff != 0:
        allow_states = np.where(np.round(eval_block_list,10) <= np.round(eval_block_list[0] + E_cutoff,10))
        allow_states = allow_states[0]#[:N_cutoff]
        print('Warning: adjusted N_cutoff to: ', len(allow_states))
        sim.N_cutoff = len(allow_states)
    else:
        allow_states = np.where(eval_block_list >= eval_block_list[0] + E_cutoff)

    allow_ran = range(np.min(allow_states), np.max(allow_states) + 1)
    num_trunc_states = len(np.arange(np.min(allow_states), np.max(allow_states) + 1))  # np.max(allow_states)
    print('Number of Exciton States: ', num_trunc_states)
    num_trunc_osc = len(umk_point_list)
    num_of_states = len(eval_block_list)
    pad_num = num_of_states ** 2 - num_trunc_states
    trunc_bse_evals = eval_block_list[allow_ran]
    trunc_bse_evecs = evec_block_list[:, allow_ran]


    @jit(nopython=True, fastmath=True)
    def get_evec(alpha):
        evec_block = np.ascontiguousarray(np.zeros(num_of_states)) + 0.0j
        dk_ran = dk_id == dk_id_sort[alpha]
        evec = trunc_bse_evecs[:, alpha][:block_dims[dk_id_sort[alpha]]]
        evec_block[dk_ran] = evec
        evec_out = evec_block[sim.block_unsort]
        return evec_out
    @jit(nopython=True, fastmath=True)
    def get_evec_block(alpha):
        dk_ran = dk_id == dk_id_sort[alpha]
        evec = trunc_bse_evecs[:, alpha][:block_dims[dk_id_sort[alpha]]]
        return evec, np.arange(len(dk_ran))[dk_ran]

    ke_mat_block_sparse, kh_mat_block_sparse, bvec_h_block_sparse, bvec_e_block_sparse = get_Gab_inds_sparse(sim)
    num_phonon_modes = len(sim.phonon_modes)
    block_sort = sim.block_sort
    ke_id = sim.ke_id
    kh_id = sim.kh_id
    c_band_id = sim.c_band_id
    v_band_id = sim.v_band_id
    se_id = sim.se_id
    sh_id = sim.sh_id
    G_ex_ph_DP = np.zeros((num_phonon_modes, num_trunc_states, num_trunc_states), dtype=complex)
    #G_ex_ph_F = np.zeros((num_phonon_modes, num_trunc_states, num_trunc_states), dtype=complex)
    #G_ex_ph_PZ = np.zeros((num_phonon_modes, num_trunc_states, num_trunc_states), dtype=complex)

    be_id = ke_id[block_sort]
    bh_id = kh_id[block_sort]
    b_c_band_id = c_band_id[block_sort]
    b_v_band_id = v_band_id[block_sort]
    bse_id = se_id[block_sort]
    bsh_id = sh_id[block_sort]

    # inds = (ke_mat_block_sparse, kh_mat_block_sparse, be_id, bh_id, b_c_band_id, b_v_band_id, bvec_e_block_sparse, bvec_h_block_sparse)
    inds = (ke_mat_block_sparse, kh_mat_block_sparse, be_id, bh_id, b_c_band_id, b_v_band_id, bse_id, bsh_id,
            bvec_e_block_sparse,
            bvec_h_block_sparse)

    ray_inds = ray.put(inds)
    g_inds = (sim.g_list_DP)
    g_inds_ray = ray.put(g_inds)

    def get_ran(coo_mat, row_ran, col_ran): #accepts sparse matrix and dk_rans returns dense matrix in dk_rans
        row = coo_mat.row
        col = coo_mat.col
        dat = coo_mat.data
        min_col = (np.min(col_ran)).astype(int)
        max_col = (np.max(col_ran)).astype(int)
        min_row = (np.min(row_ran)).astype(int)
        max_row = (np.max(row_ran)).astype(int)
        max_row_bool = (row <= max_row).astype(int)
        min_row_bool = (row >= min_row).astype(int)
        max_col_bool = (col <= max_col).astype(int)
        min_col_bool = (col >= min_col).astype(int)
        tot_int = (max_row_bool * min_row_bool * max_col_bool * min_col_bool).astype(bool)
        row_out = row[tot_int]
        col_out = col[tot_int]
        dat_out = dat[tot_int]
        if len(row_out) != 0:
            row_out = row_out - min_row
            col_out = col_out - min_col
        #coo_mat_out = sparse.coo_matrix((dat_out,(row_out,col_out)),shape=(len(row_ran),len(col_ran)),dtype=dat.dtype)
        #return coo_mat_out.toarray()
        coo_mat_out = np.zeros((len(row_ran), len(col_ran)), dtype=dat.dtype)
        coo_mat_out[(row_out, col_out)] = dat_out
        return coo_mat_out
    def get_Gab3(g,evec_a,evec_b,kap_id, inds):
        if kap_id == -1:
            return 0
        else:
            ke_mat_block_ran, kh_mat_block_ran, be_id_ran, bh_id_ran, b_c_band_id_ran_1, b_v_band_id_ran_1,\
                b_c_band_id_ran_2, b_v_band_id_ran_2, bse_id_ran_1, bsh_id_ran_1, bse_id_ran_2, bsh_id_ran_2,\
                    bvec_e_block_ran, bvec_h_block_ran= inds
            e1_block, e2_block = np.where(ke_mat_block_ran == 1)
            h1_block, h2_block = np.where(kh_mat_block_ran == 1)
            g_e_mat = np.zeros_like(ke_mat_block_ran, dtype=complex)
            g_h_mat = np.zeros_like(kh_mat_block_ran, dtype=complex)
            g_e_mat[(e1_block, e2_block)] = g[kap_id, be_id_ran[e1_block], b_c_band_id_ran_1[e1_block],\
                                              b_c_band_id_ran_2[e2_block], bse_id_ran_1[e1_block], bse_id_ran_2[e2_block]]
            g_h_mat[(h1_block, h2_block)] = g[kap_id, bh_id_ran[h2_block], b_v_band_id_ran_2[h2_block],\
                                              b_v_band_id_ran_1[h1_block], bsh_id_ran_2[h2_block], bsh_id_ran_1[h1_block]]
            prod_mat_e = g_e_mat * ke_mat_block_ran * bvec_e_block_ran
            prod_mat_h = g_h_mat * kh_mat_block_ran * bvec_h_block_ran
            tot = (np.dot(np.conj(evec_a), np.matmul(prod_mat_e, evec_b.reshape((-1, 1))))) + \
                (np.dot(np.conj(evec_a), np.matmul(prod_mat_h, evec_b.reshape((-1, 1)))))
            return tot

    ray.init(ignore_reinit_error=True, _temp_dir='/tmp/ray_tmp/')

    @ray.remote
    def get_G_mels(ind):
        inds = ray.get(ray_inds)
        (ke_mat_block_sparse, kh_mat_block_sparse, be_id, bh_id, b_c_band_id, b_v_band_id, bse_id, bsh_id,
         bvec_e_block_sparse, bvec_h_block_sparse) = inds
        g_inds = ray.get(g_inds_ray)
        g_list_DP = g_inds
        n = ind[0]  # abn_list[ind][0]
        alpha_n = ind[1]  # abn_list[ind][1]
        beta_n = ind[2]  # abn_list[ind][2]
        evec_a, dk_ran_a = get_evec_block(alpha_n)
        evec_b, dk_ran_b = get_evec_block(beta_n)
        kap_id = sim.kappa_umk_ind_mat[dk_id_sort[alpha_n]][dk_id_sort[beta_n]]
        if kap_id == -1:
            return 0, n, alpha_n, beta_n
        else:
            ke_mat_block_ran = get_ran(ke_mat_block_sparse, dk_ran_a,
                                       dk_ran_b)  # ke_mat_block[dk_ran_a][:,dk_ran_b]#ke_mat_block_sparse.todense()[dk_ran_a][:,dk_ran_b]#
            kh_mat_block_ran = get_ran(kh_mat_block_sparse, dk_ran_a,
                                       dk_ran_b)  # kh_mat_block[dk_ran_a][:,dk_ran_b]#kh_mat_block_sparse.todense()[dk_ran_a][:,dk_ran_b]#
            be_id_ran = be_id[dk_ran_a]
            bh_id_ran = bh_id[dk_ran_b]
            b_c_band_id_ran_1 = b_c_band_id[dk_ran_a]
            b_c_band_id_ran_2 = b_c_band_id[dk_ran_b]
            b_v_band_id_ran_1 = b_v_band_id[dk_ran_a]
            b_v_band_id_ran_2 = b_v_band_id[dk_ran_b]
            bse_id_ran_1 = bse_id[dk_ran_a]
            bse_id_ran_2 = bse_id[dk_ran_b]
            bsh_id_ran_1 = bsh_id[dk_ran_a]
            bsh_id_ran_2 = bsh_id[dk_ran_b]
            bvec_e_block_ran = get_ran(bvec_e_block_sparse, dk_ran_a,
                                       dk_ran_b)  # bvec_e_block[dk_ran_a][:,dk_ran_b]#bvec_e_block_sparse.todense()[dk_ran_a][:,dk_ran_b]#
            bvec_h_block_ran = get_ran(bvec_h_block_sparse, dk_ran_a,
                                       dk_ran_b)  # bvec_h_block[dk_ran_a][:,dk_ran_b]#bvec_h_block_sparse.todense()[dk_ran_a][:,dk_ran_b]#
            inds_out = (ke_mat_block_ran, kh_mat_block_ran, be_id_ran, bh_id_ran, b_c_band_id_ran_1, b_v_band_id_ran_1,
                        b_c_band_id_ran_2, b_v_band_id_ran_2, bse_id_ran_1, bsh_id_ran_1, bse_id_ran_2, bsh_id_ran_2,
                        bvec_e_block_ran, bvec_h_block_ran)
            G_mel_DP = get_Gab3(g_list_DP[:, sim.phonon_modes[n], :, :, :, :, :], evec_a, evec_b, kap_id, inds_out)
            #G_mel_F = G_mel_DP
            #G_mel_PZ = G_mel_DP
            # TODO ONLY DOES DP!!
            # G_mel_F = get_Gab3(g_list_F[:, phonon_modes[n], :, :, :, :, :], evec_a, evec_b, kap_id, inds_out)
            # G_mel_PZ = get_Gab3(g_list_DP[:, phonon_modes[n], :, :, :, :, :], evec_a, evec_b, kap_id, inds_out)
            return G_mel_DP, n, alpha_n, beta_n

    if not(sim.debug) and os.path.exists(exciton_dir+'/G_ex_DP.npy'):# and os.path.exists(exciton_dir+'/G_ex_F.npy') and os.path.exists(exciton_dir+'/G_ex_PZ.npy'):
        print('Found Exciton-Phonon coupling ')
        G_ex_ph_DP = np.load(exciton_dir+'/G_ex_DP.npy')
        #G_ex_ph_F = np.load(exciton_dir+'/G_ex_F.npy')
        #G_ex_ph_PZ = np.load(exciton_dir+'/G_ex_PZ.npy')
        Ex_kap_ind_mat = np.load(exciton_dir+'/Ex_kap_ind_mat.npy')
        if len(Ex_kap_ind_mat) >= sim.N_cutoff:
            Ex_kap_ind_mat= Ex_kap_ind_mat[0:sim.N_cutoff, 0:sim.N_cutoff]
            G_ex_ph_DP = G_ex_ph_DP[:,0:sim.N_cutoff, 0:sim.N_cutoff]
        else:
            print('Adding Exciton-Phonon couplings ')
            start_time = time.time()
            need_states = len(Ex_kap_ind_mat) + np.arange(sim.N_cutoff - len(Ex_kap_ind_mat))
            abn_list_1 = np.array(list(itertools.product(sim.phonon_modes, need_states, np.arange(num_trunc_states))))
            abn_list_2 = np.array(list(itertools.product(sim.phonon_modes, np.arange(num_trunc_states), need_states)))
            abn_list = np.vstack((abn_list_1, abn_list_2))
            ncalcs = int(len(abn_list) / sim.nprocs) + 1
            new_G_ex_ph_DP = np.zeros((num_phonon_modes, num_trunc_states, num_trunc_states), dtype=complex)
            new_G_ex_ph_DP[:,0:len(Ex_kap_ind_mat),0:len(Ex_kap_ind_mat)] = G_ex_ph_DP
            del G_ex_ph_DP
            G_ex_ph_DP = new_G_ex_ph_DP
            for n in tqdm(range(ncalcs)):
                rans = n * sim.nprocs + np.arange(sim.nprocs)
                rans = rans[rans < len(abn_list)]
                results = [get_G_mels.remote(abn_list[i]) for i in rans]
                m = 0
                for r in results:
                    G_mel_DP, ph_n, alpha_n, beta_n = ray.get(r)
                    G_ex_ph_DP[ph_n, alpha_n, beta_n] = G_mel_DP
                    # G_ex_ph_F[ph_n, alpha_n, beta_n] = G_mel_F
                    # G_ex_ph_PZ[ph_n, alpha_n, beta_n] = G_mel_PZ
                    m += 1
            end_time = time.time()
            print('took ', np.round(end_time - start_time, 3))

            np.save(exciton_dir + '/G_ex_DP', G_ex_ph_DP)
            # np.save(exciton_dir+'/G_ex_F',G_ex_ph_F)
            # np.save(exciton_dir+'/G_ex_PZ',G_ex_ph_PZ)
            new_Ex_kap_ind_mat = np.zeros((num_trunc_states, num_trunc_states), dtype=int)
            new_Ex_kap_ind_mat[0:len(Ex_kap_ind_mat),0:len(Ex_kap_ind_mat)] = Ex_kap_ind_mat
            del Ex_kap_ind_mat
            Ex_kap_ind_mat = new_Ex_kap_ind_mat
            for n in range(len(abn_list)):
                alpha_n = abn_list[n][1]
                beta_n = abn_list[n][2]  # Ex_kap_ind_mat[beta_n, alpha_n]
                Ex_kap_ind_mat[alpha_n, beta_n] = sim.kappa_umk_ind_mat[dk_id_sort[alpha_n]][dk_id_sort[beta_n]]
            np.save(exciton_dir + '/Ex_kap_ind_mat', Ex_kap_ind_mat)





    else:
        print('Generating Exciton-Phonon coupling...')
        start_time = time.time()
        abn_list = np.array(
            list(itertools.product(sim.phonon_modes, np.arange(num_trunc_states), np.arange(num_trunc_states))))
        ncalcs = int(len(abn_list) / sim.nprocs) + 1
        for n in tqdm(range(ncalcs)):
            rans = n * sim.nprocs + np.arange(sim.nprocs)
            rans = rans[rans < len(abn_list)]
            results = [get_G_mels.remote(abn_list[i]) for i in rans]
            m = 0
            for r in results:
                G_mel_DP, ph_n, alpha_n, beta_n = ray.get(r)
                G_ex_ph_DP[ph_n, alpha_n, beta_n] = G_mel_DP
                #G_ex_ph_F[ph_n, alpha_n, beta_n] = G_mel_F
                #G_ex_ph_PZ[ph_n, alpha_n, beta_n] = G_mel_PZ
                m += 1
        end_time = time.time()
        print('took ',np.round(end_time-start_time,3))

        np.save(exciton_dir+'/G_ex_DP',G_ex_ph_DP)
        #np.save(exciton_dir+'/G_ex_F',G_ex_ph_F)
        #np.save(exciton_dir+'/G_ex_PZ',G_ex_ph_PZ)
        Ex_kap_ind_mat = np.zeros((num_trunc_states, num_trunc_states), dtype=int)
        for n in range(len(abn_list)):
            alpha_n = abn_list[n][1]
            beta_n = abn_list[n][2] #Ex_kap_ind_mat[beta_n, alpha_n]
            Ex_kap_ind_mat[alpha_n, beta_n] = sim.kappa_umk_ind_mat[dk_id_sort[alpha_n]][dk_id_sort[beta_n]]
        np.save(exciton_dir+'/Ex_kap_ind_mat',Ex_kap_ind_mat)
    kap_pos = np.where(Ex_kap_ind_mat != -1)
    unique_kap_inds = np.unique(Ex_kap_ind_mat[kap_pos].flatten())
    kap_inds = np.arange(len(unique_kap_inds))
    red_ex_kap_ind_mat = np.zeros((num_trunc_states, num_trunc_states), dtype=int) - 1
    n = 0
    for kap_ind in unique_kap_inds:
        pos = np.where(Ex_kap_ind_mat == kap_ind)
        red_ex_kap_ind_mat[pos] = kap_inds[n]
        n += 1
    print('Number of Phonons: ', len(unique_kap_inds))
    exciton_kappa_inds = (
        G_ex_ph_DP, red_ex_kap_ind_mat, unique_kap_inds)  # Ex_kap_ind_mat)
    sim.Ex_kap_ind_mat = Ex_kap_ind_mat
    sim.G_ex_ph_DP = G_ex_ph_DP
    sim.red_ex_kap_ind_mat = red_ex_kap_ind_mat
    sim.unique_kap_inds = unique_kap_inds
    sim.trunc_bse_evecs = trunc_bse_evecs
    sim.trunc_bse_evals = trunc_bse_evals
    sim.bse_evals = eval_block_list
    sim.e_spin_block_list = np.load(exciton_dir + '/e_spin_block_list.npy')
    sim.h_spin_block_list = np.load(exciton_dir + '/h_spin_block_list.npy')
    sim.trunc_e_spin_block_list = sim.e_spin_block_list[allow_ran]
    sim.trunc_h_spin_block_list = sim.h_spin_block_list[allow_ran]
    #np.save(exciton_dir + '/e_spin_block_list', e_spin_block_list)
    #np.save(exciton_dir + '/h_spin_block_list', h_spin_block_list)
    #np.save(exciton_dir + '/e_tau_block_list', e_tau_block_list)
    #np.save(exciton_dir + '/h_tau_block_list', h_tau_block_list)
    #np.save(exciton_dir + '/eval_block_list', eval_block_list)
    #np.save(exciton_dir + '/evec_block_list', evec_block_list)
    #np.save(exciton_dir + '/dk_id_sort', dk_id_sort)
    return sim


def dynamics_init_kspace(sim):
    if not(os.path.exists(sim.calc_dir)):
        os.mkdir(sim.calc_dir)
    foldername = sim.calc_dir
    phonon_modes = sim.phonon_modes
    phonon_couples = sim.phonon_couples
    dq_mel_list_full = list(np.zeros((len(phonon_modes)), dtype=complex))
    dq_mel_list_ind_full = list(np.zeros((len(phonon_modes)), dtype=int))
    dp_mel_list_full = list(np.zeros((len(phonon_modes)), dtype=complex))
    dp_mel_list_ind_full = list(np.zeros((len(phonon_modes)), dtype=int))
    unique_ph_modes = np.unique(phonon_modes)
    for n in range(len(phonon_modes)):
        mode_pos = np.where(unique_ph_modes == phonon_modes[n])[0][0]
        mode_ind = unique_ph_modes[mode_pos]
        couple_type = phonon_couples[n]
        if not(sim.debug) and os.path.exists(foldername + '/dq_mel_list_' + str(mode_ind) + '_' + couple_type + '.npy') and \
                os.path.exists(foldername + '/dq_mel_list_ind_' + str(mode_ind) + '_' + couple_type + '.npy') and \
                os.path.exists(foldername + '/dp_mel_list_' + str(mode_ind) + '_' + couple_type + '.npy') and \
                os.path.exists(foldername + '/dp_mel_list_ind_' + str(mode_ind) + '_' + couple_type + '.npy'):
            print('Found derrivative matrices for mode ', mode_ind, ' Coupling type: ', couple_type)
            dq_mel_list_full[n] = list(
                np.load(foldername + '/dq_mel_list_' + str(mode_ind) + '_' + couple_type + '.npy', allow_pickle=True))
            dq_mel_list_ind_full[n] = list(
                np.load(foldername + '/dq_mel_list_ind_' + str(mode_ind) + '_' + couple_type + '.npy',
                        allow_pickle=True))
            dp_mel_list_full[n] = list(
                np.load(foldername + '/dp_mel_list_' + str(mode_ind) + '_' + couple_type + '.npy', allow_pickle=True))
            dp_mel_list_ind_full[n] = list(
                np.load(foldername + '/dp_mel_list_ind_' + str(mode_ind) + '_' + couple_type + '.npy',
                        allow_pickle=True))
        else:
            print('Generating derrivative matrices for mode ', mode_ind, ' Coupling type: ', couple_type)
            if couple_type == 'DP':
                bloch_mat, kappa_ind_mat = operators.gen_bk_mat(sim.g_list_DP[:, mode_pos, :, :, :], sim)  # gen_bk_mat_holstein(mode_n)
                dq_mel_list, dq_mel_list_ind, dp_mel_list, dp_mel_list_ind = operators.der_mat_gen(bloch_mat, kappa_ind_mat,
                                                                                         sim.w_list[:, mode_pos],
                                                                                         sim)
                dq_mel_list_full[n] = dq_mel_list
                dq_mel_list_ind_full[n] = dq_mel_list_ind
                dp_mel_list_full[n] = dp_mel_list
                dp_mel_list_ind_full[n] = dp_mel_list_ind
                np.save(foldername + '/dq_mel_list_' + str(mode_ind) + '_' + couple_type + '.npy',
                        np.array(dq_mel_list, dtype=object), allow_pickle=True)
                np.save(foldername + '/dq_mel_list_ind_' + str(mode_ind) + '_' + couple_type + '.npy',
                        np.array(dq_mel_list_ind, dtype=object), allow_pickle=True)
                np.save(foldername + '/dp_mel_list_' + str(mode_ind) + '_' + couple_type + '.npy',
                        np.array(dp_mel_list, dtype=object), allow_pickle=True)
                np.save(foldername + '/dp_mel_list_ind_' + str(mode_ind) + '_' + couple_type + '.npy',
                        np.array(dp_mel_list_ind, dtype=object), allow_pickle=True)
    dq_mel_list_full = ray.put(dq_mel_list_full)
    dq_mel_list_ind_full = ray.put(dq_mel_list_ind_full)
    dp_mel_list_full = ray.put(dp_mel_list_full)
    dp_mel_list_ind_full = ray.put(dp_mel_list_ind_full)
    sim.rayvars = (dq_mel_list_full, dq_mel_list_ind_full, dp_mel_list_full, dp_mel_list_ind_full)
    sim.filename = foldername + '/' +foldername
    return sim

def dynamics_init_exciton(sim):
    if not(os.path.exists(sim.calc_dir)):
        os.mkdir(sim.calc_dir)
    foldername = sim.calc_dir
    phonon_modes = sim.phonon_modes
    w_list = sim.w_list
    Ex_kappa_ind_mat = sim.Ex_kap_ind_mat
    phonon_couples = sim.phonon_couples
    dp_mat = np.zeros((len(phonon_modes), np.shape(w_list)[0], len(Ex_kappa_ind_mat), len(Ex_kappa_ind_mat)),
                      dtype=complex)
    dq_mat = np.zeros((len(phonon_modes), np.shape(w_list)[0], len(Ex_kappa_ind_mat), len(Ex_kappa_ind_mat)),
                      dtype=complex)
    unique_ph_modes = np.unique(phonon_modes)
    for n in range(len(phonon_modes)):
        mode_pos = np.where(unique_ph_modes == phonon_modes[n])[0][0]
        couple_type = phonon_couples[n]
        if couple_type == 'DP':
            dq_mat_term, dp_mat_term = operators.der_mat_gen_Ex(sim.G_ex_ph_DP[mode_pos, :, :], w_list[:, mode_pos], Ex_kappa_ind_mat)
            dp_mat[mode_pos, :, :, :] += dp_mat_term
            dq_mat[mode_pos, :, :, :] += dq_mat_term
        #if couple_type == 'F':
        #    dq_mat_term, dp_mat_term = operators.der_mat_gen_Ex(G_ex_ph_F[mode_pos, :, :], w_list[:, mode_pos], Ex_kappa_ind_mat)
        #    dp_mat[mode_pos, :, :, :] += dp_mat_term
        #    dq_mat[mode_pos, :, :, :] += dq_mat_term
        #if couple_type == 'PZ':
        #    dq_mat_term, dp_mat_term = operators.der_mat_gen_Ex(G_ex_ph_PZ[mode_pos, :, :], w_list[:, mode_pos], Ex_kappa_ind_mat)
        #    dp_mat[mode_pos, :, :, :] += dp_mat_term
        #    dq_mat[mode_pos, :, :, :] += dq_mat_term
    dq_shape = np.shape(dq_mat)
    dp_shape = np.shape(dp_mat)
    dq_mel_list_ind = np.zeros((dq_shape[0], dq_shape[1]), dtype=object)
    dq_mel_list = np.zeros((dq_shape[0], dq_shape[1]), dtype=object)
    dp_mel_list_ind = np.zeros((dp_shape[0], dp_shape[1]), dtype=object)
    dp_mel_list = np.zeros((dp_shape[0], dp_shape[1]), dtype=object)
    for mode_n in range(dq_shape[0]):
        for q_n in range(dq_shape[1]):
            nz_pos_q = np.where(dq_mat[mode_n, q_n] != 0.0 + 0.0j)
            nz_mels_q = dq_mat[mode_n, q_n][nz_pos_q]
            dq_mel_list_ind[mode_n, q_n] = nz_pos_q
            dq_mel_list[mode_n, q_n] = nz_mels_q
            nz_pos_p = np.where(dp_mat[mode_n, q_n] != 0.0 + 0.0j)
            nz_mels_p = dp_mat[mode_n, q_n][nz_pos_p]
            dp_mel_list_ind[mode_n, q_n] = nz_pos_p
            dp_mel_list[mode_n, q_n] = nz_mels_p

    dq_mel_list = ray.put(dq_mel_list)
    dq_mel_list_ind = ray.put(dq_mel_list_ind)
    dp_mel_list = ray.put(dp_mel_list)
    dp_mel_list_ind = ray.put(dp_mel_list_ind)
    dq_mat = ray.put(dq_mat)
    dp_mat = ray.put(dp_mat)
    rayvars = (dq_mat, dp_mat)
    sim.dq_mat = dq_mat
    sim.dp_mat = dp_mat
    sim.rayvars = rayvars
    # rayvars_ind = (dq_mel_list_ind, dq_mel_list, dp_mel_list_ind, dp_mel_list)
    return sim