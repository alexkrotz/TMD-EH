import numpy as np
import constants
from numba import jit
from tqdm import tqdm
import ray
import itertools

#np.random.seed(1234)

@jit(nopython=True)
def vec_single_x_array(single, array):
    single1 = np.ascontiguousarray(single) + 0.0j
    array1 = np.ascontiguousarray(array) + 0.0j
    out = np.ascontiguousarray(np.zeros(len(array))) + 0.0j
    for n in range(len(array)):
        out[n] = np.sum(single1 * array1[n])
    return out


def init_classical(sim):
    w_list = sim.w_list
    temp = sim.temp
    p_out = np.zeros(np.shape(w_list))
    q_out = np.zeros(np.shape(w_list))
    for mode_n in range(np.shape(w_list)[1]):
        for osc_n in range(np.shape(w_list)[0]):
            if w_list[osc_n, mode_n] != 0:
                p_out[osc_n, mode_n] = np.random.normal(0, np.sqrt(constants.kB * temp), 1)
                q_out[osc_n, mode_n] = np.random.normal(0, np.sqrt(constants.kB * temp) / w_list[osc_n, mode_n], 1)
    return p_out, q_out

def init_classical_parallel(sim):
    p = np.zeros((sim.trials, len(sim.w_list[:, 0]), len(sim.w_list[0, :])))
    q = np.zeros((sim.trials, len(sim.w_list[:, 0]), len(sim.w_list[0, :])))
    for i in range(sim.trials):
        p[i, :, :], q[i, :, :] = init_classical(sim)
    return p, q
@jit(nopython=True)
def screened_int(k_dist, chi_2D):
    return 2 * np.pi / (k_dist * (1 + 2 * np.pi * chi_2D * k_dist))
def screened_int_near_zero(d_k, chi_2D):
    # perform a 4D integration over two cubes separated by 0, with sidelength d_k
    res_fine = 4
    d_k_fine = d_k / res_fine
    k_combis = np.array(
        list(itertools.product(range(res_fine), range(res_fine), range(res_fine), range(res_fine)))) * d_k_fine
    return np.sum(
        screened_int(np.sqrt((k_combis[:, 0] - k_combis[:, 2]) ** 2 + (k_combis[:, 1] - k_combis[:, 3]) ** 2) + 1e-4,
                     chi_2D)) * d_k_fine ** 4
@jit(nopython=True)
def unscreened_int(k_dist):
    return 2*np.pi/(k_dist * k_dist)
dfac = 1
efac = 1

def H_BSE(sim, mat):
    bands_list = sim.bands_list
    bvec_list = sim.bvec_list
    ke_id = sim.ke_id
    kh_id = sim.kh_id
    se_id = sim.se_id
    sh_id = sim.sh_id
    c_band_id = sim.c_band_id
    v_band_id = sim.v_band_id
    umk_ind_mat = sim.umk_ind_mat
    umk_point_mat = sim.umk_point_mat
    umk_point_list = sim.umk_point_list
    out_mat = np.zeros((sim.num_exciton_states, sim.num_exciton_states), dtype=complex)
    ran = np.arange(0, sim.num_exciton_states)
    for n in range(0, sim.num_exciton_states):
        se1 = se_id[n]
        ke1 = ke_id[n]
        cb1 = c_band_id[n]
        sh1 = sh_id[n]
        kh1 = kh_id[n]
        vb1 = v_band_id[n]
        # bool_e is where kappa between initial and final electronic states is zero with True = 1 False = 0
        bool_e = ke1 == ke_id  # 1 - (ke1 != ke_id)
        bool_h = kh1 == kh_id  # 1 - (kh1 != kh_id)  # now false is 1, true is 0
        # tot_bool is where at least one of the kappas (electron or hole) is nonzero
        tot_bool = (1 - (bool_e * bool_h)).astype(bool)
        not_tot_bool = (1 - tot_bool.astype(int)).astype(bool)
        # j_vals takes the positions where both the kappas are the same and both the initial and final
        # electron/hole spins are the same
        same_kappa = umk_ind_mat[ke1, ke_id] == umk_ind_mat[kh1, kh_id]
        same_espin = se1 == se_id
        same_hspin = sh1 == sh_id
        # j_vals = (ran[tot_bool])[
        #    (umk_ind_mat[ke1, ke_id] == umk_ind_mat[kh1, kh_id])[tot_bool]]
        j_vals = (ran[tot_bool])[(same_kappa * same_espin * same_hspin)[tot_bool]]
        # zero_vals are the positions where the initial difference between the electron and hole are different
        # and their initial and final spins are different
        diff_kbar = umk_ind_mat[ke1, kh1] != umk_ind_mat[ke_id, kh_id]
        diff_espin = (1 - same_espin).astype(bool)
        diff_hspin = (1 - same_hspin).astype(bool)
        zero_vals = (ran[tot_bool])[diff_kbar[tot_bool] * diff_espin[tot_bool] * diff_hspin[tot_bool]]
        k_dist = np.linalg.norm(umk_point_mat[ke1, ke_id[j_vals]], axis=1)
        out_mat[n, j_vals] =  dfac * -1.0 * screened_int(k_dist / mat.lattice_const, mat.chi_2D) * mat.int_fac * \
                             vec_single_x_array(bvec_list[sh1, kh1, :, vb1],
                                                bvec_list[sh_id[j_vals], kh_id[j_vals], :, v_band_id[j_vals]].conj()) * \
                             vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                bvec_list[se_id[j_vals], ke_id[j_vals], :, c_band_id[j_vals]])

        out_mat[n, zero_vals] = 0.0 + 0.0j

        # p_vals is essentially the diagonal so kappa is zero, same kappa, espin and hspin.
        p_vals = ran[not_tot_bool][(same_kappa * same_espin * same_hspin)[not_tot_bool]]
        out_mat[n, p_vals] = (bands_list[se_id[p_vals], ke_id[p_vals], c_band_id[p_vals]] -
                              bands_list[sh_id[p_vals], kh_id[p_vals], v_band_id[p_vals]]) * constants.eV_to_Hartree - \
                              dfac * screened_int_near_zero(mat.d_k, mat.chi_2D) * mat.int_fac * \
                             vec_single_x_array(bvec_list[sh1, kh1, :, vb1],
                                                bvec_list[sh_id[p_vals], kh_id[p_vals], :, v_band_id[p_vals]].conj()) * \
                             vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                bvec_list[se_id[p_vals], ke_id[p_vals], :, c_band_id[p_vals]])
        same_spin_0 = se1 == sh1
        same_spin_1 = se_id == sh_id
        nz_diff = np.linalg.norm(umk_point_mat[ke1, kh1]) >= 1e-9
        e_vals = ran[same_spin_0 * same_spin_1 * nz_diff * same_kappa]
        k_diff = np.linalg.norm(umk_point_mat[ke1, kh1])
        if np.abs(k_diff) >= 1e-9:
            # out_mat[n, e_vals] += efac * screened_int(k_diff / lattice_const, chi_2D) * int_fac * \
            #                      np.dot(np.conj(bvec_list[se1, ke1, :, cb1]), bvec_list[sh1, kh1, :, vb1]) * \
            #                      np.sum(np.conj(bvec_list[sh_id[e_vals], kh_id[e_vals], :, v_band_id[e_vals]]) * \
            #                             bvec_list[se_id[e_vals], ke_id[e_vals], :, c_band_id[e_vals]], axis=1)
            out_mat[n, e_vals] += efac * unscreened_int(k_diff / mat.lattice_const) * mat.int_fac * \
                                  np.dot(np.conj(bvec_list[se1, ke1, :, cb1]), bvec_list[sh1, kh1, :, vb1]) * \
                                  np.sum(np.conj(bvec_list[sh_id[e_vals], kh_id[e_vals], :, v_band_id[e_vals]]) * \
                                         bvec_list[se_id[e_vals], ke_id[e_vals], :, c_band_id[e_vals]], axis=1)

    out_mat = out_mat
    return out_mat / constants.therm_Ha

def H_BSE_block(block_num, sim, mat):
    mask = (sim.dk_id == block_num)
    bse_id_t = sim.b_se_id[mask]
    bsh_id_t = sim.b_sh_id[mask]
    be_id_t = sim.be_id_1b[mask]
    bh_id_t = sim.bh_id_1b[mask]
    bc_id_t = sim.b_c_band_id[mask]
    bv_id_t = sim.b_v_band_id[mask]
    umk_ind_mat = sim.umk_ind_mat
    umk_point_mat = sim.umk_point_mat
    bands_list = sim.bands_list
    bvec_list = sim.bvec_list
    out_mat = np.zeros((len(be_id_t), len(be_id_t)), dtype=complex)
    ran = np.arange(0, len(be_id_t))
    for n in range(0, len(be_id_t)):
        se1 = bse_id_t[n]
        ke1 = be_id_t[n]
        cb1 = bc_id_t[n]
        sh1 = bsh_id_t[n]
        kh1 = bh_id_t[n]
        vb1 = bv_id_t[n]
        bool_e = ke1 == be_id_t #1 - (ke1 != be_id_t)
        bool_h = kh1 == bh_id_t #1 - (kh1 != bh_id_t)  # now false is 1, true is 0
        tot_bool = (1 - (bool_e * bool_h)).astype(bool)
        not_tot_bool = (1 - tot_bool.astype(int)).astype(bool)
        same_kappa = umk_ind_mat[ke1, be_id_t] == umk_ind_mat[kh1, bh_id_t]
        same_espin = se1 == bse_id_t
        same_hspin = sh1 == bsh_id_t
        #j_vals = (ran[tot_bool])[
        #    (umk_ind_mat[ke1, be_id_t] == umk_ind_mat[kh1, bh_id_t])[tot_bool]]
        j_vals = (ran[tot_bool])[same_kappa[tot_bool] * same_espin[tot_bool] * same_hspin[tot_bool]]
        diff_kbar = umk_ind_mat[ke1, kh1] != umk_ind_mat[be_id_t, bh_id_t]
        diff_espin = (1 - same_espin).astype(bool)
        diff_hspin = (1 - same_hspin).astype(bool)
        #zero_vals = (ran[tot_bool])[
        #    (umk_ind_mat[ke1, kh1] != umk_ind_mat[be_id_t, bh_id_t])[tot_bool]]
        zero_vals = (ran[tot_bool])[diff_kbar[tot_bool] * diff_espin[tot_bool] * diff_hspin[tot_bool]]
        k_dist = np.linalg.norm(umk_point_mat[ke1, be_id_t[j_vals]], axis=1)
        out_mat[n, j_vals] = dfac * -1.0 * screened_int(k_dist / mat.lattice_const, mat.chi_2D) * mat.int_fac * \
                             vec_single_x_array(bvec_list[sh1, kh1, :, vb1],
                                                bvec_list[bsh_id_t[j_vals], bh_id_t[j_vals], :, bv_id_t[j_vals]].conj()) * \
                             vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                bvec_list[bse_id_t[j_vals], be_id_t[j_vals], :, bc_id_t[j_vals]])
        out_mat[n, zero_vals] = 0.0 + 0.0j
        p_vals = ran[not_tot_bool][(same_kappa * same_espin * same_hspin)[not_tot_bool]]
        out_mat[n, p_vals] = (bands_list[bse_id_t[p_vals], be_id_t[p_vals], bc_id_t[p_vals]] -
                               bands_list[bsh_id_t[p_vals], bh_id_t[p_vals], bv_id_t[p_vals]]) * constants.eV_to_Hartree -\
                             dfac * screened_int_near_zero(mat.d_k, mat.chi_2D) * mat.int_fac * \
                             vec_single_x_array(bvec_list[sh1, kh1, :, vb1],
                                                bvec_list[bsh_id_t[p_vals], bh_id_t[p_vals], :, bv_id_t[p_vals]].conj()) * \
                             vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                bvec_list[bse_id_t[p_vals], be_id_t[p_vals], :, bc_id_t[p_vals]])
        same_spin_0 = se1 == sh1
        same_spin_1 = bse_id_t == bsh_id_t
        nz_diff = np.linalg.norm(umk_point_mat[ke1, kh1]) >= 1e-9
        e_vals = ran[same_spin_0 * same_spin_1 * nz_diff * same_kappa]
        k_diff = np.linalg.norm(umk_point_mat[ke1, kh1])
        if np.abs(k_diff) >= 1e-9:
            #out_mat[n, e_vals] += efac * screened_int(k_diff / lattice_const, chi_2D) * int_fac * \
            #                      np.dot(np.conj(bvec_list[se1, ke1, :, cb1]), bvec_list[sh1, kh1, :, vb1]) * \
            #                      np.sum(np.conj(bvec_list[bsh_id_t[e_vals], bh_id_t[e_vals], :, bv_id_t[e_vals]]) * \
            #                             bvec_list[bse_id_t[e_vals], be_id_t[e_vals], :, bc_id_t[e_vals]], axis=1)
            out_mat[n, e_vals] += efac * unscreened_int(k_diff / mat.lattice_const) * mat.int_fac * \
                                  np.dot(np.conj(bvec_list[se1, ke1, :, cb1]), bvec_list[sh1, kh1, :, vb1]) * \
                                  np.sum(np.conj(bvec_list[bsh_id_t[e_vals], bh_id_t[e_vals], :, bv_id_t[e_vals]]) * \
                                         bvec_list[bse_id_t[e_vals], be_id_t[e_vals], :, bc_id_t[e_vals]], axis=1)
    return out_mat / constants.therm_Ha

def H_q(sim):
    bands_list = sim.bands_list
    ke_id = sim.ke_id
    kh_id = sim.kh_id
    se_id = sim.se_id
    sh_id = sim.sh_id
    c_band_id = sim.c_band_id
    v_band_id = sim.v_band_id
    out_mat = np.diag(bands_list[se_id, ke_id, c_band_id] - bands_list[sh_id, kh_id, v_band_id]) * constants.eV_to_Hartree

    return out_mat

def H_qc(q, p, sim):
    g = sim.g_list_DP
    w = sim.w_list
    ke_id = sim.ke_id
    kh_id = sim.kh_id
    se_id = sim.se_id
    sh_id = sim.sh_id
    c_band_id = sim.c_band_id
    v_band_id = sim.v_band_id
    umk_ind_mat = sim.umk_ind_mat
    bvec_list = sim.bvec_list
    out_mat_e = np.zeros((sim.num_exciton_states, sim.num_exciton_states), dtype=complex)
    out_mat_h = np.zeros((sim.num_exciton_states, sim.num_exciton_states), dtype=complex)
    ran = np.arange(0, sim.num_exciton_states)
    for n in range(0, sim.num_exciton_states):
        se1 = se_id[n]
        ke1 = ke_id[n]
        cb1 = c_band_id[n]
        sh1 = sh_id[n]
        kh1 = kh_id[n]
        vb1 = v_band_id[n]
        j_vals_e = ran[((kh_id[ran] == kh1).astype(int) * (v_band_id[ran] == vb1).astype(int) * \
                        (sh_id[ran] == sh1).astype(int) * (se_id[ran] == se1).astype(int)).astype(
            bool)]  # find values where hole has remained the same
        j_vals_h = ran[((ke_id[ran] == ke1).astype(int) * (c_band_id[ran] == cb1).astype(int) * \
                        (se_id[ran] == se1).astype(int) * (sh_id[ran] == sh1)).astype(
            bool)]  # find values where elec has remained the same
        kappa_ind_e = umk_ind_mat[ke1][ke_id[j_vals_e]]  # kappa = ke1 - ke2(kh1 = kh2) ke1= k1 + kappa ke2 = k1
        m_kappa_ind_e = umk_ind_mat[ke_id[j_vals_e]][:, ke1]  # -kappa = ke2 - ke1, ke1 = k1 + kappa ke2 = k1
        kappa_ind_h = umk_ind_mat[kh1][kh_id[j_vals_h]]  # kappa = kh1 - kh2  kh1 = k2 + kappa, ke2 = k2
        m_kappa_ind_h = umk_ind_mat[kh_id[j_vals_h]][:, kh1]  # g[ke_id[j_vals_e], kappa_ind_e, cb1, c_band_id[
        # j_vals_e]] * \
        out_mat_e[n][j_vals_e] += (1 / np.sqrt(2)) * g[ kappa_ind_e, 0, ke_id[j_vals_e], c_band_id[j_vals_e], cb1, se_id[j_vals_e], se1] * \
                                  vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                     bvec_list[se_id[j_vals_e], ke_id[j_vals_e], :,
                                                     c_band_id[j_vals_e]]) * \
                                  np.sqrt(w[kappa_ind_e, 0]) * (
                                          w[kappa_ind_e, 0] * (q[m_kappa_ind_e, 0] + q[kappa_ind_e, 0]) - 1.0j * (
                                          p[m_kappa_ind_e, 0] - p[kappa_ind_e, 0]))
        out_mat_h[j_vals_h, n] += (1 / np.sqrt(2)) * g[
            kappa_ind_h, 0, kh1, vb1, v_band_id[j_vals_h], sh1, sh_id[j_vals_h]] * \
                                  vec_single_x_array(bvec_list[sh1, kh1, :, vb1].conj(),
                                                     bvec_list[sh_id[j_vals_h], kh_id[j_vals_h], :,
                                                     v_band_id[j_vals_h]]) * \
                                  np.sqrt(w[kappa_ind_h, 0]) * (
                                          w[kappa_ind_h, 0] * (q[m_kappa_ind_h, 0] + q[kappa_ind_h, 0]) - 1.0j * (
                                          p[m_kappa_ind_h, 0] - p[kappa_ind_h, 0]))
    return (out_mat_e + out_mat_h)  # * np.outer(np.conj(gauge_list), gauge_list)


def gen_bk_mat(g, sim):
    se_id = sim.se_id
    ke_id = sim.ke_id
    c_band_id = sim.c_band_id
    sh_id = sim.sh_id
    kh_id = sim.kh_id
    v_band_id = sim.v_band_id
    umk_ind_mat = sim.umk_ind_mat
    bvec_list = sim.bvec_list
    num_of_states = len(ke_id)
    bloch_mat = np.zeros((num_of_states, num_of_states), dtype=complex)
    kappa_ind_mat = np.zeros((num_of_states, num_of_states), dtype=int)
    mask_mat = np.ones((num_of_states, num_of_states))
    ran = np.arange(0, num_of_states)
    for n in range(0, num_of_states):
        se1 = se_id[n]
        ke1 = ke_id[n]
        cb1 = c_band_id[n]
        sh1 = sh_id[n]
        kh1 = kh_id[n]
        vb1 = v_band_id[n]
        j_vals_e = ran[((kh_id[ran] == kh1).astype(int) * (v_band_id[ran] == vb1).astype(int) * \
                        (sh_id[ran] == sh1).astype(int) * (se_id[ran] == se1).astype(int)).astype(
            bool)]  # find values where hole has remained the same
        j_vals_h = ran[((ke_id[ran] == ke1).astype(int) * (c_band_id[ran] == cb1).astype(int) * \
                        (se_id[ran] == se1).astype(int) * (sh_id[ran] == sh1)).astype(
            bool)]  # find values where elec has remained the same
        kappa_ind_e = umk_ind_mat[ke1][ke_id[j_vals_e]]
        kappa_ind_h = umk_ind_mat[kh1][kh_id[j_vals_h]]
        bloch_mat[n, j_vals_e] += g[kappa_ind_e, ke_id[j_vals_e], c_band_id[j_vals_e], cb1, se_id[j_vals_e], se1] * \
                                  vec_single_x_array(bvec_list[se1, ke1, :, cb1].conj(),
                                                     bvec_list[se_id[j_vals_e], ke_id[j_vals_e], :,
                                                     c_band_id[j_vals_e]])
        bloch_mat[j_vals_h, n] += g[kappa_ind_h, kh1, vb1, v_band_id[j_vals_h], sh1, sh_id[j_vals_h]] * \
                                  vec_single_x_array(bvec_list[sh1, kh1, :, vb1].conj(),
                                                     bvec_list[sh_id[j_vals_h], kh_id[j_vals_h], :,
                                                     v_band_id[j_vals_h]])
        kappa_ind_mat[n, j_vals_e] = umk_ind_mat[ke1, ke_id[j_vals_e]]
        kappa_ind_mat[j_vals_h, n] = umk_ind_mat[kh1, kh_id[j_vals_h]]
        mask_mat[n, j_vals_e] = 0
        mask_mat[j_vals_h, n] = 0
    return np.ma.masked_array(bloch_mat, mask=mask_mat), np.ma.masked_array(kappa_ind_mat, mask=mask_mat, dtype=int)

def der_mat_gen(bloch_mat, kappa_ind_mat, w, sim):
    umk_ind_mat = sim.umk_ind_mat
    umk_point_mat = sim.umk_point_mat
    umk_point_list = sim.umk_point_list
    tally_ind = np.zeros(len(umk_point_list), dtype=int)
    for n in range(len(kappa_ind_mat)):
        for a in kappa_ind_mat[n][~kappa_ind_mat[n].mask].data:
            tally_ind[a] += 2
    dq_mel_list = [np.zeros(a, dtype=complex) for a in tally_ind]
    dq_mel_list_ind = [np.zeros((2, a), dtype=int) for a in tally_ind]
    dq_mel_list_ind_out = [(1, 1) for _ in tally_ind]
    dp_mel_list = [np.zeros(a, dtype=complex) for a in tally_ind]
    dp_mel_list_ind = [np.zeros((2, a), dtype=int) for a in tally_ind]
    dp_mel_list_ind_out = [(1, 1) for _ in tally_ind]
    der_ind_mat_dq = (np.sqrt(w[kappa_ind_mat] ** 3) / (np.sqrt(2))) * bloch_mat
    der_ind_mat_dp = 1.0j * (np.sqrt(w[kappa_ind_mat]) / (np.sqrt(2))) * bloch_mat
    m_kappa_ind_mat = np.transpose(kappa_ind_mat)
    index = np.zeros(len(umk_point_list), dtype=int)
    for n in tqdm(range(len(kappa_ind_mat))):
        a_n = 0
        dq_line = der_ind_mat_dq[n][~der_ind_mat_dq[n].mask].data
        dp_line = der_ind_mat_dp[n][~der_ind_mat_dq[n].mask].data
        ind_line = np.arange(len(kappa_ind_mat[n]))[~der_ind_mat_dq[n].mask]
        for a in kappa_ind_mat[n][~kappa_ind_mat[n].mask].data:
            # print(n,a,index[a])
            dq_mel_list[a][index[a]] += dq_line[a_n]
            dp_mel_list[a][index[a]] += dp_line[a_n]
            dq_mel_list_ind[a][:, index[a]] = np.array([n, ind_line[a_n]]).astype(int)
            dp_mel_list_ind[a][:, index[a]] = np.array([n, ind_line[a_n]]).astype(int)
            index[a] += 1
            a_n += 1
        a_n = 0
        for b in m_kappa_ind_mat[n][~m_kappa_ind_mat[n].mask].data:
            dq_mel_list[b][index[b]] += dq_line[a_n]
            dp_mel_list[b][index[b]] -= dp_line[a_n]
            dq_mel_list_ind[b][:, index[b]] = np.array([n, ind_line[a_n]]).astype(int)
            dp_mel_list_ind[b][:, index[b]] = np.array([n, ind_line[a_n]]).astype(int)
            index[b] += 1
            a_n += 1
    for n in range(len(dq_mel_list_ind)):
        dq_mel_list_ind_out[n] = (np.array(dq_mel_list_ind[n][0]), np.array(dq_mel_list_ind[n][1]))
        dp_mel_list_ind_out[n] = (np.array(dp_mel_list_ind[n][0]), np.array(dp_mel_list_ind[n][1]))
    return dq_mel_list, dq_mel_list_ind_out, dp_mel_list, dp_mel_list_ind_out

def get_dkk(eig_k, eig_j, evdiff, sim):
    shape = np.shape(sim.w_list)
    phonon_modes = sim.phonon_modes
    dq_mel_list_full, dq_mel_list_ind_full, dp_mel_list_full, dp_mel_list_ind_full = sim.rayvars
    dqmel_list_full = ray.get(dq_mel_list_full)
    dqmel_list_ind_full = ray.get(dq_mel_list_ind_full)
    dpmel_list_full = ray.get(dp_mel_list_full)
    dpmel_list_ind_full = ray.get(dp_mel_list_ind_full)
    dkkq = np.zeros(shape, dtype=complex)
    dkkp = np.zeros(shape, dtype=complex)
    unique_ph_modes = np.unique(phonon_modes)
    for n in range(len(phonon_modes)):
        mode_pos = np.where(unique_ph_modes == phonon_modes[n])[0][0]
        dqmel_list = dqmel_list_full[n]
        dqmel_list_ind = dqmel_list_ind_full[n]
        dpmel_list = dpmel_list_full[n]
        dpmel_list_ind = dpmel_list_ind_full[n]
        for q_in in range(shape[0]):
            dkkq[q_in, mode_pos] = np.dot(np.conj(eig_k[dqmel_list_ind[q_in][0].astype(int)]),
                                          dqmel_list[q_in] * eig_j[dqmel_list_ind[q_in][1].astype(int)])
            dkkp[q_in, mode_pos] = np.dot(np.conj(eig_k[dpmel_list_ind[q_in][0].astype(int)]),
                                          dpmel_list[q_in] * eig_j[dpmel_list_ind[q_in][1].astype(int)])
    return dkkq / evdiff, dkkp / evdiff

def quantumForce(cg, sim):
    phonon_modes = sim.phonon_modes
    shape = np.shape(sim.w_list)
    dq_mel_list_full, dq_mel_list_ind_full, dp_mel_list_full, dp_mel_list_ind_full = sim.rayvars
    dqmel_list_full = ray.get(dq_mel_list_full)
    dqmel_list_ind_full = ray.get(dq_mel_list_ind_full)
    dpmel_list_full = ray.get(dp_mel_list_full)
    dpmel_list_ind_full = ray.get(dp_mel_list_ind_full)
    fq = np.zeros(shape, dtype=complex)
    fp = np.zeros(shape, dtype=complex)
    unique_ph_modes = np.unique(phonon_modes)
    for n in range(len(phonon_modes)):
        mode_pos = np.where(unique_ph_modes == phonon_modes[n])[0][0]
        dqmel_list = dqmel_list_full[n]
        dqmel_list_ind = dqmel_list_ind_full[n]
        dpmel_list = dpmel_list_full[n]
        dpmel_list_ind = dpmel_list_ind_full[n]
        for q_in in range(shape[0]):
            fq[q_in, mode_pos] = np.dot(np.conj(cg[dqmel_list_ind[q_in][0]]),
                                        dqmel_list[q_in] * cg[dqmel_list_ind[q_in][1]])
            fp[q_in, mode_pos] = np.dot(np.conj(cg[dpmel_list_ind[q_in][0]]),
                                        dpmel_list[q_in] * cg[dpmel_list_ind[q_in][1]])
            if np.abs(np.imag(fq[q_in, mode_pos])) > 1e-10:
                print('QUANTUM FORCE ERROR: im(Fp) = ', np.imag(fq[q_in, n]))
            if np.abs(np.imag(fp[q_in, mode_pos])) > 1e-10:
                print('QUANTUM FORCE ERROR: im(Fq) = ', np.imag(fp[q_in, n]))
    return np.real(fq), np.real(fp)

def H_BSE_Ex(sim):
    return np.diag(sim.trunc_bse_evals)

def H_qc_Ex(q,p,sim):
    kappa_ind = sim.Ex_kap_ind_mat
    m_kappa_ind = np.transpose(kappa_ind)
    m_kappa_zeros = np.where(m_kappa_ind == -1)
    p_kappa_zeros = np.where(kappa_ind == -1)
    fac_mat_m = np.ones(np.shape(kappa_ind))
    fac_mat_p = np.ones(np.shape(kappa_ind))
    fac_mat_m[m_kappa_zeros] = 0
    fac_mat_p[p_kappa_zeros] = 0
    w = sim.w_list
    g = sim.G_ex_ph_DP[0,:,:]
    out = g * fac_mat_m * fac_mat_p * (1 / np.sqrt(2)) * np.sqrt(w[kappa_ind,0]) * (
            w[kappa_ind,0] * (q[m_kappa_ind,0] + q[kappa_ind,0]) - 1.0j * (p[m_kappa_ind,0] - p[kappa_ind,0]))
    return out

def der_mat_gen_Ex(G, w, kappa_ind):
    num_osc = len(w)
    dq_mat = np.zeros((num_osc, len(kappa_ind), len(kappa_ind[0])), dtype=complex)
    dp_mat = np.zeros((num_osc, len(kappa_ind), len(kappa_ind[0])), dtype=complex)
    m_kappa_ind = np.transpose(kappa_ind)
    for n in range(num_osc):
        p_ind = np.where(kappa_ind == n)
        m_ind = np.where(m_kappa_ind == n)
        dq_mat[n][p_ind] += (1 / np.sqrt(2)) * w[kappa_ind][p_ind] ** (3 / 2) * G[p_ind]
        dq_mat[n][m_ind] += (1 / np.sqrt(2)) * w[m_kappa_ind][m_ind] ** (3 / 2) * G[m_ind]
        dp_mat[n][p_ind] += (1 / np.sqrt(2)) * 1.0j * w[kappa_ind][p_ind] ** (1 / 2) * G[p_ind]
        dp_mat[n][m_ind] += (1 / np.sqrt(2)) * -1.0j * w[m_kappa_ind][m_ind] ** (1 / 2) * G[m_ind]
    return dq_mat, dp_mat
@jit(nopython=True, fastmath=True)
def matprod(mat, vec1, vec2):  # takes hijk mat and returns product ih with vec1 vec2
    vec1 = np.conj(np.ascontiguousarray(vec1))
    vec2 = np.ascontiguousarray(vec2)
    sh = np.shape(mat)
    out_vec = np.ascontiguousarray(np.zeros((sh[1], sh[0]))) + 0.0j
    for n in range(sh[1]):
        for m in range(sh[0]):
            sub_mat = np.ascontiguousarray(mat[m, n, :, :])
            # nz_pos = np.where(sub_mat != 0.0+0.0j)
            out_vec[n, m] = np.dot(vec1, np.dot(sub_mat,
                                                vec2))  # np.sum(vec1[nz_pos[0]]*get_ind(sub_mat,nz_pos[0],nz_pos[1])*vec2[nz_pos[1]])#
    return out_vec
def quantumForce_Ex(cg,sim):
    dq_mat, dp_mat = sim.rayvars
    dq_mat_full = ray.get(dq_mat)
    dp_mat_full = ray.get(dp_mat)
    fq = matprod(dq_mat_full, cg,
                 cg)  # indprod(dqmel_list_ind, dqmel_list, cg, cg, sh)#np.einsum('hij,j->ih', np.einsum('hijk,k->hij', dq_mat_full, cg), np.conj(cg))
    fp = matprod(dp_mat_full, cg,
                 cg)  # indprod(dpmel_list_ind, dpmel_list, cg, cg, sh)#np.einsum('hij,j->ih', np.einsum('hijk,k->hij', dp_mat_full, cg), np.conj(cg))
    if np.sum(np.abs(np.imag(fq))) > 1e-10:
        print('QUANTUM FORCE ERROR: im(Fp) = ', np.imag(fq))
    if np.sum(np.abs(np.imag(fp))) > 1e-10:
        print('QUANTUM FORCE ERROR: im(Fq) = ', np.imag(fp))
    return np.real(fq), np.real(fp)

def get_dkk_Ex(eig_k, eig_j, evdiff, sim):
    dq_mat, dp_mat = sim.rayvars
    dq_mat_full = ray.get(dq_mat)
    dp_mat_full = ray.get(dp_mat)
    dkkq = matprod(dq_mat_full, eig_k,
                   eig_j)  # np.einsum('j,hij->ih', np.conj(eig_k), np.einsum('hijk,k->hij', dq_mat_full, eig_j)) / evdiff
    dkkp = matprod(dp_mat_full, eig_k,
                   eig_j)  # np.einsum('j,hij->ih', np.conj(eig_k), np.einsum('hijk,k->hij', dp_mat_full, eig_j)) / evdiff
    return dkkq/evdiff, dkkp/evdiff

##@jit(nopython=True)
def boltz(egrid, sim):
    if sim.temp == 0:
        out = np.zeros_like(egrid)
    else:
        z = np.sum(np.exp(-1.0 * (1.0 / (constants.kB * sim.temp)) * egrid))
        if np.abs(z) < 1e-10:
            out = np.zeros_like(egrid)
        else:
            out  = (1 / z) * np.exp(-1.0 * (1.0 / (constants.kB * sim.temp)) * egrid)
    return out


@jit(nopython=True)
def rho_0_adb_to_db(rho_0_adb, eigvec):
    rho_0_db = np.dot(np.dot(np.conj(eigvec), rho_0_adb + 0.0j), eigvec.transpose())
    return rho_0_db


@jit(nopython=True)
def rho_0_db_to_adb(rho_0_db, eigvec):
    rho_0_db = np.dot(np.dot(np.conj(eigvec).transpose(), rho_0_db + 0.0j), eigvec)
    return rho_0_db


def vec_adb_to_db(psi_adb, eigvec):
    # in each branch, take eigvector matrix (last two indices) and multiply by psi (a raw in a matrix):
    psi_db = np.matmul(eigvec, psi_adb)  # np.einsum('...ij,...j', eigvec, psi_adb)
    return psi_db


def vec_db_to_adb(psi_db, eigvec):
    # in each branch, take eigvector matrix (last two indices) and multiply by psi (a raw in a matrix):
    psi_db = np.matmul(psi_db, np.conj(eigvec))  # np.einsum('...ij,...j', eigvec, psi_adb)
    return psi_db

def eC(p, q, wgrid):
    return np.real(np.sum(((wgrid ** 2) / 2) * q ** 2 + (1 / 2) * p ** 2))


def eQ(mat, cg):
    return np.real(np.dot(np.conj(cg), np.dot(mat, cg.reshape((-1, 1))))[0])

@jit(nopython=True)
def RK4(p_bath, q_bath, QF, wgrid, dt):
    Fq, Fp = QF
    K1 = dt * (p_bath + Fp)
    L1 = -dt * (wgrid ** 2 * q_bath + Fq)  # [wn2] is w_alpha ^ 2
    K2 = dt * ((p_bath + 0.5 * L1) + Fp)
    L2 = -dt * (wgrid ** 2 * (q_bath + 0.5 * K1) + Fq)
    K3 = dt * ((p_bath + 0.5 * L2) + Fp)
    L3 = -dt * (wgrid ** 2 * (q_bath + 0.5 * K2) + Fq)
    K4 = dt * ((p_bath + L3) + Fp)
    L4 = -dt * (wgrid ** 2 * (q_bath + K3) + Fq)
    q_bath = q_bath + 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    p_bath = p_bath + 0.166667 * (L1 + 2 * L2 + 2 * L3 + L4)
    return p_bath, q_bath

@jit(nopython=True)
def nan_num(num):
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num


nan_num_vec = np.vectorize(nan_num)