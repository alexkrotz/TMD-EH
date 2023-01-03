import numpy as np
import time
import ray
from asyncio import Event
from typing import Tuple
from ray.actor import ActorHandle
from numba import jit
from tqdm import tqdm
import sys
from os import path
import operators
import constants

#np.random.seed(1234)

@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.
        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.
        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.
        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return
@ray.remote
def FSSH_dynamics_kspace(index, p, q, sim, mat, pba):
    start_time = time.time()
    #sim = ray.get(ray_sim)
    #mat = ray.get(ray_mat)
    tmax = sim.tmax
    dt = sim.dt
    dt_bath = sim.dt_bath
    qmat = operators.H_BSE(sim, mat)
    qcmat = operators.H_qc(q, p, sim)
    mat = qmat + qcmat
    eigval, eigvec = np.linalg.eigh(mat)
    psi_adb_0 = operators.vec_db_to_adb(sim.cg_db_0, eigvec)
    pops_adb_0 = np.conj(psi_adb_0) * psi_adb_0
    intervals = np.zeros(len(psi_adb_0))
    hop_count = 0
    for n in range(0, len(psi_adb_0)):
        intervals[n] = np.sum(np.real(pops_adb_0)[0:n+1])
    rand_val = np.random.rand()
    act_0 = (np.arange(len(pops_adb_0))[intervals > rand_val])[0]
    act_surf_ind = act_0
    nstates = len(pops_adb_0)
    act_surf = np.zeros(nstates, dtype=int)
    act_surf[act_surf_ind] = 1
    tdat = np.arange(0, tmax + dt, dt)
    tdat_bath = np.arange(0, tmax + dt_bath, dt_bath)
    den_mat_db_tot = np.zeros((len(tdat), nstates, nstates), dtype=complex)
    pops_db_tot = np.zeros((len(tdat), nstates), dtype=complex)
    pops_adb_tot = np.zeros((len(tdat), nstates), dtype=complex)
    eig_vals = np.zeros((len(tdat), nstates))
    pop_vals = np.zeros((len(tdat), nstates))
    flip_count = 0
    eclist = np.zeros(len(tdat))
    eqlist = np.zeros(len(tdat))
    b_pop_db = np.zeros((len(tdat), nstates))
    cg_adb = psi_adb_0
    cg_db = sim.cg_db_0
    ec = operators.eC(p, q, sim.w_list)
    eq = eigval[act_surf_ind]
    e_init = ec + eq
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if sim.use_tqdm:
            pba.update.remote(1)
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath:
            den_mat_adb = np.dot(np.conj(cg_adb).reshape((-1, 1)), np.array([cg_adb]))
            den_mat_adb[range(nstates), range(nstates)] = act_surf  # put act_surf on diagonal of rho_adb
            den_mat_db = operators.rho_0_adb_to_db(den_mat_adb, eigvec)
            den_mat_db_tot[t_ind, :, :] = den_mat_db_tot[t_ind, :, :] + den_mat_db
            pops_db_tot[t_ind, :] = pops_db_tot[t_ind, :] + np.diag(den_mat_db)
            pops_adb_tot[t_ind, :] = pops_adb_tot[t_ind, :] + np.diag(den_mat_adb)
            b_pop_adb = np.diag(operators.boltz(eigval, sim)) + 0.0j
            b_p_db = operators.rho_0_adb_to_db(b_pop_adb, eigvec)
            b_pop_db[t_ind] = np.real(np.diag(b_p_db))
            pop_vals[t_ind, :] = np.real(act_surf)
            eig_vals[t_ind, :] = np.real(eigval)
            ec = operators.eC(p, q, sim.w_list)
            eq = eigval[act_surf_ind]
            eclist[t_ind] = ec
            eqlist[t_ind] = eq
            energy = ec + eq
            if np.abs(e_init - energy) > 0.01 * np.abs(e_init):
                print('ERROR energy not conserved ', (e_init - energy) / e_init * 100, '% error')
            if np.abs(np.sum(np.real(np.conj(cg_db) * cg_db)) - 1.0) > 1e-5:
                # print(pops_db_tot[t_ind])
                print('Diabatic norm not conserved: ', tdat[t_ind], np.real(np.conj(cg_db) * cg_db))
            if np.abs(np.sum(np.real(np.conj(cg_adb) * cg_adb)) - 1.0) > 1e-5:
                # print(pops_db_tot[t_ind])
                print('Adiabatic norm not conserved: ', tdat[t_ind], np.real(np.conj(cg_adb) * cg_adb))
                #    print('Active surface: ',act_surf_ind)
            #print(4)
            t_ind += 1
        fq, fp = operators.quantumForce(eigvec[:, act_surf_ind], sim)
        p, q = operators.RK4(p, q, (fq, fp), sim.w_list, dt_bath)
        eigvec_previous = np.copy(eigvec)
        qcmat = operators.H_qc(q, p, sim)
        eigval, eigvec =np.linalg.eigh(qmat + qcmat)
        eigval_exp = np.exp(-1j * eigval * sim.dt_bath)
        diag_matrix = np.diag(eigval_exp)
        cg_adb = np.dot(diag_matrix, operators.vec_db_to_adb(cg_db, eigvec))
        cg_db = operators.vec_adb_to_db(cg_adb, eigvec)
        rand = np.random.rand()
        prod_A1_0 = (np.matmul(np.conj(eigvec[:, act_surf_ind]), eigvec_previous))
        prod_A1 = np.abs(prod_A1_0) * np.sign(np.real(prod_A1_0))
        prod_B_0 = np.sum(np.conj(eigvec_previous) * eigvec, axis=0)
        # prod_B = np.abs(prod_B_0) #* np.sign(np.real(prod_B_0))
        # phase_B = prod_B_0 / prod_B
        phase_B = np.exp(1.0j * np.angle(prod_B_0))
        hop_prob = -2 * np.real(prod_A1_0 * (cg_adb / cg_adb[act_surf_ind]) * phase_B)
        hop_prob[act_surf_ind] = 0
        bin_edge = 0
        for k in range(len(hop_prob)):
            hop_prob[k] = operators.nan_num(hop_prob[k])
            bin_edge = bin_edge + hop_prob[k]
            if rand < bin_edge:
                eig_k = eigvec[:, act_surf_ind]
                eig_j = eigvec[:, k]
                eigval_k = eigval[act_surf_ind]
                eigval_j = eigval[k]
                ev_diff = eigval_j - eigval_k
                dkkq, dkkp = operators.get_dkk(eig_k, eig_j, ev_diff, sim)
                phase_A1 = prod_A1_0 / prod_A1
                ab_phase = (phase_A1 * phase_B)[k]
                dkkq = dkkq * np.conj(ab_phase)
                dkkp = dkkp * np.conj(ab_phase)
                phase = np.imag(np.log(dkkq[np.argmax(np.abs(dkkq))]))
                dkkq = dkkq * np.exp(-1.0j * phase)
                phase = np.imag(np.log(dkkp[np.argmax(np.abs(dkkp))]))
                dkkp = dkkp * np.exp(-1.0j * phase)
                if np.max(np.abs(np.imag(dkkq)))/np.max(np.abs(np.real(dkkq))) > 1e-7:
                    print('ERROR IMAGINARY DKKQ: ', np.max(np.abs(np.imag(dkkq)))/np.max(np.abs(np.real(dkkq))))
                if np.max(np.abs(np.imag(dkkp)))/np.max(np.abs(np.real(dkkp))) > 1e-7:
                    print('ERROR IMAGINARY DKKP: ', np.max(np.abs(np.imag(dkkp)))/np.max(np.abs(np.real(dkkp))))
                delta_q = np.real(dkkp)
                delta_p = np.real(dkkq)
                akkq = np.sum((1 / 2) * np.abs(
                    delta_p) ** 2)  # (1 / 2) * np.sum(((w_list * dkkp_i) - dkkq_r) * ((w_list * dkkp_i) - dkkq_r))
                akkp = np.sum((1 / 2) * (sim.w_list ** 2) * np.abs(
                    delta_q) ** 2)  # (1 / 2) * np.sum((w_list ** 2) * (dkkp_r + (1/w_list) * dkkq_i) * (dkkp_r + (1/w_list) * dkkq_i))
                bkkq = np.sum((p * delta_p))
                bkkp = -np.sum((sim.w_list ** 2) * q * delta_q)
                disc = (bkkq + bkkp) ** 2 - 4 * (akkq + akkp) * ev_diff
                if disc >= 0:
                    if bkkq + bkkp < 0:
                        gamma = (bkkq + bkkp) + np.sqrt(disc)
                    else:
                        gamma = (bkkq + bkkp) - np.sqrt(disc)
                    if akkp + akkq == 0:
                        gamma = 0
                    else:
                        gamma = gamma / (2 * (akkq + akkp))

                    p = p - np.real(gamma) * delta_p  # rescale
                    q = q + np.real(gamma) * delta_q  # rescale
                    # print('hop',tdat_bath[t_bath_ind], act_surf_ind,eigval[act_surf_ind], k,eigval[k])
                    act_surf_ind = k
                    act_surf = np.zeros_like(act_surf)
                    act_surf[act_surf_ind] = 1
                    hop_count += 1
                # rev = False
                # if disc < 0 and rev:
                #    if akkp + akkq == 0:
                #        gamma = 0
                #    else:
                #        gamma = (bkkq + bkkp) / (akkq + akkp)
                #    p = p - np.real(gamma) * dkkq  # flip
                #    q = q + np.real(gamma) * dkkp  # flip
                #    flip_count += 1
                break
    pops_db = np.real(pops_db_tot)
    simCdb = pops_db
    simT = tdat
    simN = np.sum(pops_db, axis=1)
    simEq = eqlist
    simEc = eclist
    simPopB = b_pop_db
    end_time = time.time()
    msg = 'trial index: ' + str(index) + ' hop count: ' + str(hop_count) + ' flip count: ' + str(
        flip_count) + ' time: ' + str(end_time - start_time)
    return simCdb, simT, simN, simEq, simEc, simPopB, msg
def parallel_run_ray_kspace(sim, mat):
    sim.print_dynamics_info()
    print('Starting dynamics calculation...')
    st = time.time()
    r_ind = 0
    if sim.nprocs > sim.trials:
        sim.nprocs = sim.trials
    num_ticks = len(range(round((sim.tmax / sim.dt_bath)))) * sim.nprocs
    ray_sim = ray.put(sim)
    ray_mat = ray.put(mat)
    for run in range(0, int(sim.trials / sim.nprocs)):
        if sim.use_tqdm:
            pb = ProgressBar(num_ticks)
            actor = pb.actor
        p, q = operators.init_classical_parallel(sim)

        if sim.method == 'FSSH':
            results = [
                FSSH_dynamics_kspace.remote(run * sim.nprocs + i, p[i, :, :], q[i, :, :], ray_sim, ray_mat, actor)\
                for i in range(sim.nprocs)]
        if sim.use_tqdm:
            pb.print_until_done()
        for r in results:
            simCdb, simT, simN, simEq, simEc, simPopB, msg = ray.get(r)
            print(msg)
            if run == 0 and r_ind == 0:
                simCdbdat = np.zeros_like(simCdb)
                simTdat = np.zeros_like(simT)
                simPopBdat = np.zeros_like(simPopB)
                simEcdat = np.zeros_like(simEc)
                simEqdat = np.zeros_like(simEq)
                simNdat = np.zeros_like(simN)
            simCdbdat += simCdb
            simTdat += simT
            simPopBdat += simPopB
            simEcdat += simEc
            simEqdat += simEq
            simNdat += simN
            r_ind += 1
    filename = sim.calc_dir + '/' + sim.calc_dir
    if path.exists(filename + '_resCdb.csv'):
        simCdbdat += np.loadtxt(filename + '_resCdb.csv', delimiter=",")
    if path.exists(filename + '_resT.csv'):
        simTdat += np.loadtxt(filename + '_resT.csv', delimiter=",")
    if path.exists(filename + '_resPopB.csv'):
        simPopBdat += np.loadtxt(filename + '_resPopB.csv', delimiter=',')
    if path.exists(filename + '_resEc.csv'):
        simEcdat += np.loadtxt(filename + '_resEc.csv', delimiter=",")
    if path.exists(filename + '_resEq.csv'):
        simEqdat += np.loadtxt(filename + '_resEq.csv', delimiter=",")
    if path.exists(filename + '_resN.csv'):
        simNdat += np.loadtxt(filename + '_resN.csv', delimiter=",")
    et = time.time()
    print('took ', np.round(et - st, 3), ' seconds')
    ray.shutdown()
    sim.resCdb = simCdbdat
    sim.resT = simTdat
    sim.resN = simNdat
    sim.resEq = simEqdat
    sim.resEc = simEcdat
    sim.resPopB = simPopBdat
    return sim

def dynamics_kspace(sim, mat):

    num_exciton_states = sim.num_exciton_states
    dk_id = sim.dk_id
    dk_id_sort = sim.dk_id_sort
    evec_block_list = sim.evec_block_list
    block_dims = sim.block_dims
    block_unsort = sim.block_unsort
    def get_evec(alpha):
        evec_block = np.ascontiguousarray(np.zeros(num_exciton_states)) + 0.0j
        dk_ran = dk_id == dk_id_sort[alpha]
        evec = evec_block_list[:, alpha][:block_dims[dk_id_sort[alpha]]]
        evec_block[dk_ran] = evec
        evec_out = evec_block[block_unsort]
        return evec_out

    cg_db_0 = get_evec(sim.init_state_num)
    sim.cg_db_0 = cg_db_0
    sim = parallel_run_ray_kspace(sim, mat)
    filename = sim.calc_dir + '/' + sim.calc_dir
    np.savetxt(filename + '_resCdb.csv', sim.resCdb, delimiter=",")
    np.savetxt(filename + '_resT.csv', sim.resT, delimiter=",")
    np.savetxt(filename + '_resPopB.csv', sim.resPopB, delimiter=",")
    np.savetxt(filename + '_resEc.csv', sim.resEc, delimiter=",")
    np.savetxt(filename + '_resEq.csv', sim.resEq, delimiter=",")
    np.savetxt(filename + '_resN.csv', sim.resN, delimiter=",")
    np.savetxt(filename + '_kgrid.csv', sim.k, delimiter=",")

    return sim
@ray.remote
def FSSH_dynamics_exciton(index, p, q, sim, mat, pba):
    start_time = time.time()
    #print(ray_sim)
    #sim = ray.get(ray_sim)
    #mat = ray.get(ray_mat)
    tmax = sim.tmax
    dt = sim.dt
    dt_bath = sim.dt_bath
    qmat = operators.H_BSE_Ex(sim)
    qcmat = operators.H_qc_Ex(q, p, sim)
    qcmat_orig = np.copy(qcmat)
    full_mat = qmat + qcmat
    fmat_orig = np.copy(full_mat)
    eigval, eigvec = np.linalg.eigh(full_mat)
    psi_adb_0 = operators.vec_db_to_adb(sim.cg_db_0, eigvec)
    pops_adb_0 = np.conj(psi_adb_0) * psi_adb_0
    intervals = np.zeros(len(psi_adb_0))
    hop_count = 0
    for n in range(0, len(psi_adb_0)):
        intervals[n] = np.sum(np.real(pops_adb_0)[0:n+1])
    rand_val = np.random.rand()
    act_0 = (np.arange(len(pops_adb_0))[intervals > rand_val])[0]
    act_surf_ind = act_0
    nstates = len(pops_adb_0)
    act_surf = np.zeros(nstates, dtype=int)
    act_surf[act_surf_ind] = 1
    tdat = np.arange(0, tmax + dt, dt)
    tdat_bath = np.arange(0, tmax + dt_bath, dt_bath)
    den_mat_db_tot = np.zeros((len(tdat), nstates, nstates), dtype=complex)
    pops_db_tot = np.zeros((len(tdat), nstates), dtype=complex)
    pops_adb_tot = np.zeros((len(tdat), nstates), dtype=complex)
    eig_vals = np.zeros((len(tdat), nstates))
    pop_vals = np.zeros((len(tdat), nstates))
    flip_count = 0
    eclist = np.zeros(len(tdat))
    eqlist = np.zeros(len(tdat))
    b_pop_db = np.zeros((len(tdat), nstates, nstates), dtype=complex)
    cg_adb = psi_adb_0
    cg_db = sim.cg_db_0
    ec = operators.eC(p, q, sim.w_list)
    eq = eigval[act_surf_ind]
    e_init = ec + eq
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if sim.use_tqdm:
            pba.update.remote(1)
        if t_ind == len(tdat):
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath:
            den_mat_adb = np.dot(np.conj(cg_adb).reshape((-1, 1)), np.array([cg_adb]))
            den_mat_adb[range(nstates), range(nstates)] = act_surf  # put act_surf on diagonal of rho_adb
            den_mat_db = operators.rho_0_adb_to_db(den_mat_adb, eigvec)
            den_mat_db_tot[t_ind, :, :] = den_mat_db_tot[t_ind, :, :] + den_mat_db
            pops_db_tot[t_ind, :] = pops_db_tot[t_ind, :] + np.diag(den_mat_db)
            pops_adb_tot[t_ind, :] = pops_adb_tot[t_ind, :] + np.diag(den_mat_adb)
            b_pop_adb = np.diag(operators.boltz(eigval, sim)) + 0.0j
            b_p_db = operators.rho_0_adb_to_db(b_pop_adb, eigvec)
            b_pop_db[t_ind] = b_p_db
            pop_vals[t_ind, :] = np.real(act_surf)
            eig_vals[t_ind, :] = np.real(eigval)
            ec = operators.eC(p, q, sim.w_list)
            eq = eigval[act_surf_ind]
            eclist[t_ind] = ec
            eqlist[t_ind] = eq
            energy = ec + eq
            if np.abs(e_init - energy) > 0.01 * np.abs(e_init):
                print('ERROR energy not conserved ', (e_init - energy) / e_init * 100, '% error')
            if np.abs(np.sum(np.real(np.conj(cg_db) * cg_db)) - 1.0) > 1e-5:
                # print(pops_db_tot[t_ind])
                print('Diabatic norm not conserved: ', tdat[t_ind], np.real(np.conj(cg_db) * cg_db))
            if np.abs(np.sum(np.real(np.conj(cg_adb) * cg_adb)) - 1.0) > 1e-5:
                # print(pops_db_tot[t_ind])
                print('Adiabatic norm not conserved: ', tdat[t_ind], np.real(np.conj(cg_adb) * cg_adb))
                #    print('Active surface: ',act_surf_ind)
            #print(4)
            t_ind += 1
        fq, fp = operators.quantumForce_Ex(eigvec[:, act_surf_ind], sim)
        if sim.temp == 0.0:
            fq = fq * 0
            fp = fp * 0
        p, q = operators.RK4(p, q, (fq, fp), sim.w_list, dt_bath)
        eigvec_previous = np.copy(eigvec)
        qcmat = operators.H_qc_Ex(q, p, sim)
        full_mat = qmat + qcmat
        eigval, eigvec = np.linalg.eigh(full_mat)
        eigval_exp = np.exp(-1j * eigval * sim.dt_bath)
        diag_matrix = np.diag(eigval_exp)
        cg_adb = np.dot(diag_matrix, operators.vec_db_to_adb(cg_db, eigvec))
        cg_db = operators.vec_adb_to_db(cg_adb, eigvec)
        rand = np.random.rand()
        #prod_A1_0 = (np.matmul(np.conj(eigvec[:, act_surf_ind]), eigvec_previous))
        #prod_A1 = np.abs(prod_A1_0) * np.sign(np.real(prod_A1_0))

        #prod_B_0 = np.sum(np.conj(eigvec_previous) * eigvec, axis=0)
        # prod_B = np.abs(prod_B_0) #* np.sign(np.real(prod_B_0))
        # phase_B = prod_B_0 / prod_B
        #phase_B = np.exp(1.0j * np.angle(prod_B_0))
        #hop_prob = -2 * np.real(prod_A1_0 * (cg_adb / cg_adb[act_surf_ind]) * phase_B)
        if cg_adb[act_surf_ind] == 0.0+0.0j:
            print('ERROR: ')
        hop_prob = -2 * np.real( (cg_adb / cg_adb[act_surf_ind]))*0
        hop_prob[act_surf_ind] = 0
        bin_edge = 0
        for k in range(len(hop_prob)):
            hop_prob[k] = operators.nan_num(hop_prob[k])
            bin_edge = bin_edge + hop_prob[k]
            if rand < bin_edge:
                eig_k = eigvec[:, act_surf_ind]
                eig_j = eigvec[:, k]
                eigval_k = eigval[act_surf_ind]
                eigval_j = eigval[k]
                ev_diff = eigval_j - eigval_k
                dkkq, dkkp = operators.get_dkk_Ex(eig_k, eig_j, ev_diff, sim)
                #phase_A1 = prod_A1_0 / prod_A1
                #ab_phase = (phase_A1 * phase_B)[k]
                #dkkq = dkkq * np.conj(ab_phase)
                #dkkp = dkkp * np.conj(ab_phase)
                phase = np.imag(np.log(dkkq[np.argmax(np.abs(dkkq))]))
                dkkq = dkkq * np.exp(-1.0j * phase)
                phase = np.imag(np.log(dkkp[np.argmax(np.abs(dkkp))]))
                dkkp = dkkp * np.exp(-1.0j * phase)

                if np.max(np.abs(np.imag(dkkq)))/np.max(np.abs(np.real(dkkq))) > 1e-7:
                    import matplotlib.pyplot as plt
                    plt.scatter(np.real(dkkq), np.imag(dkkq))
                    plt.show()
                    print('ERROR IMAGINARY DKKQ: ', np.max(np.abs(np.imag(dkkq))), np.max(np.abs(np.real(dkkq))),np.max(np.abs(np.imag(dkkq)))/np.max(np.abs(np.real(dkkq))))
                if np.max(np.abs(np.imag(dkkp)))/np.max(np.abs(np.real(dkkp))) > 1e-7:
                    print('ERROR IMAGINARY DKKP: ', np.max(np.abs(np.imag(dkkp))), np.max(np.abs(np.real(dkkp))),np.max(np.abs(np.imag(dkkp)))/np.max(np.abs(np.real(dkkp))))
                delta_q = np.real(dkkp)
                delta_p = np.real(dkkq)
                akkq = np.sum((1 / 2) * np.abs(
                    delta_p) ** 2)  # (1 / 2) * np.sum(((w_list * dkkp_i) - dkkq_r) * ((w_list * dkkp_i) - dkkq_r))
                akkp = np.sum((1 / 2) * (sim.w_list ** 2) * np.abs(
                    delta_q) ** 2)  # (1 / 2) * np.sum((w_list ** 2) * (dkkp_r + (1/w_list) * dkkq_i) * (dkkp_r + (1/w_list) * dkkq_i))
                bkkq = np.sum((p * delta_p))
                bkkp = -np.sum((sim.w_list ** 2) * q * delta_q)
                disc = (bkkq + bkkp) ** 2 - 4 * (akkq + akkp) * ev_diff
                if disc >= 0:
                    if bkkq + bkkp < 0:
                        gamma = (bkkq + bkkp) + np.sqrt(disc)
                    else:
                        gamma = (bkkq + bkkp) - np.sqrt(disc)
                    if akkp + akkq == 0:
                        gamma = 0
                    else:
                        gamma = gamma / (2 * (akkq + akkp))

                    p = p - np.real(gamma) * delta_p  # rescale
                    q = q + np.real(gamma) * delta_q  # rescale
                    # print('hop',tdat_bath[t_bath_ind], act_surf_ind,eigval[act_surf_ind], k,eigval[k])
                    act_surf_ind = k
                    act_surf = np.zeros_like(act_surf)
                    act_surf[act_surf_ind] = 1
                    hop_count += 1
                # rev = False
                # if disc < 0 and rev:
                #    if akkp + akkq == 0:
                #        gamma = 0
                #    else:
                #        gamma = (bkkq + bkkp) / (akkq + akkp)
                #    p = p - np.real(gamma) * dkkq  # flip
                #    q = q + np.real(gamma) * dkkp  # flip
                #    flip_count += 1
                break
    pops_db = np.real(pops_db_tot)
    simCdb = pops_db
    simT = tdat
    simN = np.sum(pops_db, axis=1)
    simEq = eqlist
    simEc = eclist
    simPopB = b_pop_db
    simRhoDb = den_mat_db_tot
    end_time = time.time()
    msg = 'trial index: ' + str(index) + ' hop count: ' + str(hop_count) + ' flip count: ' + str(
        flip_count) + ' time: ' + str(end_time - start_time)
    return simCdb, simT, simN, simEq, simEc, simPopB, simRhoDb, msg
def parallel_run_ray_exciton(sim, mat):
    sim.print_dynamics_info()
    print('Starting dynamics calculation...')
    st = time.time()
    r_ind = 0
    if sim.nprocs > sim.trials:
        sim.nprocs = sim.trials
    num_ticks = len(range(round((sim.tmax / sim.dt_bath)))) * sim.nprocs
    ray_sim = ray.put(sim)
    ray_mat = ray.put(mat)
    for run in range(0, int(sim.trials / sim.nprocs)):
        if sim.use_tqdm:
            pb = ProgressBar(num_ticks)
            actor = pb.actor
        else:
            actor = 1
        p, q = operators.init_classical_parallel(sim)

        if sim.method == 'FSSH':
            results = [
                FSSH_dynamics_exciton.remote(run * sim.nprocs + i, p[i, :, :], q[i, :, :], ray_sim, ray_mat, actor) \
                for i in range(sim.nprocs)]
        if sim.use_tqdm:
            pb.print_until_done()
        for r in results:
            simCdb, simT, simN, simEq, simEc, simPopB, simRhoDb, msg = ray.get(r)
            print(msg)
            if run == 0 and r_ind == 0:
                simCdbdat = np.zeros_like(simCdb)
                simTdat = np.zeros_like(simT)
                simPopBdat = np.zeros_like(simPopB)
                simEcdat = np.zeros_like(simEc)
                simEqdat = np.zeros_like(simEq)
                simNdat = np.zeros_like(simN)
                simRhoDbdat = np.zeros_like(simRhoDb)
            simCdbdat += simCdb
            simTdat += simT
            simPopBdat += simPopB
            simEcdat += simEc
            simEqdat += simEq
            simNdat += simN
            simRhoDbdat += simRhoDb
            r_ind += 1
    filename = sim.calc_dir + '/' + sim.calc_dir
    if path.exists(filename + '_resCdbEX.csv'):
        simCdbdat += np.loadtxt(filename + '_resCdbEX.csv', delimiter=",")
    if path.exists(filename + '_resT.csv'):
        simTdat += np.loadtxt(filename + '_resT.csv', delimiter=",")
    if path.exists(filename + '_resPopBEXfull.npy'):
        simPopBdat += np.load(filename + '_resPopBEXfull.npy')
    if path.exists(filename + '_resRhoDBEXfull.npy'):
        simRhoDbdat += np.load(filename + '_resRhoDBEXfull.npy')
    if path.exists(filename + '_resEc.csv'):
        simEcdat += np.loadtxt(filename + '_resEc.csv', delimiter=",")
    if path.exists(filename + '_resEq.csv'):
        simEqdat += np.loadtxt(filename + '_resEq.csv', delimiter=",")
    if path.exists(filename + '_resN.csv'):
        simNdat += np.loadtxt(filename + '_resN.csv', delimiter=",")
    et = time.time()
    print('took ', np.round(et - st, 3), ' seconds')
    ray.shutdown()
    sim.resCdbEX = simCdbdat
    sim.resT = simTdat
    sim.resN = simNdat
    sim.resEq = simEqdat
    sim.resEc = simEcdat
    sim.resPopBEX = simPopBdat
    sim.resRhoDbEX = simRhoDbdat
    return sim

def dynamics_exciton(sim, mat):
    trunc_bse_evals = sim.trunc_bse_evals
    eval_block_list = sim.eval_block_list
    dk_id = sim.dk_id
    dk_id_sort = sim.dk_id_sort
    evec_block_list = sim.evec_block_list
    block_dims = sim.block_dims
    block_unsort = sim.block_unsort
    cg_db_0 = np.zeros((len(trunc_bse_evals)), dtype=complex)
    cg_db_0[sim.init_state_num] = 1
    num_of_states = len(eval_block_list)
    num_trunc_states = len(trunc_bse_evals)
    @ray.remote
    def transform_mat(mat, basis, i):
        return i, np.real(np.einsum('ii->i', np.einsum('ki,ij->kj', np.conj(np.transpose(basis)),
                                                       np.einsum('ij,jk->ik', mat, basis))))

    @ray.remote
    def transform_mat_diag(mat, basis, i):
        mat_1 = np.einsum('ij,jk->ik', mat, basis)
        out_vec = np.zeros((np.shape(mat_1)[1]))
        for n in range(np.shape(mat_1)[1]):
            out_vec[n] = np.real(np.dot(np.conj(basis[:, n]), mat_1[:, n]))
        return i, out_vec

    def transform(rho_t, basis):
        t_num = len(rho_t[:, 0, 0])
        out_dim = len(basis[0])
        out_array = np.empty((t_num, out_dim))
        for n in tqdm(range(int(t_num / sim.nprocs) + 1)):
            nlist = np.arange(n * sim.nprocs, (n + 1) * sim.nprocs, 1)
            nlist = nlist[nlist < t_num]
            if len(nlist) == 0:
                break
            results = [transform_mat_diag.remote(rho_t[i], basis, i) for i in nlist]
            for r in results:
                i, line = ray.get(r)
                out_array[i, :] = line
        return np.real(out_array)

    def get_evec(alpha):
        evec_block = np.ascontiguousarray(np.zeros((len(eval_block_list)))) + 0.0j
        dk_ran = dk_id == dk_id_sort[alpha]
        evec = evec_block_list[:, alpha][:block_dims[dk_id_sort[alpha]]]
        evec_block[dk_ran] = evec
        evec_out = evec_block[block_unsort]
        return evec_out

    sim.cg_db_0 = cg_db_0
    sim = parallel_run_ray_exciton(sim, mat)
    filename = sim.calc_dir + '/' + sim.calc_dir

    np.savetxt(filename + '_resCdbEX.csv', np.real(sim.resCdbEX), delimiter=",")
    np.savetxt(filename + '_resT.csv', sim.resT, delimiter=",")
    resPopBout = np.einsum('tii->ti', sim.resPopBEX)
    np.save(filename + '_resRhoDBEXfull.npy', sim.resRhoDbEX)
    np.save(filename + '_resPopBEXfull.npy', sim.resPopBEX)
    np.savetxt(filename + '_resPopBEX.csv', np.real(resPopBout), delimiter=",")
    np.savetxt(filename + '_resEc.csv', sim.resEc, delimiter=",")
    np.savetxt(filename + '_resEq.csv', sim.resEq, delimiter=",")
    np.savetxt(filename + '_resN.csv', sim.resN, delimiter=",")
    np.savetxt(filename + '_kgrid.csv', sim.k, delimiter=",")
    if sim.trans_to_k:
        print('Transforming to k-space...')
        evec_mat = np.zeros((num_of_states, num_trunc_states), dtype=complex)
        for n in range(0, num_trunc_states):
            evec_mat[:, n] = get_evec(n)
        resCdbK = transform(sim.resRhoDbEX, np.transpose(np.conj(evec_mat)))
        np.savetxt(filename + '_resCdb.csv', resCdbK, delimiter=",")
        resPopBK = transform(sim.resPopBEX, np.transpose(np.conj(evec_mat)))
        np.savetxt(filename + '_resPopB.csv', resPopBK, delimiter=",")

    return sim