import operators
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import diagonalize
np.random.seed(1234)
def test(mat, sim):
    #H_bse = operators.H_BSE(sim, mat)
    p,q = operators.init_classical(sim)
    #H_qc = operators.H_qc(q, p, sim)
    #evals, evecs = np.linalg.eigh(H_bse + H_qc)
    #print(evals[:20])
    H_bse_ex = operators.H_BSE_Ex(sim)
    H_qc_ex = operators.H_qc_Ex(q, p, sim)
    evals, evecs = np.linalg.eigh(H_bse_ex + H_qc_ex)
    print(evals[:20])
    e_spin = sim.trunc_e_spin_block_list
    h_spin = sim.trunc_h_spin_block_list
    for n in range(len(evals)):
        es = np.sum(np.conjugate(evecs[:,n])*evecs[:,n]*e_spin)
        hs = np.sum(np.conjugate(evecs[:,n])*evecs[:,n]*h_spin)
        print(n, evals[n],"ES HS", es,hs)
    dkkq, dkkp = operators.get_dkk_Ex(evecs[:, 0], evecs[:, 3], 1, sim)

    plt.scatter(sim.umk_point_list_MK[:, 0], sim.umk_point_list_MK[:, 1], marker='.', c=np.abs(dkkq))
    plt.show()
    plt.scatter(sim.umk_point_list_MK[:, 0], sim.umk_point_list_MK[:, 1], marker='.', c=np.abs(dkkp))
    plt.show()
    phase = np.imag(np.log(dkkq[np.argmax(np.abs(dkkq))]))
    dkkq = dkkq * np.exp(-1.0j * phase)
    phase = np.imag(np.log(dkkp[np.argmax(np.abs(dkkp))]))
    dkkp = dkkp * np.exp(-1.0j * phase)
    print(dkkq)
    plt.scatter(np.real(dkkq), np.imag(dkkq))
    plt.show()
    #print(sim.trunc_bse_evals)
    #print(sim.trunc_e_spin_block_list)
    #plt.plot(sim.bse_evals)
    #plt.show()
    #print(sim.bse_evals)
    #print(phase)
    #mat = np.zeros_like(H_bse)
    #pos = np.where(np.abs(H_bse) > 1e-15)
    #mat[pos] = 1
    #plt.imshow(np.abs(mat))
    #plt.show()
    #print(sim.se_id)
    #print(sim.sh_id)
    #print(sim.ke_id)
    #print(sim.kh_id)
    # check for singlet states that give imaginary rescaling
    #for n in tqdm(range(len(evals))):
    #    for m in range(len(evals)):
    #        if np.abs(np.sum(np.abs(evecs[:, n]) ** 2 * (2*sim.se_id - 1))) < 1e-5:
    #            if np.abs(np.sum(np.abs(evecs[:, m]) ** 2 * (2*sim.se_id - 1)))<1e-5:
    #                dkkq, dkkp = operators.get_dkk(evecs[:, n], evecs[:, m], 1, sim)
    #                phase = np.imag(np.log(dkkq[np.argmax(np.abs(dkkq))]))
    #                dkkq = dkkq * np.exp(-1.0j * phase)
    #                phase = np.imag(np.log(dkkp[np.argmax(np.abs(dkkp))]))
    #                dkkp = dkkp * np.exp(-1.0j * phase)
    #                if np.sum(np.abs(np.imag(dkkq))) > 1e-5 or np.sum(np.abs(np.imag(dkkp))) > 1e-5:
    #                    print(n,m,np.abs(np.sum(np.abs(evecs[:, n]) ** 2 * (2*sim.se_id - 1))),np.abs(np.sum(np.abs(evecs[:, m]) ** 2 * (2*sim.se_id - 1))),np.sum(np.abs(np.imag(dkkq))),np.sum(np.abs(np.imag(dkkp))))

    #dkkq, dkkp = operators.get_dkk(evecs[:,2],evecs[:,6],1,sim)
    #phase = np.imag(np.log(dkkq[np.argmax(np.abs(dkkq))]))
    #dkkq = dkkq * np.exp(-1.0j * phase)
    #phase = np.imag(np.log(dkkp[np.argmax(np.abs(dkkp))]))
    #dkkp = dkkp * np.exp(-1.0j * phase)
    ##print(dkkq)
    #plt.scatter(np.real(dkkq), np.imag(dkkq))
    #plt.show()
    exit()
    dkkq_0 = dkkq
    dkkp_0 = dkkp
    dkkq, dkkp = operators.get_dkk(evecs[:, 3], evecs[:, 4], 1, sim)
    phase = np.imag(np.log(dkkq[np.argmax(np.abs(dkkq))]))
    dkkq = dkkq * np.exp(-1.0j * phase)
    phase = np.imag(np.log(dkkp[np.argmax(np.abs(dkkp))]))
    dkkp = dkkp * np.exp(-1.0j * phase)
    print(phase)

    dkkq_1 = dkkq
    dkkp_1 = dkkp

    plt.scatter(np.real(dkkq_0), np.imag(dkkq_0))
    plt.scatter(np.real(dkkq_1), np.imag(dkkq_1))
    plt.show()
    plt.scatter(np.real(dkkp_0), np.imag(dkkp_0))
    plt.scatter(np.real(dkkp_1), np.imag(dkkp_1))
    plt.show()
    print(dkkq_1 + dkkq_0)
    print('dkkq: ', np.max(np.imag(dkkq_0 + dkkq_1)))
    print('dkkp: ', np.max(np.imag(dkkp_0 + dkkp_1)))

    return