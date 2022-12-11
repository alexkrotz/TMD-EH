import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def kgrid_plot(k, sim):
    fig = plt.figure(tight_layout=False, dpi=300)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0])
    ax.scatter(k[:,0]/np.pi,k[:,1]/np.pi,marker='.',color='black')
    fig.set_figwidth(4.0)
    fig.set_figheight(4.0)
    plt.savefig(sim.init_dir + '/k_plot.pdf')
    plt.close()
    return

def tb_psi_plot(k,eigvecs,sim):
    fig = plt.figure(tight_layout=False, dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])
    kx_mesh = k[:, 0].reshape((sim.res, sim.res)) / np.pi
    ky_mesh = k[:, 1].reshape((sim.res, sim.res)) / np.pi
    ax0.pcolormesh(kx_mesh, ky_mesh, np.real(eigvecs[0, :, 0, 0].reshape((sim.res, sim.res))))
    ax0.set_title(r'$\Re(\langle 0\vert \psi_{v}\rangle)$')
    ax1.pcolormesh(kx_mesh, ky_mesh, np.imag(eigvecs[0, :, 0, 0].reshape((sim.res, sim.res))))
    ax1.set_title(r'$\Im(\langle 0\vert \psi_{v}\rangle)$')
    ax2.pcolormesh(kx_mesh, ky_mesh, np.real(eigvecs[0, :, 1, 0].reshape((sim.res, sim.res))))
    ax2.set_title(r'$\Re(\langle 1\vert \psi_{v}\rangle)$')
    ax3.pcolormesh(kx_mesh, ky_mesh, np.imag(eigvecs[0, :, 1, 0].reshape((sim.res, sim.res))))
    ax3.set_title(r'$\Im(\langle 1\vert \psi_{v}\rangle)$')
    plt.savefig(sim.init_dir + '/reim_v_plot.pdf')
    plt.close()
    fig = plt.figure(tight_layout=False, dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])
    kx_mesh = k[:, 0].reshape((sim.res, sim.res)) / np.pi
    ky_mesh = k[:, 1].reshape((sim.res, sim.res)) / np.pi
    ax0.pcolormesh(kx_mesh, ky_mesh, np.real(eigvecs[0, :, 0, 1].reshape((sim.res, sim.res))))
    ax0.set_title(r'$\Re(\langle 0\vert \psi_{c}\rangle)$')
    ax1.pcolormesh(kx_mesh, ky_mesh, np.imag(eigvecs[0, :, 0, 1].reshape((sim.res, sim.res))))
    ax1.set_title(r'$\Im(\langle 0\vert \psi_{c}\rangle)$')
    ax2.pcolormesh(kx_mesh, ky_mesh, np.real(eigvecs[0, :, 1, 1].reshape((sim.res, sim.res))))
    ax2.set_title(r'$\Re(\langle 1\vert \psi_{c}\rangle)$')
    ax3.pcolormesh(kx_mesh, ky_mesh, np.imag(eigvecs[0, :, 1, 1].reshape((sim.res, sim.res))))
    ax3.set_title(r'$\Im(\langle 1\vert \psi_{c}\rangle)$')
    plt.savefig(sim.init_dir + '/reim_c_plot.pdf')
    #plt.show()
    plt.close()
    return