import numpy as np


lattice_vec_1 = 2 * np.pi / np.sqrt(3.0) * np.array([np.sqrt(3.0), -1.0])
lattice_vec_2 = 4 * np.pi / np.sqrt(3.0) * np.array([0.0, 1.0])
def k_Umklapp(k, centers):
    mask = np.zeros(k.shape[0], dtype=int)
    for k_id in range(k.shape[0]): \
            mask[k_id] = np.argmin(np.round(np.apply_along_axis(np.linalg.norm, 1, k[k_id] - centers), 7))
    return k - centers[mask]
def Monkhorst_Pack(res):
    lattice_vec_1 = 2 * np.pi / np.sqrt(3.0) * np.array([np.sqrt(3.0), -1.0])
    lattice_vec_2 = 4 * np.pi / np.sqrt(3.0) * np.array([0.0, 1.0])
    grid = []
    for res_id_1 in range(res):
        p = res_id_1 + 1
        up = (2 * p - res - 1) / (2.0 * res)
        for res_id_2 in range(res):
            r = res_id_2 + 1
            ur = (2 * r - res - 1) / (2.0 * res)
            grid.append(up * lattice_vec_1 + ur * lattice_vec_2)
    return np.array(grid)

def find_index_trunc(k,points):
    out_ind = np.zeros((len(k)))
    for k_n in range(len(k)):
        norms = np.linalg.norm(k[k_n] - points,axis=1)
        if np.min(norms) < 1e-5:
            out_ind[k_n] = np.argmin(norms)
        else:
            out_ind[k_n] = -1
    return out_ind
def k_to_q(k, K_points, K_points_tau):
    mask = np.zeros(k.shape[0], dtype=int)
    for k_id in range(k.shape[0]):
        mask[k_id] = np.argmin(
            np.around(np.apply_along_axis(np.linalg.norm, 1, k[k_id] - K_points), decimals=4))
    return k - K_points[mask], K_points_tau[mask]

def k_Umklapp2(k, centers):
    k_array = np.zeros((len(k), len(centers), 2))
    center_array = np.zeros((len(k), len(centers), 2))
    center_array[:] = centers
    for n in range(len(centers)):
        k_array[:, n, :] = k - center_array[:, n, :]
    x_array = np.abs(k_array[:, :, 0])
    y_array = np.abs(k_array[:, :, 1])
    norm_array = np.round(np.sqrt(x_array ** 2 + y_array ** 2), 7)
    mask_array = np.argmin(norm_array, axis=1)
    return k - centers[mask_array]

# TODO make adjust grid more efficient
def adjust_grid(grid,mk):
    out_grid = np.zeros_like(grid)
    tolerance = 1e-5
    for n in range(len(grid)):
        if np.min(np.linalg.norm(grid[n]-mk,axis=1)) < tolerance:
            out_grid[n,:] = grid[n,:]
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_1 + lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_1 + lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_1 + lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_1 + lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_2 + lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_2 + lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_2 + lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_2 + lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_1 - lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_1 - lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_1 - lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_1 - lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_2 - lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_2 - lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_2 - lattice_vec_2, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_2 - lattice_vec_2
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_1 + lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_1 + lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_1 + lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_1 + lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_2 + lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_2 + lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_2 + lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_2 + lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_1 - lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_1 - lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_1 - lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_1 - lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk - lattice_vec_2 - lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] - lattice_vec_2 - lattice_vec_1
        elif np.min(np.linalg.norm(grid[n] - mk + lattice_vec_2 - lattice_vec_1, axis=1)) < tolerance:
            out_grid[n, :] = grid[n, :] + lattice_vec_2 - lattice_vec_1
        else:
            print('ADJUST GRID ERROR!!!')
    return out_grid

def find_index(k, points):  # find index of k in points
    k_array = k[:, np.newaxis, :]
    center_array = np.zeros((len(k), len(points), 2))
    center_array[:] = points
    index_array = np.zeros((len(k), len(points)))
    index_array[:] = np.arange(len(points), dtype=int)
    k_array = np.abs(k_array - center_array)
    x_array_mask = k_array[:, :, 0] < 1e-6
    y_array_mask = k_array[:, :, 1] < 1e-6
    mask_array = x_array_mask * y_array_mask
    index = index_array[mask_array].flatten()
    return index