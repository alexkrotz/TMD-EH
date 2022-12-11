import numpy as np
def two_band_params(mat):
	material = mat.material
	if material == "WSe2":
		t = 1.19  										# eV
		lambda_c = mat.lambda_c#36.0e-3								# eV
		lambda_v = mat.lambda_v#466e-3								# eV
		E_g = 1.60  # eV
	elif material == "WS2":
		t = 1.37										# eV
		lambda_c = mat.lambda_c#26.0e-3								# eV
		lambda_v = mat.lambda_v#430e-3								# eV
		E_g = 1.79  # eV
	elif material == "MoSe2":
		t = 0.94  										# eV
		lambda_c = mat.lambda_c#-31.0e-3								# eV
		lambda_v = mat.lambda_v#184e-3								# eV
		E_g = 1.47  # eV
	else: # MoS2
		t = 1.10										# eV
		lambda_c = mat.lambda_c#-3.0e-3								# eV
		lambda_v = mat.lambda_v#148e-3								# eV
		E_g = 1.66  # eV
	return (E_g, lambda_c, lambda_v, t)

def two_band_noCurve(q, tau, spin, mat):
	(E_g, lambda_c, lambda_v, t) = two_band_params(mat)
	E_val = -0.5 * E_g #+ 0.5 * tau * lambda_v * spin
	E_con = 0.5 * E_g #+ 0.5 * tau * lambda_c * spin
	E_split = E_con - E_val
	E_av = 0.5 * (E_con + E_val)
	E_av_split = 0.5 * (0.5 * tau* lambda_v * spin - 0.5 * tau * lambda_c * spin)
	epsilon = 0.5 * np.sqrt(np.square(E_split) + 4 * np.square(t) * np.sum(np.square(q), 1))
	a = E_split / 2
	b = t * tau * q[:, 0]
	c = t * q[:, 1]
	coef_c = np.column_stack((np.ones(epsilon.size), -(a - epsilon) / (b - 1j * c - 1j * 1e-10)))
	coef_v = np.column_stack((np.ones(epsilon.size), -(a + epsilon) / (b - 1j * c - 1j * 1e-10)))
	coef_c_norms = np.apply_along_axis(np.linalg.norm, 1, coef_c)
	coef_c /= np.column_stack((coef_c_norms, coef_c_norms))
	coef_v_norms = np.apply_along_axis(np.linalg.norm, 1, coef_v)
	coef_v /= np.column_stack((coef_v_norms, coef_v_norms))
	return epsilon + 0.5*tau*lambda_c*spin , -epsilon + 0.5*tau*lambda_v*spin, coef_c, coef_v

def two_band(q, tau, spin, mat):
	(E_g, lambda_c, lambda_v, t) = two_band_params(mat)
	E_val = -0.5 * E_g + 0.5 * tau * lambda_v * spin
	E_con = 0.5 * E_g + 0.5 * tau * lambda_c * spin
	E_split = E_con - E_val
	E_av = 0.5 * (E_con + E_val)

	epsilon = 0.5 * np.sqrt(np.square(E_split) + 4 * np.square(t) * np.sum(np.square(q), 1))

	a = E_split / 2
	b = t * tau * q[:, 0]
	c = t * q[:, 1]

	coef_c = np.column_stack((np.ones(epsilon.size), -(a - epsilon) / (b - 1j * c - 1j * 1e-10)))
	coef_v = np.column_stack((np.ones(epsilon.size), -(a + epsilon) / (b - 1j * c - 1j * 1e-10)))

	coef_c_norms = np.apply_along_axis(np.linalg.norm, 1, coef_c)
	coef_c /= np.column_stack((coef_c_norms, coef_c_norms))
	coef_v_norms = np.apply_along_axis(np.linalg.norm, 1, coef_v)
	coef_v /= np.column_stack((coef_v_norms, coef_v_norms))

	return epsilon + E_av, -epsilon + E_av, coef_c, coef_v

def two_band_noCurve_new(q, tau, spin, mat):
	(E_g, lambda_c, lambda_v, t) = two_band_params(mat)
	e1 = -0.5 * E_g
	e2 = 0.5 * E_g
	de = e2 - e1
	r = t*np.sqrt(np.sum(np.square(q),1))
	r_ep = t*(tau * q[:,0] + 1.0j*q[:,1])
	epsilon = np.sqrt(np.square(de/2) + np.square(r))
	e = 0.5*E_g
	r_0 = np.where(r < 1e-10)
	r_1 = r
	r_1[r_0] = 1
	ep = r_ep/r_1
	ep[r_0] = 1
	coef_c = np.column_stack((ep * (-e + epsilon), r))
	coef_v = np.column_stack((r_ep, e - epsilon))
	coef_c_norms = np.apply_along_axis(np.linalg.norm, 1, coef_c)
	coef_c /= np.column_stack((coef_c_norms, coef_c_norms))
	coef_v_norms = np.apply_along_axis(np.linalg.norm, 1, coef_v)
	coef_v /= np.column_stack((coef_v_norms, coef_v_norms))
	return epsilon + 0.5*tau*lambda_c*spin , -epsilon + 0.5*tau*lambda_v*spin, coef_c, coef_v

def two_band_noCurve_noBerry(q, tau, spin, mat):
	(E_g, lambda_c, lambda_v, t) = two_band_params(mat)
	e1 = -0.5 * E_g
	e2 = 0.5 * E_g
	de = e2 - e1
	r = t*np.sqrt(np.sum(np.square(q),1))
	epsilon = np.sqrt(np.square(de/2) + np.square(r))
	e = 0.5*E_g
	coef_c = np.column_stack((-e + epsilon, r))
	coef_v = np.column_stack((r, e - epsilon))
	coef_c_norms = np.apply_along_axis(np.linalg.norm, 1, coef_c)
	coef_c /= np.column_stack((coef_c_norms, coef_c_norms))
	coef_v_norms = np.apply_along_axis(np.linalg.norm, 1, coef_v)
	coef_v /= np.column_stack((coef_v_norms, coef_v_norms))
	return epsilon + 0.5*tau*lambda_c*spin , -epsilon + 0.5*tau*lambda_v*spin, coef_c, coef_v