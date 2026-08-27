import numpy as np
from . import Exceptions as dapExceptions
from . import OBS_ERRORS

#Takes in an observation operator and a state in model space
#and returns the state in obs space

class Linear():
    def __init__(self, obsParams: dict, Nx):
        self.H = np.eye(Nx) #Linear Measurement Operator
        self.H = self.H[obsParams['obb']:Nx-obsParams['obb']:obsParams['obf'], :]

    def create_obs(self, x, true_obs_err_dist, true_obs_err_params, rng):
        Y_perf = np.matmul(self.H, x)[:, :, np.newaxis]
        Y = Y_perf + OBS_ERRORS.sample_errors(Y_perf, true_obs_err_dist, true_obs_err_params, rng)
        return Y

    def create_periodic(self, sigma, m, dx):
        if m % 2 == 0: #Even
                cx = m/2
                x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
        else: #Odd
                cx = np.floor(m/2)
                x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
        wlc = np.exp(-((dx*(x))**2)/(2*sigma**2))
        B = np.zeros((m, m))
        for i in range(m):
                B[i, :] = wlc[m - i:2*m - i]
        B = np.where(B < 0, 0, B)
        return B

    def apply_H(self, x):
        hx = np.matmul(self.H, x)
        return hx
        

class Squared():
    def __init__(self, obsParams: dict, Nx):
        self.H = np.eye(Nx) #Linear Measurement Operator
        self.H = self.H[obsParams['obb']:Nx-obsParams['obb']:obsParams['obf'], :]

    def create_obs(self, x, true_obs_err_dist, true_obs_err_params, rng):
        Y_perf = np.matmul(self.H, x**2)[:, :, np.newaxis]
        Y = Y_perf + OBS_ERRORS.sample_errors(Y_perf, true_obs_err_dist, true_obs_err_params, rng)
        return Y

    def create_periodic(self, sigma, m, dx):
        if m % 2 == 0: #Even
                cx = m/2
                x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
        else: #Odd
                cx = np.floor(m/2)
                x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
        wlc = np.exp(-((dx*(x))**2)/(2*sigma**2))
        B = np.zeros((m, m))
        for i in range(m):
                B[i, :] = wlc[m - i:2*m - i]
        B = np.where(B < 0, 0, B)
        return B

    def apply_H(self, x):
        hx = np.matmul(self.H, np.square(x))

class Log():
    def __init__(self, obsParams: dict, Nx):
        self.H = np.eye(Nx) #Linear Measurement Operator
        self.H = self.H[obsParams['obb']:Nx-obsParams['obb']:obsParams['obf'], :]

    def create_obs(self, x, true_obs_err_dist, true_obs_err_params, rng):
        Y_perf = np.matmul(self.H, np.log(np.abs(x)))[:, :, np.newaxis]
        Y = Y_perf + OBS_ERRORS.sample_errors(Y_perf, true_obs_err_dist, true_obs_err_params, rng)
        return Y

    def create_periodic(self, sigma, m, dx):
        if m % 2 == 0: #Even
                cx = m/2
                x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
        else: #Odd
                cx = np.floor(m/2)
                x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
        wlc = np.exp(-((dx*(x))**2)/(2*sigma**2))
        B = np.zeros((m, m))
        for i in range(m):
                B[i, :] = wlc[m - i:2*m - i]
        B = np.where(B < 0, 0, B)
        return B

    def apply_H(self, x):
        hx = np.matmul(self.H, np.log(np.abs(x)))
        return hx

import sys
np.set_printoptions(threshold=sys.maxsize)

class Linear3D():
    def __init__(self, obsParams: dict, modelParams: dict, Nx):
        nx = modelParams['model_params']['nx']
        ny = modelParams['model_params']['ny']
        self.H = np.eye(Nx) #Linear Measurement Operator
        self.H = self.H[obsParams['obb']:Nx-obsParams['obb'], :]

        x = np.arange(nx)[None, :]
        y = np.arange(ny)[:, None]

        mask_2d = (x % obsParams['obf_x'] == 0) & (y % obsParams['obf_y'] == 0)
        mask = np.tile(mask_2d.ravel(), 2)

        self.H = self.H[mask, :]

    def create_obs(self, x, true_obs_err_dist, true_obs_err_params, rng):
        Y_perf = np.matmul(self.H, x)[:, :, np.newaxis]
        Y = Y_perf + OBS_ERRORS.sample_errors(Y_perf, true_obs_err_dist, true_obs_err_params, rng)
        return Y

    def create_periodic(self, s_x, s_y, s_z, m_x, m_y, m_z, dx, dy, dz, dtype=np.float32):
        """
        Create an anisotropic 3D localization matrix.

        State ordering:
            (layer, x, y)

        x and y are periodic.
        z/layer is non-periodic.

        The resulting matrix has the structure

            B = B_z ⊗ B_h

        where B_h is the horizontal localization matrix and
        B_z is the vertical localization matrix.
        """

        # Number of horizontal grid points
        N_h = m_x * m_y

        # Horizontal periodic distances
        x_dist = np.minimum(np.arange(m_x), m_x - np.arange(m_x))
        y_dist = np.minimum(np.arange(m_y), m_y - np.arange(m_y))

        # Convert to physical distances
        x_dist = dx * x_dist
        y_dist = dy * y_dist

        # Horizontal localization kernel
        x_dist = x_dist[:, None]
        y_dist = y_dist[None, :]

        kernel_h = np.exp(-0.5 * ((x_dist / s_x)**2 + (y_dist / s_y)**2)).astype(dtype)

        # Construct horizontal localization matrix
        B_h = np.empty((N_h, N_h), dtype=dtype)

        row = 0

        for i in range(m_x):
            for j in range(m_y):

                weights = np.roll(kernel_h, shift=(i, j), axis=(0, 1))

                B_h[row, :] = weights.ravel()

                row += 1

        # Vertical/layer localization matrix
        layer_dist = np.abs(np.arange(m_z)[:, None] - np.arange(m_z)[None, :])

        layer_dist = dz * layer_dist

        B_z = np.exp(-0.5 * (layer_dist / s_z)**2).astype(dtype)

        # Combine vertical and horizontal localization
        B = np.kron(B_z, B_h)

        return B

    def apply_H(self, x):
        hx = np.matmul(self.H, x)
        return hx
            