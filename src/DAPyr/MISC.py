import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import warnings

def calc_SV(xa, xf):
      Nx, Ne = xa.shape
      sqrt_Ne = np.sqrt(1/(Ne - 1))
      xma = np.mean(xa, axis = 1)
      xmf = np.mean(xf, axis = 1)
      xpa, xpf = sqrt_Ne*(xa - xma[:, np.newaxis]), sqrt_Ne*(xf - xmf[:, np.newaxis])
      vals, vecs = np.linalg.eigh(np.matmul(xpf.T, xpf))
      val_sort = vals.argsort()[::-1]
      vals = vals[val_sort]
      vecs = vecs[:, val_sort]
      SVe = np.matmul(xpf, vecs)
      SVi = np.matmul(xpa, vecs)
      energy = np.sum(SVe*SVe, axis = 0)/np.sum(SVi*SVi, axis = 0)
      return SVi, SVe, energy, vals


def create_periodic(sigma, m, dx):
      if m % 2 == 0: #Even
            cx = m/2
            x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
      else: #Odd
            cx = np.floor(m/2)
            x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
      wlc = np.exp(-((dx*(x))**2)/(2*sigma*2))
      B = np.zeros((m, m))
      for i in range(m):
            B[i, :] = wlc[m - i:2*m - i]
      B = np.where(B < 0, 0, B)
      return B

def create_spherical_periodic(s_x, s_y, s_z, m_x, m_y, m_z, dx, dy, dz, dtype=np.float32):
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

    # Total number of state variables
    N = m_z * N_h

    # ---------------------------------------------------------
    # Horizontal periodic distances
    # ---------------------------------------------------------

    x_dist = np.minimum(
        np.arange(m_x),
        m_x - np.arange(m_x)
    )

    y_dist = np.minimum(
        np.arange(m_y),
        m_y - np.arange(m_y)
    )

    # Convert to physical distances
    x_dist = dx * x_dist
    y_dist = dy * y_dist

    # ---------------------------------------------------------
    # Horizontal localization kernel
    # ---------------------------------------------------------

    x_dist = x_dist[:, None]
    y_dist = y_dist[None, :]

    kernel_h = np.exp(
        -0.5 * (
            (x_dist / s_x)**2 +
            (y_dist / s_y)**2
        )
    ).astype(dtype)

    # ---------------------------------------------------------
    # Construct horizontal localization matrix
    # ---------------------------------------------------------

    B_h = np.empty(
        (N_h, N_h),
        dtype=dtype
    )

    row = 0

    for i in range(m_x):
        for j in range(m_y):

            weights = np.roll(
                kernel_h,
                shift=(i, j),
                axis=(0, 1)
            )

            B_h[row, :] = weights.ravel()

            row += 1

    # ---------------------------------------------------------
    # Vertical/layer localization matrix
    # ---------------------------------------------------------

    layer_dist = np.abs(
        np.arange(m_z)[:, None] -
        np.arange(m_z)[None, :]
    )

    layer_dist = dz * layer_dist

    B_z = np.exp(
        -0.5 * (layer_dist / s_z)**2
    ).astype(dtype)

    # ---------------------------------------------------------
    # Combine vertical and horizontal localization
    # ---------------------------------------------------------

    B = np.kron(B_z, B_h)

    return B

def find_beta(sum_exp, Neff):
    #sum_exp is of size Ne
    Ne = sum_exp.shape[0]
    beta_max = np.max([1, 10*np.max(sum_exp)])
    w = np.exp(-sum_exp)
    ws = np.sum(w)
    if ws > 0:
        w = w/ws
        Neff_init = 1/sum(w**2)
    else:
        Neff_init = 1
    
    if Neff == 1:
        return

    if Neff_init < Neff or ws == 0:
        ks, ke = 1, beta_max
        tol = 1E-5
        #Start Bisection Method

        for i in range(1000):
            w = np.exp(-sum_exp/ks)
            w = w/np.sum(w)
            fks = Neff - 1/np.sum(w**2)
            if np.isnan(fks):
                fks = Neff-1
            
            w = np.exp(-sum_exp/ke)
            w = w/np.sum(w)
            fke = Neff - 1/np.sum(w**2)

            km = (ke + ks)/2
            w = np.exp(-sum_exp/km)
            w = w/np.sum(w)
            fkm = Neff - 1/np.sum(w**2)
            if np.isnan(fkm):
                fkm = Neff-1
            if (ke-ks)/2 < tol:
                break

            if fkm*fks > 0:
                ks = km
            else:
                ke = km
            
        beta = km
        w = np.exp(-sum_exp/beta)
        w = w/np.sum(w)
        Nf = 1/np.sum(w**2)
    else:
        beta = 1
    return beta



def get_reg(Nx, Ne, C, hw, Neff, res, beta_max):
    beta = np.zeros((Nx, ))
    res_ind = np.where(res > 0.0)[0]
    beta[res <= 0.0] = beta_max
    Ny, Ne = hw.shape
    #hw is Ny x Ne
    for i in res_ind:
        wo = 0
        loc_one = np.where(C[:, i] == 1)[0]
        loc_not_one = np.where(C[:, i] != 1)[0]
        dum = np.empty_like(hw)
        dum[loc_one, :] = np.log(Ne*hw[loc_one, :]) 
        dum[loc_not_one, :] = np.log((Ne*hw[loc_not_one, :] - 1)*C[loc_not_one, i, None] + 1)
        for y in range(Ny):
            wo = wo - dum[y, :]
            wo = wo - np.min(wo)
        beta[i] = find_beta(wo, Neff)
        if res[i] < 1/beta[i]:
            beta[i] = 1/res[i]
            res[i] = 0
        else:
            res[i] = res[i] - 1/beta[i]

        beta[i] = np.min([beta[i], beta_max])
    return beta, res


# glue prior and resampled particles together given posterior moments
# that we're seeking to match and a localization length scale
# merging is done to match the vanilla pf solution when no localization is happening
# and to match the prior when we're at a state very far away from the current obs

def sampling(x, w, Ne):

    # Sort sample
    b = np.argsort(x)
    
    # Apply deterministic sampling by taking value at every 1/Ne quantile
    cum_weight = np.concatenate(([0], np.cumsum(w[b])))
    
    offset = 0.0
    base = 1 / (Ne - offset) / 2
    
    ind = np.zeros(Ne, dtype=int)
    k = 1
    for n in range(Ne):
        frac = base + (n / (Ne - offset))
        while cum_weight[k] < frac:
            k += 1
        ind[n] = k - 1
    ind = b[ind]

    # Replace removed particles with duplicated particles
    ind2 = -999*np.ones(Ne, dtype=int)
    for n in range(Ne):
        if np.sum(ind == n) != 0:
            ind2[n] = n
            dum = np.where(ind == n)[0]
            ind = np.delete(ind, dum[0])
    

    ind0 = np.where(ind2 == -999)[0]
    ind2[ind0] = ind
    ind = ind2
    
    return ind


def gaussian_L(x, y, r):
    return np.exp(-(y - x)**2 / (2 * r)).item()


#TODO Rewrite to not have to rely on scipy
def kddm(x, xo, w):
    Ne = len(w)
    sig = (max(x) - min(x)) / 6
    npoints = 300
    
    xmin = min(min(xo), min(x))
    xmax = max(max(xo), max(x))
    
    xd = np.linspace(xmin, xmax, npoints)
    qf = np.zeros_like(x)
    cdfxa = np.zeros_like(xd)
    
    for n in range(Ne):
        qf += (1 + erf((x - x[n]) / (np.sqrt(2) * sig))) / (2 * Ne)
        cdfxa += w[n] * (1 + erf((xd - xo[n]) / (np.sqrt(2) * sig))) / 2
    
    interp_func = interp1d(cdfxa, xd, bounds_error=False, fill_value="extrapolate")
    xa = interp_func(qf)
    
    if np.var(xa) < 1e-8:
        warnings.warn("Low variance detected in xa")
    
    if np.isnan(qf).any():
        warnings.warn("NaN values detected in qf")
    
    return xa

def gen_be(sigma, amp, m, dx):
    x = np.arange(1, m+1)

    # Create index array
    x1, x2 = np.meshgrid(x, x)
    x1 *= dx
    x2 *= dx
    B = amp*np.exp( - ((x1-x2)**2) / (2*sigma**2))
    return B

def gradJ(xb, v, cv, inov, U, R_inv, H):

    g = cv + v
    g = g + U.T@H.T@R_inv@(H@U@v - inov)
    return g


def cg_minimize(xg, xb, cv, inov, R_i, U, H, ntmax):

    # Set inner-loop control variable and x to zero
    v   = cv*0
    xv  = xg*0

    # Set initial d and r to be negative the gradient
    d = -1*gradJ(xb, v, cv, inov, U, R_i, H)
    r = d

    norm = np.zeros(ntmax+2)
    J    = np.zeros(ntmax+1)
    
    gdot0   = d @ d
    gdot1   = gdot0
    norm[0] = np.sqrt(gdot1)

    # disp([])
    #disp(['Target gradient norm will be ',num2str(norm(1)*1e-4)])
    #disp(' ')

    for i in range(1, ntmax+2):

        # Calculate Jb
        Jb = (cv + v) @ (cv + v) /2.0

        # Calculate Jo
        Jo = ( H@U@v - inov ).T @ R_i @ ( H@U@v - inov ) / 2.

        norm[i] = np.sqrt(gdot1)

        # disp([])
        # disp(['Iteration: ', num2str(i-1),' Grad J: ',num2str(norm(i+1)),' Jb: ',num2str(Jb),' Jo: ',num2str(Jo)])

        J[i-1] = Jb + Jo

        # Transform back to model space
        xcv = U@cv
        xv = U@v

        if norm[i]<norm[1]*1e-8:
            break

        # Calculate A*d
        dum1 = cv*0
        dum2 = inov*0
        Ad = gradJ(xb, d, dum1, dum2, U, R_i, H)

        # Update CG variables
        a = gdot1 / (d.T@Ad)
        r = r - a*Ad
        v = v + a*d
        gdot0 = gdot1
        gdot1 = r@r
        bet   = gdot1/gdot0
        d     = r + bet*d

    xcv = xcv + xv
    cv  = cv  + v
    
    return xcv, cv, J

