# @Math Utility
# New mathematical operations
# that are not originally defined
import math, numpy, numpy.linalg, bandmat, bandmat.linalg, sklearn.mixture

from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

# @Trim Zero Frames
# Remove frames with no value
# accepts (numpy_array) x, (int) eps, (string) trim
# returns (numpy_array) x
def trim_zeros_frames(x, eps=1e-7, trim='b'):

    s = numpy.sum(numpy.abs(x), axis=1)

    s[s < eps] = 0.

    if trim == 'f':

        return x[len(x) - len(numpy.trim_zeros(s, trim=trim)):]

    elif trim == 'b':

        end = len(numpy.trim_zeros(s, trim=trim)) - len(x)

        if end == 0:

            return x

        else:

            return x[: end]

    elif trim == 'fb':

        f = len(numpy.trim_zeros(s, trim='f'))

        b = len(numpy.trim_zeros(s, trim='b'))

        end = b - len(x)

        if end == 0:

            return x[len(x) - f:]

        else:

            return x[len(x) - f: end]

_logdb_const = 10.0 / numpy.log(10.0) * numpy.sqrt(2.0)

def remove_zeros_frames(x, eps=1e-7):
    T, D = x.shape
    s = numpy.sum(numpy.abs(x), axis=1)
    s[s < eps] = 0.
    return x[s > eps]

# @Custom Square Root
# accepts (numpy_array) x
# returns (numpy_array) x
def _sqrt(x):
    
    isnumpy = isinstance(x, numpy.ndarray)

    isscalar = numpy.isscalar(x)

    return numpy.sqrt(x) if isnumpy else math.sqrt(x) if isscalar else x.sqrt()

# @Custom Sum Function
# accepts (numpy_array) x
# returns (numpy_array) x
def _sum(x):

    if isinstance(x, list) or isinstance(x, numpy.ndarray):

        return numpy.sum(x)

    return float( x.sum() )

# @Mel-Cepstrum Distortion
# accepts (numpy_array) X, (numpy_array) Y and optional (int) lengths
# returns (float) mel-cepstrum distortion
def melcd(X, Y, lengths=None):

    if lengths is None:
        
        z = X - Y

        r = _sqrt((z * z).sum(-1))

        if not numpy.isscalar(r):

            r = r.mean()

        return _logdb_const * float(r)


    # Case for 1-dim features.
    if len(X.shape) == 2:

        # Add feature axis

        X, Y = X[:, :, None], Y[:, :, None]

    s = 0.0

    T = _sum(lengths)

    for x, y, length in zip(X, Y, lengths):

        x, y = x[:length], y[:length]
        
        z = x - y

        s += _sqrt((z * z).sum(-1)).sum()

    return _logdb_const * float(s) / float(T)

# @
#
#
def _delta(x, window):

    return numpy.correlate(x, window, mode="same")

# @
#
#
def _apply_delta_window(x, window):
    T, D = x.shape

    y = numpy.zeros_like(x)

    for d in range(D):

        y[:, d] = _delta(x[:, d], window)

    return y

# @
#
#
def apply_delta(x, windows):
    
    T, D = x.shape

    combined_features = numpy.empty((T, D * len(windows)), dtype=x.dtype)

    for idx, (_, _, window) in enumerate(windows):

        combined_features[:, D * idx:D * idx + D] = _apply_delta_window(x, window)

    return combined_features

# @
#
#
def build_win_mats(windows, T):
    
    win_mats = []

    for l, u, win_coeff in windows:

        win_coeffs = numpy.tile(numpy.reshape(win_coeff, (l + u + 1, 1)), T)

        win_mat = bandmat.band_c_bm(u, l, win_coeffs).T

        win_mats.append(win_mat)

    return win_mats

# @
#
#
def build_poe(b_frames, tau_frames, win_mats, sdw=None):
    if sdw is None:

        sdw = max([win_mat.l + win_mat.u for win_mat in win_mats])

    num_windows = len(win_mats)

    frames = len(b_frames)

    b = numpy.zeros((frames,))

    prec = bandmat.zeros(sdw, sdw, frames)

    for win_index, win_mat in enumerate(win_mats):

        bandmat.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)

        bandmat.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec, diag=tau_frames[:, win_index])

    return b, prec

#
#
#
def mlpg(mean_frames, variance_frames, windows):
    
    dtype = mean_frames.dtype
    
    T, D = mean_frames.shape
    
    # expand variances over frames
    if variance_frames.ndim == 1 and variance_frames.shape[0] == D:

        variance_frames = numpy.tile(variance_frames, (T, 1))

    static_dim = D // len(windows)

    num_windows = len(windows)

    win_mats = build_win_mats(windows, T)

    means = numpy.zeros((T, num_windows))

    precisions = numpy.zeros((T, num_windows))

    y = numpy.zeros((T, static_dim), dtype=dtype)

    for d in range(static_dim):

        for win_idx in range(num_windows):

            means[:, win_idx] = mean_frames[:, win_idx * static_dim + d]

            precisions[:, win_idx] = 1 / variance_frames[:, win_idx * static_dim + d]

        bs = precisions * means

        b, P = build_poe(bs, precisions, win_mats)

        y[:, d] = bandmat.linalg.solveh(P, b)

    return y
#
#
#
class MLPGBase(object):

    def __init__(self, gmm, swap=False, diff=False):

        # D: static + delta dim
        D = gmm.means_.shape[1] // 2

        self.num_mixtures = gmm.means_.shape[0]

        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, :D]

        self.tgt_means = gmm.means_[:, D:]

        self.covarXX = gmm.covariances_[:, :D, :D]

        self.covarXY = gmm.covariances_[:, :D, D:]

        self.covarYX = gmm.covariances_[:, D:, :D]

        self.covarYY = gmm.covariances_[:, D:, D:]

        if diff:

            self.tgt_means = self.tgt_means - self.src_means

            self.covarYY = self.covarXX + self.covarYY - self.covarXY - self.covarYX

            self.covarXY = self.covarXY - self.covarXX

            self.covarYX = self.covarXY.transpose(0, 2, 1)

        # swap src and target parameters
        if swap:

            self.tgt_means, self.src_means = self.src_means, self.tgt_means

            self.covarYY, self.covarXX = self.covarXX, self.covarYY

            self.covarYX, self.covarXY = self.covarXY, self.covarYX

        # p(x), which is used to compute posterior prob. for a given source
        # spectral feature in mapping stage.
        self.px = sklearn.mixture.GaussianMixture(n_components=self.num_mixtures, covariance_type="full")

        self.px.means_ = self.src_means

        self.px.covariances_ = self.covarXX

        self.px.weights_ = self.weights

        self.px.precisions_cholesky_ = _compute_precision_cholesky(self.px.covariances_, "full")

    def transform(self, src):

        if src.ndim == 2:

            tgt = numpy.zeros_like(src)

            for idx, x in enumerate(src):

                y = self._transform_frame(x)

                tgt[idx][:len(y)] = y

            return tgt

        else:
            
            return self._transform_frame(src)

    def _transform_frame(self, src):

        D = len(src)

        # Eq.(11)
        E = numpy.zeros((self.num_mixtures, D))

        for m in range(self.num_mixtures):

            xx = numpy.linalg.solve(self.covarXX[m], src - self.src_means[m])

            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx)

        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(numpy.atleast_2d(src))

        # Eq.(13) conditinal mean E[p(y|x)]
        return posterior.dot(E).flatten()


class MLPG(MLPGBase):

    def __init__(self, gmm, windows=None, swap=False, diff=False):

        super(MLPG, self).__init__(gmm, swap, diff)
        
        if windows is None:
            windows = [
                
                (0, 0, numpy.array([1.0]) ),

                (1, 1, numpy.array([-0.5, 0.0, 0.5]) ),

            ]

        self.windows = windows

        self.static_dim = gmm.means_.shape[-1] // 2 // len(windows)

    def transform(self, src):

        T, feature_dim = src.shape[0], src.shape[1]

        if feature_dim == self.static_dim:

            return super(MLPG, self).transform(src)

        # A suboptimum mixture sequence  (eq.37)
        optimum_mix = self.px.predict(src)

        # Compute E eq.(40)
        E = numpy.empty((T, feature_dim))

        for t in range(T):

            m = optimum_mix[t]  # estimated mixture index at time t

            xx = numpy.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            
            # Eq. (22)
            E[t] = self.tgt_means[m] + numpy.dot(self.covarYX[m], xx)

        # Compute D eq.(23)
        # Approximated variances with diagonals so that we can do MLPG
        # efficiently in dimention-wise manner
        D = numpy.empty((T, feature_dim))

        for t in range(T):

            m = optimum_mix[t]
            
            # Eq. (23), with approximating covariances as diagonals
            D[t] = numpy.diag(self.covarYY[m]) - numpy.diag(self.covarYX[m]) / numpy.diag(self.covarXX[m]) * numpy.diag(self.covarXY[m])

        # Once we have mean and variance over frames, then we can do MLPG
        return mlpg(E, D, self.windows)
