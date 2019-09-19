# @Math Utility
# New mathematical operations
# that are not originally defined
import math, numpy

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
