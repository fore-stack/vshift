# @Model Definitions
# Functional for easy-use and quick prototypinh
import numpy, scipy, pyworld, pysptk, sklearn.mixture, utilities.math, os, dotenv

import scipy.io.wavfile

from fastdtw import fastdtw

dotenv.load_dotenv()

# @Hyperparameters
# Can be set from the environment (.env)
# configuration file
default_sampling_rate = int( os.getenv('sampling_rate') )

default_alpha = pysptk.util.mcepalpha(default_sampling_rate)

default_order = int( os.getenv('order') )

default_frame_period = int( os.getenv('frame_period') )

default_hop_length = int(default_sampling_rate * 1e-2)

default_windows = [
    (0, 0, numpy.array([1.0]) ),
    
    (1, 1, numpy.array([-0.5, 0.0, 0.5]) ),
    
    (1, 1, numpy.array([1.0, -2.0, 1.0]) ),
]

default_padded_length = int( os.getenv('padded_length') )

default_max_iterations = int( os.getenv('max_iterations') )

# @Audio Loader
# Loads audio file into the right format
# accepts (string) filepath
# returns (int) sampling rate, (numpy_array) audio data
def load_audio(filepath):
    # TODO:
    # - add argument sampling rate to resample audio automatically
    # - add functionality to load audio files other than Wave (.wav)
    sampling_rate, audio_data = scipy.io.wavfile.read(filepath)

    return audio_data

# @Feature Extractor
# Extract the features from audio data
# accepts (numpy_array) audio data and optional (int) sampling rate
# returns (numpy_array) mel spectrum
def extract_features(audio_data, sampling_rate=default_sampling_rate):

    audio_data = audio_data.astype(numpy.float64)

    fundamental_frequency, time_axis = pyworld.dio(audio_data, sampling_rate)

    fundamental_frequency = pyworld.stonemask(audio_data, fundamental_frequency, time_axis, sampling_rate)

    spectrogram = pyworld.cheaptrick(audio_data, fundamental_frequency, time_axis, sampling_rate)

    spectrogram = utilities.math.trim_zeros_frames(spectrogram)

    mel_spectrum = pysptk.sp2mc(spectrogram, default_order, default_alpha)

    return mel_spectrum

# @Feature Pad
# Pad the features to the given length
# accepts (numpy_array) features
# returns (numpy_array) features
def pad_features(features, pad_length=default_padded_length):

    padding = numpy.zeros( (pad_length - features.shape[0], features.shape[1]) )

    return numpy.concatenate( (features, padding) )

# @Feature Alignment
# accepts (numpy_array) features one two, and optional (function: int, int) distance, and (int) radius
# returns aligned features (numpy_array) x and (numpy_array) y
def align(feature1, feature2, distance=utilities.math.melcd, radius=1):

    feature1, feature2 = utilities.math.trim_zeros_frames(feature1), utilities.math.trim_zeros_frames(feature2)
    
    distance, path = fastdtw(feature1, feature2, radius=radius, dist=distance)
    
    distance /= ( len(feature1) + len(feature2) )
    
    path_x = list( map(lambda l: l[0], path) )
    
    path_y = list( map(lambda l: l[1], path) )
    
    feature1, feature2 = feature1[path_x], feature2[path_y]

    return feature1, feature2

# @
#
#
def apply_delta(features, windows=default_windows):
    
    return utilities.math.apply_delta(features, windows)

# @
#
#
def get_joint_matrix(feature1, feature2):

    return numpy.concatenate( (feature1, feature2) , axis=-1).reshape(-1, feature1.shape[-1] * 2)

# @
#
#
def create_model(max_iterations=default_max_iterations):

    model = sklearn.mixture.GaussianMixture(n_components=64, covariance_type='full', max_iter=max_iterations)

    return model