# TODO:
# - improve documentation
# - support for audio file formats other than (.wav)
# - integrate audio resampling and stereo-to-mono

# @Main
# Access the core features of the software
# The voice conversion system's training function works like:
#  (Source Audio)                    (Target Audio)
#         |                                 |
#         |____(WORLD Vocoder Features)_____|
#                          |
#                          |
#                (Dynamic Time Warping)
#                          |
#                          |
#               (Gaussian Mixture Model)
#                          |
#                          |
#                (Conversion Function)
#
# While the conversion works like:
#                   (Source Audio) 
#                          |
#                          |
#                  (Speech Analysis)
#                          |
#                          |
#                 (WORLD Vocoder Features)
#                          |
#                          |
#        (Gaussian Mixture Model Voice Conversion)
#                          |
#                          |
#              (WORLD Vocoder Resynthesis)

import numpy, scipy.io, sklearn.metrics, model, joblib, os, utilities.math, utilities.filesystem, click, dotenv

from fastdtw import fastdtw

from tqdm import tqdm 

default_sampling_rate = int( os.getenv('sampling_rate') )

default_padded_length = int( os.getenv('padded_length') )

dotenv.load_dotenv()

# @Defaults
# These defaults can be set from the environment file
default_models_directory = os.getenv('models_directory')

default_audio_directory = os.getenv('audio_directory')

default_cmu_directory = os.getenv('cmu_directory')

default_cmu_max_range = int( os.getenv('cmu_max_range') )

default_skip_frames = int( os.getenv('skip_frames') )

default_auto_pad_length = int( os.getenv('auto_pad_length') )

# @Save Model
# Save models as pickle files
# accepts (gmm) model, (string) filename, and optional (string) directory
def save_model_as(model, filename, directory=default_models_directory):
    
    filename = utilities.filesystem.extension(filename, '.pkl')

    joblib.dump(model, os.path.join(directory, filename) )

# @Generic Pipeline
# This is what usually happens given source data and target data
# accepts (numpy_array) source data, and (numpy_array) target data, and optional (int) padded length
def generic_data_pipeline(source_data, target_data, padded_length=default_padded_length):
    
    source_data, target_data = model.extract_features(source_data), model.extract_features(target_data)
    
    source_data, target_data = model.pad_features(source_data, padded_length), model.pad_features(target_data, padded_length)

    source_data, target_data = model.align(source_data, target_data)

    source_data, target_data = source_data[:, 1:], target_data[:, 1:]

    source_data, target_data = model.pad_features(source_data, padded_length), model.pad_features(target_data, padded_length)

    source_data, target_data = model.apply_delta(source_data), model.apply_delta(target_data)

    return source_data, target_data

# @CMU Training Routine
# Computes the models for CMU Dataset
# accepts (string) source name, and (string) target name and optionals
def cmu_arctic_training(source_name, target_name, data_range=default_cmu_max_range, root_directory=default_cmu_directory):

    #
    source = utilities.filesystem.listdirectory( os.path.join(root_directory, source_name, 'wav') )

    target = utilities.filesystem.listdirectory( os.path.join(root_directory, target_name, 'wav') )

    #
    click.secho('Processing Arctic Dataset ({}-{}) 🧠 '.format(source_name, target_name), fg='blue')

    source_dataset, target_dataset = [], []

    for index in tqdm( range(data_range) ) :

        source_data, target_data = model.load_audio(source[index]), model.load_audio(target[index])

        source_data, target_data = generic_data_pipeline(source_data, target_data)

        source_dataset.append(source_data), target_dataset.append(target_data)

    source_dataset, target_dataset = numpy.asarray(source_dataset), numpy.asarray(target_dataset)

    joint_distribution = utilities.math.remove_zeros_frames( model.get_joint_matrix(source_dataset, target_dataset) )

    #
    click.secho('Training Gaussian Mixture Model ({}-{}) 📚'.format(source_name, target_name), fg='blue')

    gaussian_mixture_model = model.create_model()

    gaussian_mixture_model.fit(joint_distribution)

    save_model_as(gaussian_mixture_model, '{}-{}.pkl'.format(source_name, target_name) )

    click.secho('Training Finished on Gaussian Mixture Model ({}-{}) ✓'.format(source_name, target_name), fg='green')

    return gaussian_mixture_model

# @Benchmark
# Assessment with Carnegie Mellon University Arctic
# Dataset on Male-to-Female (BDL-CLB) and Scottish-to-Canadian (AWB-JMK)
def benchmark():

    cmu_arctic_training('bdl', 'clb')

    cmu_arctic_training('awb', 'jmk')

# @Analyze
# Routine where parallel corpora is 
# computed as a joint matrix and trains
# Gaussian Mixture Model
def analyze(source_path, target_path):

    click.secho('Loading Audio Files: {} {} 🔍 '.format(source_path, target_path), fg='blue')

    # source_dataset, target_dataset = [], []

    # source_data, target_data = model.extract_features( model.load_audio(source_path) ), model.extract_features(model.load_audio(target_path))

    # source_data, target_data = model.align(source_data, target_data)

    # shorter = min([ source_data.shape[0], target_data.shape[0] ])

    # pad_length = max([ source_data.shape[0], target_data.shape[0] ]) + 200

    # for index in tqdm( range( int( (shorter - default_padded_length) / default_skip_frames) - 1 ) ):

    #     current_source_data = source_data[index + (index * default_skip_frames):index + default_padded_length + (index * default_skip_frames), :]

    #     current_target_data = target_data[index + (index * default_skip_frames):index + default_padded_length + (index * default_skip_frames), :]

    #     current_source_data, current_target_data = model.pad_features(current_source_data, pad_length=pad_length), model.pad_features(current_target_data, pad_length=pad_length)

    #     current_source_data, current_target_data = current_source_data[:, 1:], current_target_data[:, 1:]

    #     current_source_data, current_target_data = model.apply_delta(current_source_data), model.apply_delta(current_target_data)

    #     source_dataset.append(current_source_data), target_dataset.append(current_target_data)

    # source_dataset, target_dataset = numpy.asarray(source_dataset), numpy.asarray(target_dataset)

    # joint_distribution = utilities.math.remove_zeros_frames( model.get_joint_matrix(source_dataset, target_dataset) )

    source_data = model.load_audio(source_path)

    target_data = model.load_audio(target_path)

    padded_length = max([ source_data.shape[0], target_data.shape[0] ]) + default_auto_pad_length

    source_data, target_data = generic_data_pipeline(source_data, target_data, padded_length)

    joint_distribution = utilities.math.remove_zeros_frames( model.get_joint_matrix(source_data, target_data) )

    gaussian_mixture_model = model.create_model()

    click.secho('Training Gaussian Mixture Model 🎓 '.format(source_path, target_path), fg='blue')

    gaussian_mixture_model.fit(joint_distribution)

    start_name = utilities.filesystem.extension(os.path.basename(source_path), '')

    end_name = utilities.filesystem.extension(os.path.basename(target_path), '')

    save_model_as(gaussian_mixture_model, '{}-{}.pkl'.format(start_name, end_name) )

    click.secho('Training Finished on Gaussian Mixture Model ({}-{}) ✓'.format(start_name, end_name), fg='green')
    
# @Convert
# Voice conversion with the Gaussian Mixture Model
def convert(model_path, audio_path):

    click.secho('Loading Matrix File: {} 🔍 '.format(model_path), fg='blue')

    loaded_model = joblib.load(model_path)

    click.secho('Converting Audio: {} 🔢'.format(audio_path), fg='blue')
    
    converted = model.gaussian_voice_conversion(loaded_model, audio_path)

    start_name = utilities.filesystem.extension(os.path.basename(model_path), '')

    end_name = utilities.filesystem.extension(os.path.basename(audio_path), '')

    save_location = os.path.join(default_audio_directory, '{}-{}.wav'.format(start_name, end_name) )
    
    scipy.io.wavfile.write(save_location, default_sampling_rate, converted)

    click.secho('Successfully Converted Audio ({}-{}) ✓'.format(start_name, end_name), fg='green')