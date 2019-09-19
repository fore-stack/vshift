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
#               (Conversion Function)
#
# While the conversion works like:
#
#

import numpy, model, joblib, os, utilities.filesystem, click, dotenv

from fastdtw import fastdtw

from tqdm import tqdm 

dotenv.load_dotenv()

# @
#
#
default_models_directory = os.getenv('models_directory')

default_cmu_directory = os.getenv('cmu_directory')

default_cmu_max_range = int( os.getenv('cmu_max_range') )

# @
#
#
def save_model_as(model, filename, directory=default_models_directory):
    
    filename = utilities.filesystem.extension(filename, '.pkl')

    joblib.dump(model, os.path.join(directory, filename) )

# @
#
#
def cmu_data_pipeline(source_data, target_data):
    
    source_data, target_data = model.extract_features(source_data), model.extract_features(target_data)

    source_data, target_data = model.pad_features(source_data), model.pad_features(target_data)

    source_data, target_data = model.align(source_data, target_data)

    source_data, target_data = source_data[:, 1:], target_data[:, 1:]

    source_data, target_data = model.pad_features(source_data), model.pad_features(target_data)

    source_data, target_data = model.apply_delta(source_data), model.apply_delta(target_data)

    return source_data, target_data

# @
#
#
def cmu_arctic_training(source_name, target_name, data_range=default_cmu_max_range, root_directory=default_cmu_directory):

    #
    source = utilities.filesystem.listdirectory( os.path.join(root_directory, source_name, 'wav') )

    target = utilities.filesystem.listdirectory( os.path.join(root_directory, target_name, 'wav') )

    #
    click.secho('Processing Arctic Dataset ({}-{}) ✓'.format(source_name, target_name), fg='green')

    source_dataset, target_dataset = [], []

    for index in tqdm( range(data_range) ) :

        source_data, target_data = model.load_audio(source[index]), model.load_audio(target[index])

        source_data, target_data = cmu_data_pipeline(source_data, target_data)

        source_dataset.append(source_data), target_dataset.append(target_data)

    source_dataset, target_dataset = numpy.asarray(source_dataset), numpy.asarray(target_dataset)

    joint_distribution = utilities.math.trim_zeros_frames( model.get_joint_matrix(source_dataset, target_dataset) )

    #
    click.secho('Training Gaussian Mixture Model ({}-{}) 📚'.format(source_name, target_name), fg='blue')

    gaussian_mixture_model = model.create_model()

    gaussian_mixture_model.fit(joint_distribution)

    save_model_as(gaussian_mixture_model, '{}-{}.pkl'.format(source_name, target_name) )

    click.secho('Training Finished on Gaussian Mixture Model ({}-{})'.format(source_name, target_name), fg='green')

    return gaussian_mixture_model

# @Benchmark
# Assessment with Carnegie Mellon University Arctic
# Dataset on Male-to-Female (BDL-CLB) and Scottish-to-Canadian (AWB-JMK)
def benchmark():

    bdl_clb_model = cmu_arctic_training('bdl', 'clb')

    awb_jmk_model = cmu_arctic_training('awb', 'jmk')

benchmark()