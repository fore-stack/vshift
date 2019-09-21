# VShift &middot; ![GitHub repo size](https://img.shields.io/github/repo-size/ralphlouisgopez/vshift) ![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fralphlouisgopez%2Fvshift)

Real-Time High Quality Voice Conversion Software

## Installation
There are no precompiled binaries yet. A virtual environment known as [pipenv](https://github.com/pypa/pipenv) is utilized to manage the dependencies.

## Usage
A command-line interface can be run via interpreter. Use help to see the arguments of the interface.

There are two-steps in the process of voice-conversion:
- Analysis - where the source speaker's voice and target speaker's voice are analyzed and conversion matrices are computed.
- Conversion - where the source speaker's voice is converted to the target speaker's voice using the conversion matrices.

To use the system, install dependencies with `pipenv install`, after that activate virtual shell with `pipenv shell`, then on the root directory of the project `cd source` and run `python cli.py --help` for help. Use the environment file inside the source directory to change hyperparameters of the models and specify filesystem parameters.

## To-Dos
- [ ] Support for various audio file formats
- [ ] Audio resampling and stereo-to-mono
- [ ] Streaming audio input and output live
