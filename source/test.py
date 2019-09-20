import scipy.io .wavfile

import numpy
_, audio = scipy.io.wavfile.read('/home/ralph/audio/bdl-clb-cy.wav')

print(audio.astype(numpy.float64))