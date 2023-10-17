import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import interactive
from sklearn import preprocessing

# Load the audio file
AUDIO_FILE = '/home/saie/Desktop/MachineLearning/dataset_1/OSR_us_000_0037_8k.wav'
samples, sample_rate = librosa.load(AUDIO_FILE, sr=48000)


fig1 = plt.figure(1)
librosa.display.waveshow(samples, sr=sample_rate)

fig2 = plt.figure(2)
mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
# Center MFCC coefficient dimensions to the mean and unit variance
mfcc = preprocessing.scale(mfcc, axis=1)
librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')

fig3 = plt.figure(3)
sgram = librosa.stft(samples)
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
mel_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()