import torch
import librosa
import torchaudio
import numpy as np
import librosa.display
from AudioUtil import AudioUtil
import matplotlib.pyplot as plt
from IPython.display import Audio
import torchaudio.functional as F
import torchaudio.transforms as T
from sklearn import preprocessing
from matplotlib import interactive
from matplotlib.patches import Rectangle

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")



# Load the audio file
AUDIO_FILE = '/home/saie/Desktop/MachineLearning/dataset_1/OSR_us_000_0037_8k.wav'
aud = AudioUtil.open(AUDIO_FILE)
sample_rate = 44100
duration = 4000
channel = 2
shift_pct = 0.4
# Some sounds have a higher sample rate, or fewer channels compared to the
# majority. So make all sounds have the same number of channels and same 
# sample rate. Unless the sample rate is the same, the pad_trunc will still
# result in arrays of different lengths, even though the sound duration is
# the same.
reaud = AudioUtil.resample(aud, sample_rate)
rechan = AudioUtil.rechannel(reaud, channel)

dur_aud = AudioUtil.pad_trunc(rechan, duration)
shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

plot_spectrogram(aug_sgram)


"""fig1 = plt.figure(1)
librosa.display.waveshow(aug_sgram, sr=sample_rate)

fig2 = plt.figure(2)
mfcc = librosa.feature.mfcc(y=aug_sgram, sr=sample_rate)
# Center MFCC coefficient dimensions to the mean and unit variance
mfcc = preprocessing.scale(mfcc, axis=1)
librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')

fig3 = plt.figure(3)
sgram = librosa.stft(aug_sgram)
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
mel_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()"""