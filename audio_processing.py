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
import pandas as pd
import os

run_waveform = False


def plot_waveform(
    waveform, sr, title="Waveform", ax=None, file_name="images/waveform.png"
):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    if ax is None:
        _, ax = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels))
    if num_channels == 1:
        ax = [ax]
    for i in range(num_channels):
        if title is not None:
            ax[i].set_title(f"{file_name} - Channel {i+1}")
        ax[i].set_ylabel("Amplitude")
        ax[i].plot(time_axis, waveform[i], linewidth=1)
        # ax[i].imshow(waveform, origin="lower", aspect="auto", interpolation="nearest")
        ax[i].grid(True)
        ax[i].set_xlim([0, time_axis[-1]])
        ax[i].set_title(title)
    # plt.savefig(file_name)
    # plt.close()
    print(f"{file_name} waveform done")
    return plt


def plot_spectrogram(
    specgram, title=None, ylabel="freq_bin", ax=None, file_name="images/spectrogram.png"
):
    num_channels = specgram.shape[0]
    if ax is None:
        _, ax = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels))
    if num_channels == 1:
        ax = [ax]
    for i in range(num_channels):
        if title is not None:
            ax[i].set_title(f"{file_name} - Channel {i+1}")
        ax[i].set_ylabel(ylabel)
        ax[i].imshow(
            librosa.power_to_db(specgram[i]),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
    plt.tight_layout()
    print(f"{file_name} spectrogram done")
    return plt


# Read metadata file
metadata_file = "metadata_osr.csv"
df = pd.read_csv(metadata_file)
df.head()

# Take relevant columns
df = df[["relative_path", "classID"]]
df.head()


current_directory = os.getcwd()
IMAGE_DIR = "images_osr"
APP_IMAGE_DIR = "display/images"
WAVEFORM_DIR = "waveform"
TARGET_VOICE_DIR = "target_voice"
OTHER_VOICE_DIR = "other_voice_osr"

# IMAGE Directories
if not os.path.exists(IMAGE_DIR) and not os.path.isdir(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)
if not os.path.exists(f"{IMAGE_DIR}/{TARGET_VOICE_DIR}") and not os.path.isdir(
    f"{IMAGE_DIR}/{TARGET_VOICE_DIR}"
):
    os.mkdir(f"{IMAGE_DIR}/{TARGET_VOICE_DIR}")
if not os.path.exists(f"{IMAGE_DIR}/{OTHER_VOICE_DIR}") and not os.path.isdir(
    f"{IMAGE_DIR}/{OTHER_VOICE_DIR}"
):
    os.mkdir(f"{IMAGE_DIR}/{OTHER_VOICE_DIR}")

# ## WAVEFORM Directories
# if not os.path.exists(f"{WAVEFORM_DIR}") and not os.path.isdir(f"{WAVEFORM_DIR}"):
#     os.mkdir(f"{WAVEFORM_DIR}")
# if not os.path.exists(f"{WAVEFORM_DIR}/{TARGET_VOICE_DIR}") and not os.path.isdir(
#     f"{WAVEFORM_DIR}/{TARGET_VOICE_DIR}"
# ):
#     os.mkdir(f"{WAVEFORM_DIR}/{TARGET_VOICE_DIR}")
# if not os.path.exists(f"{WAVEFORM_DIR}/{OTHER_VOICE_DIR}") and not os.path.isdir(
#     f"{WAVEFORM_DIR}/{OTHER_VOICE_DIR}"
# ):
#     os.mkdir(f"{WAVEFORM_DIR}/{OTHER_VOICE_DIR}")

## APP directories
if not os.path.exists(APP_IMAGE_DIR) and not os.path.isdir(APP_IMAGE_DIR):
    os.mkdir(APP_IMAGE_DIR)
if not os.path.exists(f"{APP_IMAGE_DIR}/{TARGET_VOICE_DIR}") and not os.path.isdir(
    f"{APP_IMAGE_DIR}/{TARGET_VOICE_DIR}"
):
    os.mkdir(f"{APP_IMAGE_DIR}/{TARGET_VOICE_DIR}")
if not os.path.exists(f"{APP_IMAGE_DIR}/{OTHER_VOICE_DIR}") and not os.path.isdir(
    f"{APP_IMAGE_DIR}/{OTHER_VOICE_DIR}"
):
    os.mkdir(f"{APP_IMAGE_DIR}/{OTHER_VOICE_DIR}")

for index in range(len(df)):
    try:
        AUDIO_FILE = f'{current_directory}/{df.loc[index, "relative_path"]}'
        aud = AudioUtil.open(AUDIO_FILE)
        sig, sr = aud
        sample_rate = 44100
        duration = 4000
        channel = 1
        shift_pct = 0
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, sample_rate)
        rechan = AudioUtil.rechannel(reaud, channel)
        # noise_aud = AudioUtil.add_noise(rechan, 0.05)
        dur_aud = AudioUtil.pad_trunc(rechan, duration)
        shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)

        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=80, n_fft=2048, hop_len=None)

        aug_sgram = AudioUtil.spectro_augment(
            sgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
        )
        file_name = f'{df.loc[index, "relative_path"]}.png'
        spectrogram_file_name = f"{current_directory}/{IMAGE_DIR}/{file_name}"
        waveform_file_name = f"{current_directory}/{WAVEFORM_DIR}/{file_name}"
        app_file_name = f"{APP_IMAGE_DIR}/{file_name}"
        spectrogram = plot_spectrogram(
            aug_sgram, title=file_name, file_name=spectrogram_file_name
        )
        spectrogram.savefig(spectrogram_file_name)
        print(app_file_name)
        spectrogram.savefig(app_file_name)
        # spectrogram.savefig(app_file_name)
        spectrogram.close()
        if run_waveform is True:
            waveform = plot_waveform(sig, sr=sr, file_name=waveform_file_name)
            waveform.savefig(waveform_file_name)
            waveform.close()
    except Exception as e:
        print(f"An error occured: {e}")


# fig1 = plt.figure(1)
# librosa.display.waveshow(aug_sgram, sr=sample_rate)

# fig2 = plt.figure(2)
# mfcc = librosa.feature.mfcc(y=aug_sgram, sr=sample_rate)
# # Center MFCC coefficient dimensions to the mean and unit variance
# mfcc = preprocessing.scale(mfcc, axis=1)
# librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')

# fig3 = plt.figure(3)
# sgram = librosa.stft(aug_sgram)
# sgram_mag, _ = librosa.magphase(sgram)
# mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
# mel_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
# librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.show()
