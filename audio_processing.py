import os
import torch
import librosa
import webrtcvad
import torchaudio
import numpy as np
import pandas as pd
import librosa.display
from AudioUtil import AudioUtil
import matplotlib.pyplot as plt
from IPython.display import Audio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.io.wavfile import write
from sklearn import preprocessing
from matplotlib import interactive
from matplotlib.patches import Rectangle
import pandas as pd
import os

run_waveform = False


def _plot_waveform(
    waveform, sr, title="Waveform", ax=None, file_name="images/waveform.png", 
    ylabel="Amplitude",
    xlabel="Time (s)",
    suptitle=None,
    left_margin=0.075,  # Default value for left margin
    right_margin=0.95,  # Default value for right margin
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
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].plot(time_axis, waveform[i], linewidth=1)
        # ax[i].imshow(waveform, origin="lower", aspect="auto", interpolation="nearest")
        #ax[i].grid(True)
        ax[i].set_xlim([0, time_axis[-1]])
        ax[i].set_title(title)
    
    # Set super title if provided
    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.savefig(file_name)
    plt.close()
    print(f"{file_name} waveform done")
    #return plt


def _plot_spectrogram(
    specgram, title=None, ylabel="freq_bin", ax=None, file_name="images/spectrogram.png", num_channels=1
):  

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
    plt.savefig(file_name)
    print(f"{file_name} spectrogram done")
    plt.close()
    #return plt



## THE CODE BELOW PROCESSES RAW AUDIO FILES AND STORES THEM AS .WAV FILE IN PROCESSED_AUDIO FOLDER
def processing(metadata_files: list, plot=True):
    
    current_directory = os.getcwd()
    PROCESS_DIR='processed_audio'

    vad = webrtcvad.Vad(3)
    for metadata_file in metadata_files:
        df = pd.read_csv(metadata_file)
        df.head()

        # Take relevant columns
        df = df[['relative_path', 'classID']]
        df.head()

        for index in range(len(df)):
            try:
                AUDIO_FILE = f'{current_directory}/{df.loc[index, "relative_path"]}'
                aud, sr = AudioUtil.open(AUDIO_FILE)
                
                frame_duration = 10 # 10 ms, 20 ms, 30 ms
                process_sample_rate = 8000 # 8000 Hz, 16000 Hz, 32000 Hz
                frame_size = int(frame_duration * process_sample_rate / 1000)

                audio_len = len(aud[0])
                num_of_frames = int(audio_len / frame_size)

                processed_audio_out_file =f'{current_directory}/{PROCESS_DIR}/{df.loc[index, "relative_path"]}'
                
                output_dir_path = os.path.dirname(os.path.realpath(processed_audio_out_file))
                # check if file exists
                if os.path.isfile(processed_audio_out_file):
                    continue
                # check if directory exists
                elif not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
                    os.makedirs(output_dir_path)

                output_wav = []
                for index in range(num_of_frames):
                    is_speech = vad.is_speech(aud[0][index*frame_size : (index*frame_size) + frame_size ].numpy().tobytes(), process_sample_rate)
                    if(is_speech):
                        voice_data = aud[0][index*frame_size : (index*frame_size) + frame_size ].numpy()
                        for item in voice_data:
                            output_wav.append(item)
                
                write(processed_audio_out_file, sr, np.array(output_wav))

            except Exception as e:
                print(f"An error occured: {e}")
    
    if plot:
        plot_spectrogram(metadata_files)


######################### THE CODE BELOW SAVE AUDIO AS IMAGE FOR FEATURE INSPECTION ###########################################
def plot_spectrogram(metadata_files: list):
    # Read metadata file
    current_directory = os.getcwd()
    IMAGE_DIR='images'    

    for metadata_file in metadata_files:
        df = pd.read_csv(metadata_file)
        df.head()

        # Take relevant columns
        df = df[['relative_path', 'classID']]
        df.head()
        
        for index in range(len(df)):
            try:
                AUDIO_FILE = f'{current_directory}/processed_audio/{df.loc[index, "relative_path"]}'
                aud = AudioUtil.open(AUDIO_FILE)
                
                sample_rate = 48000
                duration = 4000
                channel = 1
                shift_pct = 0.4
                # Some sounds have a higher sample rate, or fewer channels compared to the
                # majority. So make all sounds have the same number of channels and same 
                # sample rate. Unless the sample rate is the same, the pad_trunc will still
                # result in arrays of different lengths, even though the sound duration is
                # the same.
                reaud = AudioUtil.resample(aud, sample_rate)
                rechan = AudioUtil.rechannel(reaud, channel)
                dur_aud = AudioUtil.pad_trunc(rechan, duration)
                #shift_aud = AudioUtil.time_shift(rechan, shift_pct)
                sgram = AudioUtil.spectro_gram(dur_aud, n_mels=128, n_fft=512, hop_len=None)
                #aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1)
                spectrogram_file_name=f'{current_directory}/{IMAGE_DIR}/{df.loc[index, "relative_path"]}.png'
                output_dir_path = os.path.dirname(os.path.realpath(spectrogram_file_name))
                # check if file exists
                if os.path.isfile(spectrogram_file_name):
                    continue
                # check if directory exists
                elif not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
                    os.makedirs(output_dir_path)
                
                #plt.rcParams["figure.autolayout"] = True
                #fig, ax = plt.subplots(channel, 1, figsize=(4, 10))
                
                #y, sr = librosa.load(AUDIO_FILE)
                #print(shift_aud)
                #y_, sr = dur_aud
                #print(y_)
                #print(type(y_.cpu().detach().numpy()))

                """S = librosa.feature.melspectrogram(y=np.squeeze(y_.cpu().detach().numpy()), sr=sample_rate, n_mels=256, fmax=20000, hop_length=512)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis='off', y_axis='mel', sr=sample_rate, fmax=20000, ax=ax)
                plt.tight_layout()
                plt.savefig(spectrogram_file_name, bbox_inches='tight')
                print(f"{spectrogram_file_name} spectrogram done")
                plt.close()"""

                _plot_spectrogram(sgram, file_name=spectrogram_file_name, num_channels=channel)
                #plot_entropy(np.squeeze(aug_sgram), file_name=spectrogram_file_name)
            except Exception as e:
                print(f"An error occured: {e}")

def plot_waveform(metadata_files: list):
    # Read metadata file
    current_directory = os.getcwd()
    WAVEFORM_DIR='waveforms'    

    for metadata_file in metadata_files:
        df = pd.read_csv(metadata_file)
        df.head()

        # Take relevant columns
        df = df[['relative_path', 'classID']]
        df.head()
        
        for index in range(len(df)):
            try:
                AUDIO_FILE = f'{current_directory}/processed_audio/{df.loc[index, "relative_path"]}'
                aud = AudioUtil.open(AUDIO_FILE)
                
                sample_rate = 48000
                duration = 4000
                channel = 1
                shift_pct = 0.4
                # Some sounds have a higher sample rate, or fewer channels compared to the
                # majority. So make all sounds have the same number of channels and same 
                # sample rate. Unless the sample rate is the same, the pad_trunc will still
                # result in arrays of different lengths, even though the sound duration is
                # the same.
                reaud = AudioUtil.resample(aud, sample_rate)
                rechan = AudioUtil.rechannel(reaud, channel)
                dur_aud = AudioUtil.pad_trunc(rechan, duration)
                
                y_, sr = dur_aud
                
                #shift_aud = AudioUtil.time_shift(rechan, shift_pct)
                #sgram = AudioUtil.spectro_gram(shift_aud, n_mels=128, n_fft=512, hop_len=None)
                #aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1)
                waveform_file_name=f'{current_directory}/{WAVEFORM_DIR}/{df.loc[index, "relative_path"]}.png'
                output_dir_path = os.path.dirname(os.path.realpath(waveform_file_name))
                # check if file exists
                if os.path.isfile(waveform_file_name):
                    continue
                # check if directory exists
                elif not os.path.exists(output_dir_path) or not os.path.isdir(output_dir_path):
                    os.makedirs(output_dir_path)
                
                base_name = os.path.basename(waveform_file_name)
                _plot_waveform(waveform=y_.cpu().detach(), sr=sample_rate, file_name=waveform_file_name, title=f"ClassID: {df.loc[index, 'classID']}", suptitle=f"Waveform: {base_name}")
            except Exception as e:
                print(f"An error occured: {e}")