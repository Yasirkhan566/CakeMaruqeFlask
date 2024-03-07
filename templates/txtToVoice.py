# import os
# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats  # Import the scipy stats module

# # Define the directory where your audio files are stored
# data_dir = "C:/Users/yasir khan/Desktop/cakemarque flask mongodb/templates"

# # Initialize lists to store statistical features for each feature type
# feature_names = ['MFCCs', 'Chroma Features', 'Zero Crossing Rate', 'RMS Energy', 'Pitch/F0', 'Spectral Centroid']
# mean_values = []
# variance_values = []
# skewness_values = []
# kurtosis_values = []

# # Loop through all audio files in the directory
# for filename in os.listdir(data_dir):
#     if filename.endswith(".wav"):  # Assuming your audio files are in WAV format
#         audio_path = os.path.join(data_dir, filename)

#         # Load the audio file
#         y, sr = librosa.load(audio_path)

#         # Step 1: Compute the Short-Time Fourier Transform (STFT)
#         D = librosa.stft(y)

#         # Step 2: Apply a Mel filterbank to the power spectrum
#         # The number of Mel filterbanks (n_mels) is a parameter you can adjust
#         n_mels = 13
#         mel = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels)
#         mel_spec = np.dot(mel, np.abs(D)**2)

#         # Step 3: Take the logarithm of the filterbank energies
#         log_mel_spec = librosa.power_to_db(mel_spec)

#         # Step 4: Compute the discrete cosine transform (DCT) of the log filterbank energies
#         n_mfcc = 13  # Number of MFCC coefficients
#         mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mfcc)

#         # Step 5: Compute and visualize the spectrogram
#         plt.figure(figsize=(12, 13))
#         plt.subplot(7, 1, 1)
#         librosa.display.specshow(mfcc, x_axis='time')
#         plt.colorbar()
#         plt.title('MFCCs')

#         # Step 6: Compute and visualize Chroma features
#         chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#         plt.subplot(7, 1, 2)
#         librosa.display.specshow(chroma, x_axis='time', cmap='coolwarm')
#         plt.colorbar()
#         plt.title('Chroma Features')

#         # Step 7: Compute and visualize Zero Crossing Rate (ZCR)
#         zcr = librosa.feature.zero_crossing_rate(y=y)
#         plt.subplot(7, 1, 3)
#         plt.plot(zcr[0])
#         plt.ylabel('ZCR')
#         plt.title('Zero Crossing Rate')

#         # Step 8: Compute and visualize RMS Energy
#         rms_energy = librosa.feature.rms(y=y)
#         plt.subplot(7, 1, 4)
#         plt.plot(rms_energy[0])
#         plt.ylabel('RMS Energy')
#         plt.title('RMS Energy')

#         # Step 9: Compute and visualize Pitch and F0
#         pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#         pitches = np.mean(pitches, axis=0)
#         plt.subplot(7, 1, 5)
#         plt.plot(librosa.times_like(pitches), pitches)
#         plt.ylabel('Pitch/F0')
#         plt.title('Pitch/F0')

#         # Step 10: Compute and visualize Spectral Centroid
#         spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#         plt.subplot(7, 1, 6)
#         plt.semilogy(spectral_centroid[0], label='Spectral Centroid')
#         plt.ylabel('Hz')
#         plt.title('Spectral Centroid')

#         # Step 11: Compute and visualize Spectral Contrast
#         spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#         plt.subplot(8, 1, 7)
#         librosa.display.specshow(spectral_contrast, x_axis='time', cmap='viridis')
#         plt.colorbar()
#         plt.ylabel('Spectral Contrast')
#         plt.title('Spectral Contrast')

#         # Step 12: Compute and visualize Time-domain Features
#         time_mean = np.mean(y)
#         time_variance = np.var(y)
#         time_skewness = stats.skew(y)
#         time_kurtosis = stats.kurtosis(y)
#         plt.subplot(8, 1, 8)
#         plt.plot(y, label='Audio Signal')
#         plt.axhline(time_mean, color='r', linestyle='--', label='Mean')
#         plt.axhline(time_mean + np.sqrt(time_variance), color='g', linestyle='--', label='Mean + Std Dev')
#         plt.axhline(time_mean - np.sqrt(time_variance), color='g', linestyle='--', label='Mean - Std Dev')
#         plt.legend()
#         plt.ylabel('Amplitude')
#         plt.title('Time-domain Features')

#         # Append the statistical features for each feature type
#         mean_values.append([np.mean(mfcc), np.mean(chroma), np.mean(zcr), np.mean(rms_energy), np.mean(pitches), np.mean(spectral_centroid)])
#         variance_values.append([np.var(mfcc), np.var(chroma), np.var(zcr), np.var(rms_energy), np.var(pitches), np.var(spectral_centroid)])
#         skewness_values.append([stats.skew(mfcc, axis=None), stats.skew(chroma, axis=None), stats.skew(zcr, axis=None), stats.skew(rms_energy, axis=None), stats.skew(pitches), stats.skew(spectral_centroid, axis=None)])
#         kurtosis_values.append([stats.kurtosis(mfcc, axis=None), stats.kurtosis(chroma, axis=None), stats.kurtosis(zcr, axis=None), stats.kurtosis(rms_energy, axis=None), stats.kurtosis(pitches), stats.kurtosis(spectral_centroid, axis=None)])

#         plt.tight_layout()
#         plt.show()

# # Print the computed statistical features for each feature type
# for idx, feature_name in enumerate(feature_names):
#     print(f"Statistical Features for {feature_name}:")
#     print(f"Mean: {np.mean(mean_values, axis=0)[idx]}")
#     print(f"Variance: {np.mean(variance_values, axis=0)[idx]}")
#     print(f"Skewness: {np.mean(skewness_values, axis=0)[idx]}")
#     print(f"Kurtosis: {np.mean(kurtosis_values, axis=0)[idx]}\n")

# import tensorflow as tf
# from tensorflow.keras import layers, models

# def create_fnn_model(input_shape, num_classes):
#     model = models.Sequential()
#     model.add(layers.Input(shape=input_shape))
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(num_classes, activation='softmax'))
#     return model

# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(features, target_features, test_size=0.2, random_state=42)

# # Define the model
# input_shape = (input_feature_dimension,)  # Replace with the actual shape of your input features
# num_classes = target_feature_dimension  # Replace with the actual number of target features

# model = create_fnn_model(input_shape, num_classes)
# model.compile(optimizer='adam', loss='mean_squared_error')  # You may need to adjust the loss function

# # Train the model
# epochs = 100  # Adjust as needed
# batch_size = 32  # Adjust as needed

# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # Import the scipy stats module
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the directory where your audio files are stored
data_dir = "C:/Users/yasir khan/Desktop/cakemarque flask mongodb/templates"

# Initialize lists to store statistical features for each feature type
feature_names = ['MFCCs', 'Chroma Features', 'Zero Crossing Rate', 'RMS Energy', 'Pitch/F0', 'Spectral Centroid']
mean_values = []
variance_values = []
skewness_values = []
kurtosis_values = []

# Loop through all audio files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".wav"):  # Assuming your audio files are in WAV format
        audio_path = os.path.join(data_dir, filename)

        # Load the audio file
        y, sr = librosa.load(audio_path)

        # Step 1: Compute the Short-Time Fourier Transform (STFT)
        D = librosa.stft(y)

        # Step 2: Apply a Mel filterbank to the power spectrum
        # The number of Mel filterbanks (n_mels) is a parameter you can adjust
        n_mels = 13
        mel = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels)
        mel_spec = np.dot(mel, np.abs(D)**2)

        # Step 3: Take the logarithm of the filterbank energies
        log_mel_spec = librosa.power_to_db(mel_spec)

        # Step 4: Compute the discrete cosine transform (DCT) of the log filterbank energies
        n_mfcc = 13  # Number of MFCC coefficients
        mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mfcc)

        # Step 5: Compute and visualize the spectrogram
        plt.figure(figsize=(12, 13))
        plt.subplot(7, 1, 1)
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCCs')

        # Step 6: Compute and visualize Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        plt.subplot(7, 1, 2)
        librosa.display.specshow(chroma, x_axis='time', cmap='coolwarm')
        plt.colorbar()
        plt.title('Chroma Features')

        # Step 7: Compute and visualize Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        plt.subplot(7, 1, 3)
        plt.plot(zcr[0])
        plt.ylabel('ZCR')
        plt.title('Zero Crossing Rate')

        # Step 8: Compute and visualize RMS Energy
        rms_energy = librosa.feature.rms(y=y)
        plt.subplot(7, 1, 4)
        plt.plot(rms_energy[0])
        plt.ylabel('RMS Energy')
        plt.title('RMS Energy')

        # Step 9: Compute and visualize Pitch and F0
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = np.mean(pitches, axis=0)
        plt.subplot(7, 1, 5)
        plt.plot(librosa.times_like(pitches), pitches)
        plt.ylabel('Pitch/F0')
        plt.title('Pitch/F0')

        # Step 10: Compute and visualize Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        plt.subplot(7, 1, 6)
        plt.semilogy(spectral_centroid[0], label='Spectral Centroid')
        plt.ylabel('Hz')
        plt.title('Spectral Centroid')

        # Step 11: Compute and visualize Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        plt.subplot(8, 1, 7)
        librosa.display.specshow(spectral_contrast, x_axis='time', cmap='viridis')
        plt.colorbar()
        plt.ylabel('Spectral Contrast')
        plt.title('Spectral Contrast')

        # Step 12: Compute and visualize Time-domain Features
        time_mean = np.mean(y)
        time_variance = np.var(y)
        time_skewness = stats.skew(y)
        time_kurtosis = stats.kurtosis(y)
        plt.subplot(8, 1, 8)
        plt.plot(y, label='Audio Signal')
        plt.axhline(time_mean, color='r', linestyle='--', label='Mean')
        plt.axhline(time_mean + np.sqrt(time_variance), color='g', linestyle='--', label='Mean + Std Dev')
        plt.axhline(time_mean - np.sqrt(time_variance), color='g', linestyle='--', label='Mean - Std Dev')
        plt.legend()
        plt.ylabel('Amplitude')
        plt.title('Time-domain Features')

        # Append the statistical features for each feature type
        mean_values.append([np.mean(mfcc), np.mean(chroma), np.mean(zcr), np.mean(rms_energy), np.mean(pitches), np.mean(spectral_centroid)])
        variance_values.append([np.var(mfcc), np.var(chroma), np.var(zcr), np.var(rms_energy), np.var(pitches), np.var(spectral_centroid)])
        skewness_values.append([stats.skew(mfcc, axis=None), stats.skew(chroma, axis=None), stats.skew(zcr, axis=None), stats.skew(rms_energy, axis=None), stats.skew(pitches), stats.skew(spectral_centroid, axis=None)])
        kurtosis_values.append([stats.kurtosis(mfcc, axis=None), stats.kurtosis(chroma, axis=None), stats.kurtosis(zcr, axis=None), stats.kurtosis(rms_energy, axis=None), stats.kurtosis(pitches), stats.kurtosis(spectral_centroid, axis=None)])

# Concatenate all features into a single array
all_features = np.concatenate((mean_values, variance_values, skewness_values, kurtosis_values), axis=1)
print("all_features Shape: ",all_features.shape)
from sklearn.preprocessing import StandardScaler
# Normalize features using StandardScaler
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)
print("all_features Shape after normalization: ", all_features.shape)
print(all_features)

def create_fnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='linear'))
    return model

# Calculate the total number of features
num_classes = len(mean_values[0]) + len(variance_values[0]) + len(skewness_values[0]) + len(kurtosis_values[0])
print("total number of features: ", num_classes)

# Create and compile the model
# Create an autoencoder model for style transfer
input_shape = (all_features.shape[1],)
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Dense(128, activation='relu'),  # Reduce the dimension here
    layers.Dense(input_shape[0], activation='linear')
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
epochs = 100
batch_size = 32

history = model.fit(
    all_features,  # Use the entire dataset for unsupervised training
    all_features,  # Target is the same as input for unsupervised training
    epochs=epochs,
    batch_size=batch_size
)
from gtts import gTTS
import io

def text_to_voice(input_text, model, scaler):
    # Preprocess the input text and extract relevant features
    # You need to implement this part based on your feature extraction method

    # Encode the extracted features using the trained autoencoder model
    encoded_features = model.predict(preprocessed_features)

    # Generate audio from the encoded features
    generated_audio = model.layers[-1](encoded_features)  # Assuming the last layer is for audio generation

    # Save the generated audio to a temporary file
    temp_audio_path = 'temp.mp3'
    tts = gTTS(text='', lang='en')  # Empty text, as we'll provide the audio data directly
    tts.save(temp_audio_path)

    # Read the contents of the temporary audio file into a BytesIO object
    with open(temp_audio_path, 'rb') as f:
        audio_data = io.BytesIO(f.read())

    # Clean up the temporary audio file
    os.remove(temp_audio_path)

    return audio_data

# Example usage
input_text = "Hello dear students, how are you? What's going on?"
generated_audio = text_to_voice(input_text, model, scaler)

# Save the generated audio as needed
output_audio_path = "C:/Users/yasir khan/Desktop/cakemarque flask mongodb/templates/style_transferred_audio.wav"
sf.write(output_audio_path, generated_audio, sr)  # Save the audio

# def extract_audio_features(y, sr):
#     # Calculate MFCCs (Mel-frequency cepstral coefficients)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # You can adjust n_mfcc as needed

#     # Calculate Chroma Features
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)

#     # Calculate Zero Crossing Rate
#     zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

#     # Calculate RMS Energy
#     rms_energy = librosa.feature.rms(y=y)

#     # Calculate Pitch/F0
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#     pitch = []
#     for i in range(pitches.shape[1]):
#         index = magnitudes[:, i].argmax()
#         pitch.append(pitches[index, i])
#     pitch = np.array(pitch)

#     # Calculate Spectral Centroid
#     spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

#     # Concatenate all the computed features into a single array
#     features = np.vstack((mfccs, chroma, zero_crossing_rate, rms_energy, pitch, spectral_centroids))

#     # Transpose the feature matrix so that rows correspond to features and columns to frames
#     features = features.T
#     print('Test Audio Shape: ', features.shape)
#     return features
# # User-uploaded audio style transfer function

# import numpy as np

# def generate_audio(features, sr):
#     # Assuming 'features' is a 2D array of shape (num_frames, num_features)
#     # You need to convert it back to the time domain (audio waveform)
    
#     # Inverse transform the features back to the time domain
#     reconstructed_audio = np.zeros(features.shape[0] * 512)  # Assuming frame size of 512 (adjust as needed)
    
#     for i, frame_features in enumerate(features):
#         # Assuming 'frame_features' represents the spectral content of a frame
#         # You may need to adjust this part based on your feature representation
#         # Perform inverse FFT to get the time-domain signal
#         frame_audio = np.fft.irfft(frame_features)
        
#         # Overlap and add the frame_audio to the reconstructed_audio
#         frame_len = len(frame_audio)
#         reconstructed_audio[i * frame_len : (i + 1) * frame_len] += frame_audio
    
#     # Normalize the reconstructed_audio if needed
#     max_amplitude = np.max(np.abs(reconstructed_audio))
#     if max_amplitude > 1.0:
#         reconstructed_audio = reconstructed_audio / max_amplitude
    
#     # Convert to 16-bit PCM format if needed (assuming 16-bit audio)
#     reconstructed_audio = (reconstructed_audio * 32767.0).astype(np.int16)
    
#     return reconstructed_audio

# def perform_style_transfer(user_audio_path, style_transfer_model, scaler):
#     # Load the user-uploaded audio
#     y, sr = librosa.load(user_audio_path)

#     user_features = extract_audio_features(y, sr)
#     mean_values = []
#     variance_values = []
#     skewness_values = []
#     kurtosis_values = []

#     mean_values.append([np.mean(mfcc), np.mean(chroma), np.mean(zcr), np.mean(rms_energy), np.mean(pitches), np.mean(spectral_centroid)])
#     variance_values.append([np.var(mfcc), np.var(chroma), np.var(zcr), np.var(rms_energy), np.var(pitches), np.var(spectral_centroid)])
#     skewness_values.append([stats.skew(mfcc, axis=None), stats.skew(chroma, axis=None), stats.skew(zcr, axis=None), stats.skew(rms_energy, axis=None), stats.skew(pitches), stats.skew(spectral_centroid, axis=None)])
#     kurtosis_values.append([stats.kurtosis(mfcc, axis=None), stats.kurtosis(chroma, axis=None), stats.kurtosis(zcr, axis=None), stats.kurtosis(rms_energy, axis=None), stats.kurtosis(pitches), stats.kurtosis(spectral_centroid, axis=None)])


#     # Concatenate the computed features into a single array
#     user_audio_features = np.concatenate((mean_values, variance_values, skewness_values, kurtosis_values), axis=1)

#     # Normalize user audio features using the same scaler used during training
#     user_audio_features = scaler.transform(user_audio_features)

#     # Perform style transfer using the pre-trained model
#     transferred_features = style_transfer_model.predict(user_audio_features)

#     # Reverse the normalization applied during training
#     transferred_features = scaler.inverse_transform(transferred_features)
#     # Create an audio waveform from the transferred features
#     transferred_audio = generate_audio(transferred_features, sr)

#     # Return the audio waveform and sample rate
#     return transferred_audio, sr

# user_audio_path = 'C:/Users/yasir khan/Desktop/cakemarque flask mongodb/templates/user_uploaded_audio.wav'  # Adjust to the uploaded file
# style_transferred_audio = perform_style_transfer(user_audio_path, model, scaler)

# transferred_audio, sr = style_transferred_audio

# output_path = 'C:/Users/yasir khan/Desktop/cakemarque flask mongodb/templates/style_transferred_audio.wav'  # Replace 'path_to_save' with your desired output directory

# import soundfile as sf

# sf.write(output_path, transferred_audio, sr)






    

