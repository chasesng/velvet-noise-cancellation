import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from main import UNet

def test_audio(model_path, test_wav_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    audio_array, sr = sf.read(test_wav_path, dtype='float32')
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0] # take left channel if stereo
    noisy_waveform = torch.from_numpy(audio_array).unsqueeze(0)


    # compute stft
    n_fft = 512
    hop_length = 128
    window = torch.hann_window(n_fft)

    noisy_stft = torch.stft(noisy_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    noisy_mag = torch.abs(noisy_stft).unsqueeze(0) # Shape: [1, 1, Freq, Time]
    noisy_phase = torch.angle(noisy_stft)

    with torch.no_grad():
        noisy_mag = noisy_mag.to(device)
        predicted_mask = model(noisy_mag)
        predicted_mask = predicted_mask.cpu().squeeze(0) # Back to CPU, shape: [1, Freq, Time]

    noisy_mag_cpu = noisy_mag.cpu().squeeze(0)
    clean_mag = noisy_mag_cpu * predicted_mask

    clean_complex = torch.polar(clean_mag.squeeze(0), noisy_phase.squeeze(0))

    clean_output_waveform = torch.istft(clean_complex.unsqueeze(0), n_fft=n_fft, hop_length=hop_length, window=window)

    output_array = clean_output_waveform.squeeze().numpy()
    sf.write("noisy_input.wav", noisy_waveform.squeeze().numpy(), sr)
    sf.write("clean_output.wav", output_array, sr)

    plot_spectrograms(noisy_mag_cpu.squeeze().numpy(), clean_mag.squeeze().numpy(), predicted_mask.squeeze().numpy())

def plot_spectrograms(noisy, clean, mask):
    plt.figure(figsize=(15, 5))

    # db convert
    noisy_db = 20 * np.log10(noisy + 1e-8)
    clean_db = 20 * np.log10(clean + 1e-8)

    plt.subplot(1, 3, 1)
    plt.title("Noisy Input")
    plt.imshow(noisy_db, origin='lower', aspect='auto', cmap='magma')

    plt.subplot(1, 3, 2)
    plt.title("Model's Predicted Mask")
    plt.imshow(mask, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Cleaned Output")
    plt.imshow(clean_db, origin='lower', aspect='auto', cmap='magma')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test_file = r"" //testing individual audio for cleaning
    # test_audio("unet_weights.pth", test_file)