import os
import torch
import soundfile as sf
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F


class PairedAudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, target_sample_rate=16000, duration_sec=2.0):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.target_sr = target_sample_rate
        self.num_samples = int(target_sample_rate * duration_sec)

        # 1. Grab all .wav files from both directories
        noisy_files = set(f for f in os.listdir(noisy_dir) if f.endswith('.wav'))
        clean_files = set(f for f in os.listdir(clean_dir) if f.endswith('.wav'))


        common_files = noisy_files.intersection(clean_files)

        # fopr deterministic batching
        self.filenames = sorted(list(common_files))

        print(f"-> Found {len(noisy_files)} noisy files.")
        print(f"-> Found {len(clean_files)} clean files.")
        print(f"-> Training on {len(self.filenames)} perfectly paired files.")

        if len(self.filenames) == 0:
            raise ValueError(
                "CRASH: 0 matching files found! This means the filenames in your "
                "clean folder and noisy folder are completely different "
                "(e.g., 'audio1.wav' vs 'clean_audio1.wav')."
            )
        # just for testing smaller sizes
        # self.filenames = self.filenames[:100]

    def __len__(self):
        return len(self.filenames)

    def _load_and_format(self, path):
        audio_array, sr = sf.read(path, dtype='float32')
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]

        if len(audio_array) > self.num_samples:
            audio_array = audio_array[:self.num_samples]
        else:
            pad_amount = self.num_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, pad_amount), mode='constant')

        return torch.from_numpy(audio_array).unsqueeze(0)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        noisy_waveform = self._load_and_format(noisy_path)
        clean_waveform = self._load_and_format(clean_path)

        return noisy_waveform, clean_waveform



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.dec1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        target_size = (x.size(2), x.size(3))

        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))

        d1 = torch.relu(self.dec1(e2))
        d2 = torch.sigmoid(self.dec2(d1))

        d2 = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)

        return d2

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    noisy_path = r"" #modify this to your desktop path, this will be the folder containing noisy audio
    clean_path = r"" #this one is for clean audio

    print("Loading dataset...")
    full_dataset = PairedAudioDataset(noisy_path, clean_path)

    # 80 20 split, try more later
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    train_loss_history = []
    val_loss_history = []

    n_fft = 512
    hop_length = 128
    window = torch.hann_window(n_fft, device=device)

    for epoch in range(epochs):

        model.train()
        running_train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for batch_idx, (noisy_wave, clean_wave) in enumerate(train_loop):
            noisy_wave, clean_wave = noisy_wave.to(device), clean_wave.to(device)

            noisy_wave_sq = noisy_wave.squeeze(1)
            clean_wave_sq = clean_wave.squeeze(1)

            noisy_stft = torch.stft(noisy_wave_sq, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            clean_stft = torch.stft(clean_wave_sq, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

            noisy_mag = torch.abs(noisy_stft)
            clean_mag = torch.abs(clean_stft)

            target_mask = clean_mag / (noisy_mag + 1e-8)
            target_mask = torch.clamp(target_mask, 0.0, 1.0)

            noisy_mag = noisy_mag.unsqueeze(1)
            target_mask = target_mask.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(noisy_mag)
            loss = criterion(outputs, target_mask)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)


        # validate from here
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for noisy_wave, clean_wave in val_loader:
                noisy_wave, clean_wave = noisy_wave.to(device), clean_wave.to(device)

                noisy_wave_sq = noisy_wave.squeeze(1)
                clean_wave_sq = clean_wave.squeeze(1)

                noisy_stft = torch.stft(noisy_wave_sq, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
                clean_stft = torch.stft(clean_wave_sq, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

                noisy_mag = torch.abs(noisy_stft)
                clean_mag = torch.abs(clean_stft)

                target_mask = clean_mag / (noisy_mag + 1e-8)
                target_mask = torch.clamp(target_mask, 0.0, 1.0)

                noisy_mag = noisy_mag.unsqueeze(1)
                target_mask = target_mask.unsqueeze(1)

                outputs = model(noisy_mag)
                loss = criterion(outputs, target_mask)
                running_val_loss += loss.item()

        # Record average val loss for this epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Print the summary at the end of the epoch
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


    plot_learning_curves(train_loss_history, val_loss_history)

    print("Saving models...")
    torch.save(model.state_dict(), "unet_weights.pth")

    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 1, 257, 251)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save("noise_canceller.pt")

    # print("Export complete! Saved as 'noise_canceller.pt'")

if __name__ == "__main__":
    main()

