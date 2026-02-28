# Velvet: Deep Learning Noise Cancellation Template
A robust, GPU-accelerated pipeline for training an audio noise suppression model using PyTorch.
WhiteVelvet uses a U-Net architecture to perform spectral gating, learning an Ideal Ratio Mask (IRM) to separate clean human speech from background environmental noise, the pipeline exports an optimized TorchScript model ready for lower latency LibTorch integration.

Instead of generating raw audio waveforms from scratch, Velvet operates entirely in the freq domain:

- STFT Conversion: The 1D noisy audio is converted into a 2D complex spectrogram.
- U-Net Processing: The network analyzes the magnitude of the spectrogram and outputs a gain mask (values between 0 and 1.0).   
- Spectral Gating: The mask is multiplied against the noisy magnitude to suppress non-vocal frequencies.  
- ISTFT Reconstruction: The cleaned magnitude is recombined with the original noisy phase to reconstruct a clean 1D audio waveform.


### Key Features
- Ease of reconstruction: You can modify the code to build your own model with updated parameters. This project should only be a base.
- Faster Data Loading: Audio remains as raw 1D waveforms on the CPU and is pushed to the GPU before calculating STFT, completely eliminating CPU bottlenecks during training.
- CPP Optimization Ready: Automatically traces and exports the trained U-Net to a .pt TorchScript file for zero-dependency deployment in C++ environments.
- Visual Validation: Includes inference scripts to visually plot the noisy input, the model's predicted mask, and the cleaned output spectrogram.


### Installation

#### Prerequisites
Ensure you have Python 3.8+ and a CUDA-capable NVIDIA GPU installed.

```bash
pip install torch torchaudio soundfile numpy tqdm matplotlib
```


### Usage

Training the Model
Open main.py and update the noisy_path and clean_path variables to point to your dataset directories. Then, launch the training loop:

```bash
python main.py 
```

### Acknowledgements

Special thanks to the creator of the dataset used to train and validate this model:
Libri Speech Noise Dataset curated by Kaggle user earth16. This dataset provided the essential paired clean and noisy audio samples required to effectively train the U-Net for noise suppression.
You can download the model here: https://www.kaggle.com/datasets/earth16/libri-speech-noise-dataset
