"""音频预处理：加载 m4a/wav，提取 log-Mel 频谱"""
from pathlib import Path

import torch
import torchaudio

import config


class AudioProcessor:
    """音频加载与 Mel 频谱提取"""

    def __init__(self):
        n_stft = config.N_FFT // 2 + 1
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.MEL_FMIN,
            f_max=config.MEL_FMAX,
            center=True,
            power=1.0,
        )
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_stft,
            n_mels=config.N_MELS,
            sample_rate=config.SAMPLE_RATE,
            f_min=config.MEL_FMIN,
            f_max=config.MEL_FMAX,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_iter=config.GRIFFIN_LIM_ITERS,
        )

    def load_wav(self, path: Path) -> torch.Tensor:
        """加载音频并转为单声道、目标采样率"""
        waveform, sr = torchaudio.load(str(path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)
        return waveform.squeeze(0)

    def wav_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形 -> log Mel，形状 [n_mels, T]"""
        mel = self.mel_transform(waveform.unsqueeze(0)).squeeze(0)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def load_mel(self, path: Path) -> torch.Tensor:
        return self.wav_to_mel(self.load_wav(path))

    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel -> 线性频谱 -> Griffin-Lim -> 波形"""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel_linear = torch.exp(mel)
        mel_linear = torch.clamp(mel_linear, min=1e-5)

        spec = self.inverse_mel(mel_linear)
        spec = torch.clamp(spec, min=1e-5)

        waveform = self.griffin_lim(spec)
        return waveform.squeeze(0)
