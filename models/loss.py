"""Tacotron2 损失函数"""
import torch
import torch.nn as nn


class Tacotron2Loss(nn.Module):
    """Mel 重建损失 + 停止符 BCE"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_postnet: torch.Tensor,
        gate_pred: torch.Tensor,
        mel_target: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mel_mask = self._mel_mask(mel_target, mel_lengths)
        mel_pred_loss = self.mse(mel_pred.masked_select(mel_mask), mel_target.masked_select(mel_mask))
        mel_postnet_loss = self.mse(mel_postnet.masked_select(mel_mask), mel_target.masked_select(mel_mask))

        gate_target = self._gate_target(mel_lengths, gate_pred.size(1), gate_pred.device)
        gate_loss = self.bce(gate_pred, gate_target)

        total = mel_pred_loss + mel_postnet_loss + gate_loss
        return {
            "total": total,
            "mel": mel_pred_loss,
            "mel_postnet": mel_postnet_loss,
            "gate": gate_loss,
        }

    @staticmethod
    def _mel_mask(mel: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
        max_len = mel.size(2)
        mask = torch.arange(max_len, device=mel.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand_as(mel)

    @staticmethod
    def _gate_target(mel_lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        gate = torch.zeros(mel_lengths.size(0), max_len, device=device)
        for i, length in enumerate(mel_lengths):
            idx = min(length.item() - 1, max_len - 1)
            if idx >= 0:
                gate[i, idx] = 1.0
        return gate
