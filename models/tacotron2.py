"""Tacotron2: 文本编码器 + 注意力解码器 + PostNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class LocationSensitiveAttention(nn.Module):
    """Location-Sensitive Attention"""

    def __init__(
        self,
        query_dim: int,
        encoder_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
    ):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_conv = nn.Conv1d(
            2,
            attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
            padding=(attention_location_kernel_size - 1) // 2,
            bias=False,
        )
        self.location_layer = nn.Linear(attention_location_n_filters, attention_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
    ):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_loc = self.location_layer(self.location_conv(attention_weights_cat).transpose(1, 2))
        energies = self.v(torch.tanh(processed_query + processed_memory + processed_loc))
        attention_weights = F.softmax(energies.squeeze(-1), dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return context, attention_weights


class Prenet(nn.Module):
    """Pre-net with dropout fixed at 0.5 during training"""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """5-layer Conv PostNet"""

    def __init__(self, n_mels: int, postnet_channels: int, postnet_kernel: int, postnet_n_conv: int):
        super().__init__()
        convs = []
        in_ch = n_mels
        for i in range(postnet_n_conv - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, postnet_channels, postnet_kernel, padding=(postnet_kernel - 1) // 2),
                    nn.BatchNorm1d(postnet_channels),
                    nn.Tanh(),
                )
            )
            in_ch = postnet_channels
        convs.append(
            nn.Conv1d(in_ch, n_mels, postnet_kernel, padding=(postnet_kernel - 1) // 2)
        )
        self.convs = nn.ModuleList(convs)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel
        for conv in self.convs:
            x = conv(x)
        return x


class Encoder(nn.Module):
    """字符嵌入 + Conv + BiLSTM"""

    def __init__(
        self,
        n_symbols: int,
        embedding_dim: int,
        conv_channels: int,
        conv_kernel: int,
        n_conv: int,
        lstm_hidden: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, embedding_dim, padding_idx=0)
        convs = []
        in_ch = embedding_dim
        for _ in range(n_conv):
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, conv_channels, conv_kernel, padding=(conv_kernel - 1) // 2),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
            )
            in_ch = conv_channels
        self.convs = nn.ModuleList(convs)
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, text, text_lengths):
        x = self.embedding(text).transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, text_lengths


class Decoder(nn.Module):
    """注意力解码器，teacher forcing 训练"""

    def __init__(
        self,
        n_mels: int,
        encoder_dim: int,
        attention_rnn_dim: int,
        decoder_rnn_dim: int,
        prenet_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
        max_decoder_steps: int,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.max_decoder_steps = max_decoder_steps
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim

        self.prenet = Prenet(n_mels, prenet_dim)
        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_dim, attention_rnn_dim)
        self.attention = LocationSensitiveAttention(
            attention_rnn_dim,
            encoder_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_dim, decoder_rnn_dim)
        self.linear = nn.Linear(decoder_rnn_dim + encoder_dim, n_mels)
        self.gate_layer = nn.Linear(decoder_rnn_dim + encoder_dim, 1)

        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)

    def forward(
        self,
        memory: torch.Tensor,
        mel_target: torch.Tensor,
        memory_lengths: torch.Tensor,
    ):
        batch_size = memory.size(0)
        max_time = mel_target.size(2)

        attention_hidden = torch.zeros(batch_size, self.attention_rnn_dim, device=memory.device)
        attention_cell = torch.zeros(batch_size, self.attention_rnn_dim, device=memory.device)
        decoder_hidden = torch.zeros(batch_size, self.decoder_rnn_dim, device=memory.device)
        decoder_cell = torch.zeros(batch_size, self.decoder_rnn_dim, device=memory.device)

        attention_weights = torch.zeros(batch_size, memory.size(1), device=memory.device)
        attention_weights_cum = torch.zeros(batch_size, memory.size(1), device=memory.device)
        processed_memory = self.memory_layer(memory)

        mel_outputs = []
        gate_outputs = []

        # 首帧用全零输入
        decoder_input = torch.zeros(batch_size, self.n_mels, device=memory.device)

        for t in range(max_time):
            prenet_out = self.prenet(decoder_input)
            attention_rnn_input = torch.cat([prenet_out, self._last_context(attention_weights, memory)], dim=-1)
            attention_hidden, attention_cell = self.attention_rnn(
                attention_rnn_input, (attention_hidden, attention_cell)
            )

            attention_weights_cat = torch.cat(
                [attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)], dim=1
            )
            context, attention_weights = self.attention(
                attention_hidden, memory, processed_memory, attention_weights_cat
            )
            attention_weights_cum = attention_weights_cum + attention_weights

            decoder_rnn_input = torch.cat([attention_hidden, context], dim=-1)
            decoder_hidden, decoder_cell = self.decoder_rnn(
                decoder_rnn_input, (decoder_hidden, decoder_cell)
            )

            decoder_output = torch.cat([decoder_hidden, context], dim=-1)
            mel_output = self.linear(decoder_output)
            gate_output = self.gate_layer(decoder_output)

            mel_outputs.append(mel_output.unsqueeze(-1))
            gate_outputs.append(gate_output.unsqueeze(-1))

            # teacher forcing
            decoder_input = mel_target[:, :, t]

        mel_outputs = torch.cat(mel_outputs, dim=-1)
        gate_outputs = torch.cat(gate_outputs, dim=-1).squeeze(1)
        return mel_outputs, gate_outputs

    @torch.no_grad()
    def infer(
        self,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
    ):
        """自回归推理"""
        batch_size = memory.size(0)
        device = memory.device

        attention_hidden = torch.zeros(batch_size, self.attention_rnn_dim, device=device)
        attention_cell = torch.zeros(batch_size, self.attention_rnn_dim, device=device)
        decoder_hidden = torch.zeros(batch_size, self.decoder_rnn_dim, device=device)
        decoder_cell = torch.zeros(batch_size, self.decoder_rnn_dim, device=device)

        attention_weights = torch.zeros(batch_size, memory.size(1), device=device)
        attention_weights_cum = torch.zeros(batch_size, memory.size(1), device=device)
        processed_memory = self.memory_layer(memory)

        mel_outputs = []
        gate_outputs = []
        decoder_input = torch.zeros(batch_size, self.n_mels, device=device)

        for _ in range(self.max_decoder_steps):
            prenet_out = self.prenet(decoder_input)
            attention_rnn_input = torch.cat([prenet_out, self._last_context(attention_weights, memory)], dim=-1)
            attention_hidden, attention_cell = self.attention_rnn(
                attention_rnn_input, (attention_hidden, attention_cell)
            )

            attention_weights_cat = torch.cat(
                [attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)], dim=1
            )
            context, attention_weights = self.attention(
                attention_hidden, memory, processed_memory, attention_weights_cat
            )
            attention_weights_cum = attention_weights_cum + attention_weights

            decoder_rnn_input = torch.cat([attention_hidden, context], dim=-1)
            decoder_hidden, decoder_cell = self.decoder_rnn(
                decoder_rnn_input, (decoder_hidden, decoder_cell)
            )

            decoder_output = torch.cat([decoder_hidden, context], dim=-1)
            mel_output = self.linear(decoder_output)
            gate_output = self.gate_layer(decoder_output)

            mel_outputs.append(mel_output.unsqueeze(-1))
            gate_outputs.append(gate_output.unsqueeze(-1))
            decoder_input = mel_output

            if torch.sigmoid(gate_output).mean().item() > config.GATE_THRESHOLD:
                break

        mel_outputs = torch.cat(mel_outputs, dim=-1)
        gate_outputs = torch.cat(gate_outputs, dim=-1).squeeze(1)
        return mel_outputs, gate_outputs

    @staticmethod
    def _last_context(attention_weights: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)


class Tacotron2(nn.Module):
    """完整 Tacotron2 模型"""

    def __init__(self, n_symbols: int):
        super().__init__()
        encoder_dim = config.ENCODER_LSTM_HIDDEN * 2
        self.encoder = Encoder(
            n_symbols=n_symbols,
            embedding_dim=config.SYMBOL_EMBEDDING_DIM,
            conv_channels=config.ENCODER_CONV_CHANNELS,
            conv_kernel=config.ENCODER_KERNEL_SIZE,
            n_conv=config.ENCODER_N_CONV,
            lstm_hidden=config.ENCODER_LSTM_HIDDEN,
        )
        self.decoder = Decoder(
            n_mels=config.N_MELS,
            encoder_dim=encoder_dim,
            attention_rnn_dim=config.ATTENTION_RNN_DIM,
            decoder_rnn_dim=config.DECODER_RNN_DIM,
            prenet_dim=config.PRENET_DIM,
            attention_dim=config.ATTENTION_DIM,
            attention_location_n_filters=config.ATTENTION_LOCATION_N_FILTERS,
            attention_location_kernel_size=config.ATTENTION_LOCATION_KERNEL_SIZE,
            max_decoder_steps=config.MAX_DECODER_STEPS,
        )
        self.postnet = Postnet(
            n_mels=config.N_MELS,
            postnet_channels=config.POSTNET_CHANNELS,
            postnet_kernel=config.POSTNET_KERNEL_SIZE,
            postnet_n_conv=config.POSTNET_N_CONV,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_target: torch.Tensor,
    ):
        memory, memory_lengths = self.encoder(text, text_lengths)
        mel_pred, gate_pred = self.decoder(memory, mel_target, memory_lengths)
        mel_postnet = mel_pred + self.postnet(mel_pred)
        return mel_pred, mel_postnet, gate_pred

    @torch.no_grad()
    def infer(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        memory, memory_lengths = self.encoder(text, text_lengths)
        mel_pred, _ = self.decoder.infer(memory, memory_lengths)
        mel_postnet = mel_pred + self.postnet(mel_pred)
        return mel_postnet
