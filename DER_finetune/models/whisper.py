import os
import copy
import torch
import argparse
import numpy as np
import loralib as lora
import transformers.models.whisper.modeling_whisper as whisper

from functools import lru_cache

from torch import nn
from typing import Optional
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import WhisperModel, AutoFeatureExtractor

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 80):
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

class WhisperEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = whisper.WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.config = config
        
        # merge_weights=False is REQUIRED: with the default (True), loralib merges
        # LoRA into the weight on every .eval() (validation) and un-merges on .train();
        # across Lightning's train+validate loop this corrupts LoRA so lora_B stays 0
        # (encoder never actually trains). Verified: with default, lora_B=0 after a
        # validated run; with merge_weights=False, lora_B trains normally.
        self.fc1 = lora.Linear(self.embed_dim, config.encoder_ffn_dim, r=config.lora_rank, merge_weights=False)
        self.fc2 = lora.Linear(config.encoder_ffn_dim, self.embed_dim, r=config.lora_rank, merge_weights=False)
            

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
   
class BiLSTMClassifierHead(nn.Module):
    """2-layer Bidirectional LSTM classifier head (replaces the 1D-CNN model_seq).

    I/O convention matches the existing Conv1D `model_seq`: input `[B, D, T]`,
    output `[B, num_classes, T]`, so callers (`WhisperWrapper.forward` /
    `WhisperWrapperFast.forward`) need no changes.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Conv1d(hidden_dim * 2, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias = 1

    def forward(self, x):
        # x: [B, D, T]
        x = x.transpose(1, 2)        # [B, T, D]
        x, _ = self.lstm(x)          # [B, T, 2*hidden_dim]
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)        # [B, 2*hidden_dim, T]
        return self.classifier(x)    # [B, num_classes, T]


class WhisperWrapper(nn.Module):
    def __init__(
        self,
        pretrained_model="whisper_base",
        lora_rank=8,
        use_conv_output=False,
        hidden_dim=256,
        output_class_num=4,
        head_type="conv",
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        lstm_dropout=0.2,
    ):
        super(WhisperWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        if pretrained_model == "whisper_tiny":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-tiny",
                output_hidden_states=True
            )
        elif pretrained_model == "whisper_base":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-base",
                output_hidden_states=True
            )
        elif pretrained_model == "whisper_small":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-small",
                output_hidden_states=True
            )
        elif pretrained_model == "whisper_medium":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-medium",
                output_hidden_states=True
            )
        elif pretrained_model == "breeze":
            # MediaTek-Research/Breeze-ASR-25: whisper-large-v2 architecture
            # (d_model=1280, encoder_layers=32) fine-tuned for Mandarin/Taiwanese.
            # Checkpoint is a WhisperForConditionalGeneration; WhisperModel.from_pretrained
            # strips the "model." prefix automatically and ignores proj_out.* (LM head).
            self.backbone_model = WhisperModel.from_pretrained(
                "MediaTek-Research/Breeze-ASR-25",
                output_hidden_states=True
            )
            # forward() only ever uses backbone_model.encoder; drop the ~750M-param
            # decoder now so it never occupies GPU memory (~1.5GB in fp16).
            del self.backbone_model.decoder
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)
        self.embed_positions.requires_grad = False
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.lora_rank = lora_rank 
        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList(
            [WhisperEncoderLayer(self.model_config) for _ in range(self.model_config.encoder_layers)]
        )
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        for name, p in self.backbone_model.named_parameters():
            if name in msg.missing_keys: p.requires_grad = True
            else: p.requires_grad = False
        
        # 6. Downstream models
        if head_type == "bilstm":
            self.model_seq = BiLSTMClassifierHead(
                input_dim=self.model_config.hidden_size,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                num_classes=output_class_num,
            )
        else:
            self.model_seq = nn.Sequential(
                nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(hidden_dim, output_class_num, 1, padding=0),
            )
        self.use_conv_output = use_conv_output
        if use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1  # transformer layers + input embeddings
            self.weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))
        
            
    def forward(self, x, length=None):
        # 1. feature extraction and projections
        if length is not None:
            max_audio_len = length.max().detach().cpu()
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.cuda()
        else:
            new_x = list()
            for idx in range(len(x)):
                new_x.append(x[idx].detach().cpu().numpy())
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=len(x[0])
            )
            features = features.input_features.cuda()
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            max_len = length.max()
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:max_len])
        else:
            tmp_length = self.get_feat_extract_output_lengths(len(x[0]))
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:tmp_length])
            
        # 3. transformer encoding features
        features = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states
        # 4. stacked feature
        if self.use_conv_output:
            stacked_feature = torch.stack(features, dim=0)
        else:
            stacked_feature = torch.stack(features, dim=0)[1:]
        
        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
        else:
            stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)
        
        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        predicted = self.model_seq(features)
        return predicted

    def forward_eval(self, x):
        outputs = self.forward(x)
        outputs = torch.log_softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1).detach().cpu()
        pred = pred.squeeze(0)
        return pred
        
    # From huggingface
    def get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
    
    
if __name__ == '__main__':
    
    model = WhisperWrapper().cuda()
    
    data = torch.zeros([1, 16000]).cuda()
    output = model(data)
    print(output.shape)