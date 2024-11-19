from typing import Optional, Any
import math
import copy
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['max_seq_len']
    data_window_len = config['data_window_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):

        if config['model'] == 'transformer':
            return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                        config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                        pos_encoding=config['pos_encoding'], activation=config['activation'],
                                        norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'BBTE':
            return BBTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                             config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                             pos_encoding=config['pos_encoding'], activation=config['activation'],
                                             norm=config['normalization_layer'], freeze=config['freeze'])


    if (task == "classification") or (task == "regression"):
        if task == "classification":
            num_labels = len(data.class_names)
        elif config['data_class'] == 'socdataset' or config['data_class'] == 'sotdataset':
            num_labels = 1
        elif config['data_class'] == 'sohdataset':
            num_labels = 1 # dimensionality of labels


        if config['model'] == 'transformer':
            return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                        num_classes=num_labels,
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'BBTE' and config['data_class'] == 'socdataset':
            return BBTransformerEncoderforSOC(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                       data_window_len=data_window_len,
                                                        num_classes=num_labels,
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'BBTE' and config['data_class'] == 'sotdataset':
            return BBTransformerEncoderforSOT(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                       data_window_len=data_window_len,
                                                        num_classes=num_labels,
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'BBTE' and config['data_class'] == 'sohdataset':
            return BBTransformerEncoderforSOH(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                       data_window_len=data_window_len,
                                                        num_classes=num_labels,
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'])

    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class SENet(nn.Module):
    def __init__(self, channels, reduction=1):
        super(SENet, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [seq_len, batch, channels]
        # Apply global average pooling (mean over the sequence length dimension)
        x_mean = x.mean(dim=0)  # Shape: [batch, channels]

        # Squeeze and Excitation
        se = self.fc1(x_mean)  # Shape: [batch, channels // reduction]
        se = nn.ReLU()(se)
        se = self.fc2(se)  # Shape: [batch, channels]
        se = self.sigmoid(se)  # Shape: [batch, channels]

        # Reshape and scale the input features
        se = se.unsqueeze(0).expand_as(x)  # Shape: [seq_len, batch, channels]
        return x * se


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)


        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class CurrentBiasTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, reduction=16, activation="relu"):
        super(CurrentBiasTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.senet1 = SENet(d_model,reduction)
        self.senet2 = SENet(d_model,reduction)
    def forward(self, src, src_mask=None, src_key_padding_mask=None, current_bias=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        if current_bias is not None:
            src2 += current_bias
        src2 = self.senet1(src2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.senet2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class BBTransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(BBTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        #   feat_dim = V Q T I
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.project_current_bias = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            # nn.Linear(d_model,d_model),
            # nn.Sigmoid()
        )
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        encoder_layer = CurrentBiasTransformerEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                               dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

    def calculate_current_bias(self, current, voltage, padding_masks):
        """
        Args:
            current: (batch_size, seq_length)
            voltage: (batch_size, seq_length)
            padding_masks: (batch_size, seq_length), True for valid positions, False for padding
        """
        valid_curr = padding_masks[:, 1:] & padding_masks[:, :-1]
        #current_diff = current[:, 1:] - current[:, :-1]
        voltage_diff = torch.abs(voltage[:, 1:] - voltage[:, :-1])

        #current_diff = torch.cat([torch.ones((current.size(0), 1), device=current.device), current_diff], dim=1)
        voltage_diff = torch.cat([torch.zeros((voltage.size(0), 1), device=voltage.device), voltage_diff], dim=1)
        # 添加一列全为False的列到valid_curr
        valid_curr = torch.cat(
            [torch.zeros((valid_curr.size(0), 1), dtype=torch.bool, device=valid_curr.device), valid_curr], dim=1)
        #combined_diff = current_diff * voltage_diff
        #combined_diff = voltage_diff * (current+1)
        epsilon = 1e-6
        combined_diff = torch.log(1+voltage_diff)*torch.sign(current)*(torch.abs(current)+epsilon)

        # 应用有效掩码
        combined_diff = torch.where(valid_curr, combined_diff, torch.zeros_like(combined_diff))
        # 确保偏置对于填充位置
        combined_diff = combined_diff * padding_masks.float()

        return combined_diff  # [batch_size, seq_len]

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, True means keep vector at this position, False means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # Extract current and voltage from X (assuming voltage is the second last feature)
        voltage = X[:, :, 0]
        current = X[:, :, -1]
        current_bias = self.calculate_current_bias(current, voltage, padding_masks)
        current_bias = self.project_current_bias(current_bias.unsqueeze(-1))  # [batch, seq_len, d_model]
        current_bias = current_bias.permute(1, 0, 2)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp)
        inp = self.pos_enc(inp)

        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks, current_bias=current_bias)

        output = self.act(output)  # [seq_len, batch_size, d_model]
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        output = self.dropout1(output)
        output = self.output_layer(output)  # [batch_size, seq_len, feat_num]

        return output



class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, current_bias=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, current_bias=current_bias)

        return output

class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)


        encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, num_classes)

    def build_output_module(self, d_model, num_classes):
        output_layer = nn.Linear(d_model, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        #output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
class BBTransformerEncoderforSOC(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, data_window_len, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(BBTransformerEncoderforSOC, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.project_current_bias = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
        )
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=512)


        encoder_layer = CurrentBiasTransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, num_classes, data_window_len)

    def build_output_module(self, d_ff, num_classes,data_window_len):



        #output_layer = LSTMModel(d_ff, d_ff//2, num_classes)
        output_layer = MLPforSOC(input_dim=d_ff, hidden_dim=d_ff//2, output_dim=num_classes, seq_len=data_window_len)
        return output_layer
    def calculate_current_bias(self, current, voltage, padding_masks):
        """
        Args:
            current: (batch_size, seq_length)
            voltage: (batch_size, seq_length)
            padding_masks: (batch_size, seq_length), True for valid positions, False for padding
        """
        valid_curr = padding_masks[:, 1:] & padding_masks[:, :-1]

        voltage_diff = torch.abs(voltage[:, 1:] - voltage[:, :-1])

        #current_diff = torch.cat([torch.ones((current.size(0), 1), device=current.device), current_diff], dim=1)
        voltage_diff = torch.cat([torch.zeros((voltage.size(0), 1), device=voltage.device), voltage_diff], dim=1)
        # 添加一列全为False的列到valid_curr
        valid_curr = torch.cat(
            [torch.zeros((valid_curr.size(0), 1), dtype=torch.bool, device=valid_curr.device), valid_curr], dim=1)
        #combined_diff = current_diff * voltage_diff
        #combined_diff = voltage_diff * (current+1)
        epsilon = 1e-6
        combined_diff = torch.log(1+voltage_diff)*torch.sign(current)*(torch.abs(current)+epsilon)

        # 应用有效掩码
        combined_diff = torch.where(valid_curr, combined_diff, torch.zeros_like(combined_diff))
        # 确保偏置对于填充位置
        combined_diff = combined_diff * padding_masks.float()

        return combined_diff  # [batch_size, seq_len]
    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        voltage = X[:, :, 0]
        current = X[:, :, -1]
        current_bias = self.calculate_current_bias(current, voltage, padding_masks)
        current_bias = self.project_current_bias(current_bias.unsqueeze(-1))  # [batch, seq_len, d_model]
        current_bias = current_bias.permute(1, 0, 2)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks, current_bias=current_bias)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
class BBTransformerEncoderforSOT(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, data_window_len, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(BBTransformerEncoderforSOT, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.project_current_bias = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
        )
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=512)


        encoder_layer = CurrentBiasTransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, num_classes, data_window_len)

    def build_output_module(self, d_ff, num_classes,data_window_len):


        #output_layer = MLPforSOT(input_dim=d_ff, hidden_dim=d_ff, output_dim=num_classes, seq_len=data_window_len)
        output_layer = LSTMModel(d_ff, d_ff//2, num_classes)
        return output_layer
    def calculate_current_bias(self, current, voltage, padding_masks):
        """
        Args:
            current: (batch_size, seq_length)
            voltage: (batch_size, seq_length)
            padding_masks: (batch_size, seq_length), True for valid positions, False for padding
        """
        valid_curr = padding_masks[:, 1:] & padding_masks[:, :-1]
        #current_diff = current[:, 1:] - current[:, :-1]
        voltage_diff = torch.abs(voltage[:, 1:] - voltage[:, :-1])

        #current_diff = torch.cat([torch.ones((current.size(0), 1), device=current.device), current_diff], dim=1)
        voltage_diff = torch.cat([torch.zeros((voltage.size(0), 1), device=voltage.device), voltage_diff], dim=1)
        # 添加一列全为False的列到valid_curr
        valid_curr = torch.cat(
            [torch.zeros((valid_curr.size(0), 1), dtype=torch.bool, device=valid_curr.device), valid_curr], dim=1)
        #combined_diff = current_diff * voltage_diff
        #combined_diff = voltage_diff * (current+1)
        epsilon = 1e-6
        combined_diff = torch.log(1+voltage_diff)*torch.sign(current)*(torch.abs(current)+epsilon)

        # 应用有效掩码
        combined_diff = torch.where(valid_curr, combined_diff, torch.zeros_like(combined_diff))
        # 确保偏置对于填充位置
        combined_diff = combined_diff * padding_masks.float()

        return combined_diff  # [batch_size, seq_len]
    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        voltage = X[:, :, 0]
        current = X[:, :, -1]
        current_bias = self.calculate_current_bias(current, voltage, padding_masks)
        current_bias = self.project_current_bias(current_bias.unsqueeze(-1))  # [batch, seq_len, d_model]
        current_bias = current_bias.permute(1, 0, 2)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks, current_bias=current_bias)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        #output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
class BBTransformerEncoderforSOH(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, data_window_len, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(BBTransformerEncoderforSOH, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.project_current_bias = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
        )
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=512)
        self.data_window_len = data_window_len

        encoder_layer = CurrentBiasTransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, num_classes)

    def build_output_module(self, d_ff, num_classes):


        output_layer = MLPforSOH(d_ff,d_ff//4,num_classes)

        return output_layer
    def calculate_current_bias(self, current, voltage, padding_masks):
        """
        Args:
            current: (batch_size, seq_length)
            voltage: (batch_size, seq_length)
            padding_masks: (batch_size, seq_length), True for valid positions, False for padding
        """
        valid_curr = padding_masks[:, 1:] & padding_masks[:, :-1]
        #current_diff = current[:, 1:] - current[:, :-1]
        voltage_diff = torch.abs(voltage[:, 1:] - voltage[:, :-1])

        #current_diff = torch.cat([torch.ones((current.size(0), 1), device=current.device), current_diff], dim=1)
        voltage_diff = torch.cat([torch.zeros((voltage.size(0), 1), device=voltage.device), voltage_diff], dim=1)
        # 添加一列全为False的列到valid_curr
        valid_curr = torch.cat(
            [torch.zeros((valid_curr.size(0), 1), dtype=torch.bool, device=valid_curr.device), valid_curr], dim=1)
        #combined_diff = current_diff * voltage_diff
        #combined_diff = voltage_diff * (current+1)
        epsilon = 1e-6
        combined_diff = torch.log(1+voltage_diff)*torch.sign(current)*(torch.abs(current)+epsilon)

        # 应用有效掩码
        combined_diff = torch.where(valid_curr, combined_diff, torch.zeros_like(combined_diff))
        # 确保偏置对于填充位置
        combined_diff = combined_diff * padding_masks.float()

        return combined_diff  # [batch_size, seq_len]
    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        voltage = X[:, :, 0]
        current = X[:, :, -1]

        current_bias = self.calculate_current_bias(current, voltage, padding_masks)
        current_bias = self.project_current_bias(current_bias.unsqueeze(-1))  # [batch, seq_len, d_model]
        current_bias = current_bias.permute(1, 0, 2)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks, current_bias=current_bias)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        #output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)
        output = self.act(output)
        return output
class FFN(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim, dropout=0.1, activation='gelu'):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = _get_activation_fn(activation)  # 激活函数
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x=x.squeeze(-1)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # 定义输出层
        self.linear = nn.Linear(hidden_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)


        out = self.dropout(out[:,-1,:])

        # 通过线性层输出
        out = self.linear(out)

        return out

class MLPforSOH(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation='gelu'):
        super(MLPforSOH, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = _get_activation_fn(activation)  # 激活函数
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = nn.Linear(100 * hidden_dim // 2, 128)
        self.linear4 = nn.Linear(128, 128)
        self.linear5 = nn.Linear(128, output_dim)



    def forward(self, x):
        # 初始输入值 x 被传入第一个全连接层


        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear5(x)


        return x
class MLPforSOC(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim, seq_len, dropout=0.3, activation='gelu'):
        super(MLPforSOC, self).__init__()


        self.activation = _get_activation_fn(activation)  # 激活函数
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = nn.Linear(seq_len * hidden_dim//2, 128)
        self.linear4 = nn.Linear(128, 128)
        self.linear5 = nn.Linear(128, output_dim)

    def forward(self, x):
        # 初始输入值 x 被传入第一个全连接层

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.dropout(x)
        x=self.linear5(x)

        return x

class MLPforSOT(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim, seq_len, dropout=0.5, activation='gelu'):
        super(MLPforSOT, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = _get_activation_fn(activation)  # 激活函数
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim//2)
        #self.linear3 = nn.Linear(seq_len*hidden_dim//2, 256)
        self.linear3 = nn.Linear(hidden_dim//2,256)
        self.linear4 = nn.Linear(256,256)
        self.linear5 = nn.Linear(256,1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)
        #x = x.reshape(x.shape[0], -1)
        x=self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear4(x)

        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear5(x)

        return x
class CNNforSOH(nn.Module):
    def __init__(self, d_model, kernel_size=3,dropout=0.5, activation='gelu'):
        super(CNNforSOH, self).__init__()
        # Conv1d 输入需要是 [batch, d_model, seq]，因此需要进行转置
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=64, kernel_size=kernel_size)
        # 自适应池化将输出调整到 [batch, 64, 1]，即每个序列输出一个值
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层，将 [batch, 64, 1] -> [batch, 1]
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 转换 [batch, seq, d_model] -> [batch, d_model, seq]
        x = x.transpose(1, 2)
        # 卷积操作
        x = self.conv(x)
        # 池化操作
        x = self.pool(x)
        # 去掉多余的维度
        x = x.squeeze(-1)
        # 全连接层输出
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)


        return x
