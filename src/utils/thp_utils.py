import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_non_pad_mask(seq, pad: int = 0):
    """Get the non-padding positions."""

    assert seq.dim() == 2
    return seq.ne(pad).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, pad: int = 0):
    """For masking out the padding part of key sequence."""

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """For masking out the subsequent info, i.e., masked self-attention."""

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5, attn_dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward neural network."""

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """Log-likelihood of events."""

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """Log-likelihood of non-events, using linear interpolation."""

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """Log-likelihood of non-events, using Monte Carlo integration."""

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * torch.rand(
        [*diff_time.size(), num_samples], device=data.device
    )
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types, pad: int = 0):
    """Log-likelihood of sequence."""

    non_pad_mask = get_non_pad_mask(types, pad).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    if model.integral_biased is True:
        non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    else:
        non_event_ll = compute_integral_unbiased(
            model, data, time, non_pad_mask, type_mask
        )
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """Event prediction loss, cross entropy or label smoothing."""

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """Time prediction loss."""

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
