'''
@Project ：MG-Net 
@File    ：Dir_CVAE.py
@IDE     ：PyCharm 
@Author  ：王一梵
@Date    ：2025/9/26 10:36 
'''

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EDLClassifier(nn.Module):
    def __init__(self, encoder, dim_encoder_out, dim_hidden, num_classes, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.encoder = encoder
        self.Conv1 = nn.Conv2d(dim_encoder_out, dim_hidden, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.Conv2 = nn.Conv2d(dim_hidden, num_classes, 1, 1)

    def forward(self, x):
        # instead of using regular softmax or sigmoid to output a probability distribution over the classes,
        # we output a positive vector, using a softplus on the logits, as the evidence over the classes
        x = self.encoder(x)
        x = F.relu(self.Conv1(x))
        x = self.dropout(x)
        x = self.Conv2(x)
        return F.softplus(x)

    # @torch.inference_mode()
    def predict(self, x, return_uncertainty=True):
        evidences = x
        # alphas are the parameters of the Dirichlet distribution that models the probability distribution over the
        # class probabilities and strength is the Dirichlet strength
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=1, keepdim=True)
        probabilities = alphas / strength

        if return_uncertainty:
            total_uncertainty = self.num_classes / strength
            beliefs = evidences / strength
            return probabilities, total_uncertainty, beliefs
        else:
            return probabilities


class UnsupervisedKLDivergenceLoss(nn.Module):
    def __init__(self, num_class, prior_alpha=0.5):
        """
        KL divergence loss for unsupervised learning, without a prior.
        """
        super().__init__()
        self.num_classes = num_class
        self.prior_alpha = prior_alpha

    def forward(self, evidences):
        """
        Compute the unsupervised KL divergence loss without a prior.
        Args:
            evidences (Tensor): Predicted evidences, shape (batch_size, num_classes).
        Returns:
            loss (Tensor): Scalar loss value.
        """
        # Compute Dirichlet parameters
        alphas = evidences + 1.0  # Dirichlet parameters
        strength = torch.sum(alphas, dim=1, keepdim=True)  # Sum over classes

        # KL divergence terms
        first_term = torch.lgamma(strength) - torch.sum(torch.lgamma(alphas), dim=1, keepdim=True)
        second_term = torch.sum(
            (alphas - 1.0) * (torch.digamma(alphas) - torch.digamma(strength)), dim=1, keepdim=True
        )

        # Compute mean loss
        loss = torch.mean((first_term + second_term))
        return loss
