import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiLLFunction(nn.Module):
    def __init__(self, beta=0.5, weight_f=0.9):
        super(MultiLLFunction, self).__init__()
        self.beta = beta
        self.weight_f = weight_f
        self.weight_b = 1 - weight_f

    def forward(self, predictions, targets):
        """Multilabel loss
                Args:
                    predictions: a tensor containing pixel wise predictions
                        shape: [batch_size, num_classes, width, height]
                    targets: a tensor containing binary labels
                        shape: [batch_size, num_classes, width, height]
                """


        # predictions_01 = torch.sigmoid(predictions)
        # log1 = torch.log(predictions_01 + 1e-3)
        # log2 = torch.log(1 - predictions_01 + 1e-3)
        # term1 = torch.mul(torch.mul(targets, self.beta), log1)
        # term2 = torch.mul(torch.mul(1 - targets, 1 - self.beta), log2)
        # sum_of_terms = -term1 - term2


        # background loss
        prediction = torch.sigmoid(predictions[:, 0:1, :, :])
        log1 = torch.log(prediction + 1e-3)
        log2 = torch.log(1 - prediction + 1e-3)
        term1 = torch.mul(torch.mul(targets[:, 0:1, :, :], self.beta), log1)
        term2 = torch.mul(torch.mul(1 - targets[:, 0:1, :, :], 1 - self.beta), log2)
        sum_of_terms_0 = -term1 - term2

        # foreground loss
        prediction = torch.sigmoid(predictions[:, 1:2, :, :])
        log1 = torch.log(prediction + 1e-3)
        log2 = torch.log(1 - prediction + 1e-3)
        term1 = torch.mul(torch.mul(targets[:, 1:2, :, :], self.beta), log1)
        term2 = torch.mul(torch.mul(1 - targets[:, 1:2, :, :], 1 - self.beta), log2)
        sum_of_terms_1 = -term1 - term2

        sum_of_terms = sum_of_terms_0 * self.weight_b + sum_of_terms_1 * self.weight_f

        return torch.sum(sum_of_terms)