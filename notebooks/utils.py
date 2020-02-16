import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_jsd_loss(logits_clean, logits_aug1, logits_aug2, lamb=12.):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean,
                                        dim=1), F.softmax(logits_aug1,
                                                          dim=1), F.softmax(logits_aug2,
                                                                            dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1.).log()
    jsd = lamb * (F.kl_div(p_mixture, p_clean, reduction="batchmean") +
                  F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
                  F.kl_div(p_mixture, p_aug2, reduction="batchmean")) / 3.
    return jsd


def compute_distil_loss(student_logit, teacher_logit, temp=4.):
    student_prob = F.softmax(student_logit / temp, dim=-1)
    teacher_prob = F.softmax(teacher_logit / temp, dim=-1).log()
    loss = F.kl_div(teacher_prob, student_prob, reduction="batchmean")
    return loss


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    for i, (input, _, _) in enumerate(tbar):
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))