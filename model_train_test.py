import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
import math
from .adversarial import fgsm_image, fgsm_k_image, boundary_attack_image
from tqdm import tqdm, trange
import time
import copy
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

"""
Functionalites for training and testing models.
"""

# Adapted from zeiss_umbrella.resnet.train_model
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, valid=True, ex=None, seed=None,
                scheduler=None, fundus_dataset=None, adv_training_config=None, num_epochs=25, return_best=False):
    """
    Trains the given model,and returns it,  if return_best=True, also returns the best
    model state dict as a second return value
    """
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epochbar = trange(num_epochs)
    for epoch in epochbar:
        if scheduler:
            epochbar.set_description('Epoch {}/{}, learning rate: {}'.format(epoch, num_epochs - 1, scheduler.get_lr()))
        else:
            epochbar.set_description('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase if valid is true, only training if valid is false.
        if valid:
            phase_list = ['train', 'valid']
        else:
            phase_list = ['train']

        for phase in phase_list:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_attacks = 0
            ground_truth = torch.Tensor().type(torch.long).to(device)
            predictions = torch.Tensor().type(torch.long).to(device)

            # Iterate over data.
            batches = tqdm(dataloaders[phase], total=dataset_sizes[phase] / dataloaders[phase].batch_size)
            for inputs, labels in batches:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # if model
                    if model.__class__.__name__ == 'Inception3' and phase == 'train':
                        outputs, aux = model(inputs)
                    else:
                        outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    BATCH_ACC_TOL = adv_training_config.get('batch_acc_tol', 0.7)
                    ADVERSARIAL_WEIGHT = adv_training_config.get('weight', 0.3)
                    batch_acc = (preds == labels).type(torch.float).sum() / labels.shape[0]
                    if phase == 'train':
                        if batch_acc > BATCH_ACC_TOL and adv_training_config['type'] != 'baseline' \
                                and adv_training_config is not None:
                            adversarial_samples, adversarial_labels = get_adversarial_samples(inputs, labels,
                                                                                              adv_training_config)
                            if adversarial_samples and adversarial_labels:
                                num_attacks += 1
                                adversarial_loss = criterion(model(adversarial_examples), adversarial_labels)
                                loss = loss * (1 - ADVERSARIAL_WEIGHT) + ADVERSARIAL_WEIGHT * adversarial_loss

                            # clean up model gradients
                            model.zero_grad()

                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                ground_truth = torch.cat((ground_truth, labels))
                predictions = torch.cat((predictions, preds))
                batches.set_description("Running loss:{},Running corrects:{}".format(running_loss, running_corrects))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            balanced_acc = balanced_accuracy_score(ground_truth.cpu().tolist(), predictions.cpu().tolist())
            cks = cohen_kappa_score(ground_truth.cpu().tolist(), predictions.cpu().tolist(), weights='quadratic')

            # Output metrics using sacred ex
            if ex:
                record_training_info(ex, phase, epoch_loss, epoch_acc, balanced_acc, cks)
            if phase == 'train':
                print("number of attack performed: {}".format(num_attacks))
            print('{} Loss: {:.4f} Acc: {:.4f} Balanced Acc: {:.4f} cohen kappa score: {}'
                  .format(phase, epoch_loss, epoch_acc, balanced_acc, cks))
            # deep copy the model
            if phase == 'valid' and cks > best_acc:
                best_acc = cks
                best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    if return_best:
        return model, best_model_wts, optimizer
    else:
        return model, optimizer


def record_training_info(ex, phase, epoch_loss, epoch_acc, balanced_acc, cks):
    if phase == 'train':
        ex.log_scalar("train loss", epoch_loss)
        ex.log_scalar("train accuracy", epoch_acc.item())
        ex.log_scalar("train balanced accuracy", balanced_acc)
    elif phase == 'valid':
        ex.log_scalar("valid loss", epoch_loss)
        ex.log_scalar("valid accuracy", epoch_acc.item())
        ex.log_scalar("valid balanced accuracy", balanced_acc)
        ex.log_scalar("valid cohen square kappa", cks)


def get_adversarial_samples(inputs, labels, adv_training_config):
    EPSILON_fgsm = adv_training_config.get('epsilon_fgsm', 1.0 / 255.0)
    ALPHA_fgsm = adv_training_config.get('alpha_fgsm', None)
    STEPS = adv_training_config.get('steps', None)
    EPSILON_dba = adv_training_config.get('epsilon_dba', 1.)
    DELTA_dba = adv_training_config.get('delta_dba', 0.1)
    N_STEP_MAX_dba = adv_training_config.get('n_step_max_dba', 250)
    E_STEP_MAX_dba = adv_training_config.get('e_step_max_dba', 20)
    D_STEP_MAX_dba = adv_training_config.get('d_step_max_dba', 10)
    UQSRT = adv_training_config.get('unqualified_sample_ratio_tol_dba', 0.4)
    DIFF_TOL_dba = adv_training_config.get('diff_tol_dba', 10)
    if adv_training_config['type'] == 'fgsm':
        adversarial_samples = fgsm_image(inputs, labels, EPSILON_fgsm, model, criterion,
                                         device=device)
        adversarial_labels = labels.clone()
    elif adv_training_config['type'] == 'fgsm_k_image':
        adversarial_samples = fgsm_k_image(inputs, labels, model, criterion, device=device,
                                           epsilon=EPSILON_fgsm, steps=STEPS, alpha=ALPHA_fgsm)
        adversarial_labels = labels.clone()
    elif adv_training_config['type'] == 'pgd':
        adversarial_samples = fgsm_k_image(inputs, labels, model, criterion, device=device,
                                           epsilon=EPSILON_fgsm, steps=STEPS, rand=True)
        adversarial_labels = labels.clone()
    elif adv_training_config['type'] == 'boundary_attack':
        adversarial_samples, adversarial_labels = boundary_attack_image(model, device,
                                                                        inputs, labels,
                                                                        seed=seed,
                                                                        fundus_dataset=fundus_dataset,
                                                                        epsilon=EPSILON_dba,
                                                                        delta=DELTA_dba,
                                                                        n_step_max=N_STEP_MAX_dba,
                                                                        e_step_max=E_STEP_MAX_dba,
                                                                        diff_tol=DIFF_TOL_dba,
                                                                        d_step_max=D_STEP_MAX_dba,
                                                                        unqualified_sample_ratio_tol=UQSRT)
    else:
        adversarial_samples, adversarial_labels = None, None
    return adversarial_samples, adversarial_labels


def find_lr(model, optimizer, criterion, trn_loader, device, init_value=1e-8, final_value=10., beta=0.98):
    """
    Basic learning rate finder implemented in fastai
    quoted from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    :param model: model to be trained
    :param optimizer: optimizer to be used
    :param criterion: loss function
    :param trn_loader: training loader
    :param device: 'cpu' or 'cuda'
    :param init_value: the initial value for the learning rate
    :param final_value: the final value for the learning rate
    :param beta: a weighted parameter
    :return: log10 of the lrs and the corresponding loss,
             good to pick the lr an order of magnitude smaller than the best
    """
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in trn_loader:
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.data
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


class FocalLoss_SM(nn.Module):
    """
    Focal loss for softmax function. Note that in our case the labels are mutually exclusive.
    Another possibility is the focal loss for sigmoid function which assumes that the labels are not mutually exclusive.
    """

    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss_SM, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt.exp()
        if self.alpha is not None:
            logpt = logpt * self.alpha.gather(0, target)

        loss = -1. * (1. - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def test_model(model, dataloader, dataset_size, criterion, device):
    since = time.time()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        ground_truth = torch.Tensor().type(torch.long).to(device)
        predictions = torch.Tensor().type(torch.long).to(device)

        # Iterate over data.
        # i=0
        batches = tqdm(dataloader)
        for inputs, labels in batches:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            ground_truth = torch.cat((ground_truth, labels))
            predictions = torch.cat((predictions, preds))
            # print(i)
            # i=i+1
            batches.set_description('Batch loss {:.4f}, batch accuracy {:.4f}'.format(loss.item() * inputs.size(0),
                                                                                      torch.sum(preds == labels.data)))
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        balanced_acc = balanced_accuracy_score(ground_truth.cpu().tolist(), predictions.cpu().tolist())
        chs = cohen_kappa_score(ground_truth.cpu().tolist(), predictions.cpu().tolist(), weights='quadratic')

        print('Loss: {:.4f} Acc: {:.4f} Balanced Acc: {:.4f} cohen kappa: {:.4f}'
              .format(epoch_loss, epoch_acc, balanced_acc, chs))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
