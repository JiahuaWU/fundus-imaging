import torch
import numpy as np
from torchvision import transforms


# Adapted from zeiss_umbrella.adversarial
# FGSM attack code from  https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_k_image(data, target, model, criterion, device,
                 epsilon=1.0 / 255.0, alpha=None, steps=None, return_path=False, rand=False):
    """
    Generate adversarial examples using bim(rand=False) or Madry-PGD(rand=True).
    :param data: a set of input images from which we generate the adversarial examples
    :param target: the corresponding target labels of the data
    :param epsilon: maximum pixelwise amplitude of perturbation
    :param model: model to be attacked
    :param criterion: loss for the generation of the adversarial examples
    :param device: cpu or cuda
    :param alpha: step size of each step
    :param steps: number of steps
    :param return_path: the path to store the adversarial examples
    :param rand: starting from a random point within the linf box or not. Yes for Madry-PGD, no for BIM
    :return: a set of adversarial examples.
    """
    # from https://arxiv.org/pdf/1611.01236.pdf adapted for range 0 1 instead of 0 255
    if steps is None:
        steps = int(np.round(min(epsilon + 4. / 255, 1.25 * epsilon) * 255))

    # Alpha is set to be 2.5 * epsilon / steps as in http://arxiv.org/abs/1706.06083
    if alpha is None:
        alpha = 2.5 * epsilon / steps
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        if rand:
            perturbed_image = data + (-2 * epsilon) * torch.rand_like(data) + epsilon
        else:
            perturbed_image = data
    # Set requires_grad attribute of tensor. Important for Attack
    perturbed_image.requires_grad = True
    path = [perturbed_image]
    for _ in range(steps):

        # print("step",k)
        # Forward pass the data through the model
        output = model(perturbed_image)

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        with torch.no_grad():
            # Collect datagrad
            data_grad = perturbed_image.grad.data

            # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_image = perturbed_image.detach() + alpha * sign_data_grad

            # Projected the image on the l_inf circle
            perturbed_image = torch.min(torch.max(perturbed_image, data - epsilon), data + epsilon)

            # Adding clipping to maintain [0,1] range
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
        if return_path:
            path.append(perturbed_image.detach())
        perturbed_image.requires_grad = True
    # Return the perturbed image
    if return_path:
        return perturbed_image.detach(), path
    else:
        return perturbed_image.detach()


def pgd(data, target, model, criterion, device,
        epsilon=1.0 / 255.0, alpha=None, steps=None, return_path=False):
    return fgsm_k_image(data, target, model, criterion, device,
                        epsilon=epsilon, alpha=alpha, steps=steps, return_path=return_path, rand=True)


def fgsm_image(data, target, model, criterion, device, epsilon, skip_wrong=False, **kwargs):
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = torch.max(output, 1)[1]  # get the index of the max log-probability

    # If the initial prediction is wrong, dont bother attacking, just move on
    if skip_wrong and init_pred.item() != target.item():
        return None

    # Calculate the loss
    loss = criterion(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    # so we don't collect unnecessary grads if we reuse this data...
    data.requires_grad = False

    if "return_path" in kwargs:
        return perturbed_data, None
    else:
        return perturbed_data


# Boundary attack
def orthogonal_perturbation(deltas, prev_samples, target_samples, device):
    """
    Calculate the orthogonal move
    :param device: cpu or cuda
    :param deltas: amplitudes of the move of size (batch_size)
    :param prev_samples: previous sample of size (batch_size, c, h, w)
    :param target_samples: target sample of size (batch_size, c, h, w)
    :return: the perturbation of size (batch_size, c, h, w)
    """
    prev_samples, target_samples = prev_samples.to(device), target_samples.to(device)
    # Generate perturbation
    perturb = torch.randn_like(prev_samples) / 255  # (batch_size, c, h, w)
    # Normalize and times delta * d(o, o^{k-1})
    perturb *= 1. / get_diff(perturb, torch.zeros_like(perturb), device).unsqueeze(-1).unsqueeze(-1)
    perturb *= (deltas * torch.mean(get_diff(target_samples, prev_samples, device))).unsqueeze(-1).unsqueeze(
        -1).unsqueeze(-1)

    # Calculate unit vector pointing to target samples.
    diff = (target_samples - prev_samples).type(torch.float32)  # (batch_size, c, h, w)
    diff *= 1. / get_diff(target_samples, prev_samples, device).unsqueeze(-1).unsqueeze(-1)

    # Projection onto the equidistant disc
    # perturb -= torch.matmul(perturb, diff) * diff

    # Calculate the inner product corresponding to frobenius norm: tr(sqrt(A.t().matmul(B)))
    inner_prods = torch.einsum('...ii->...i', perturb.transpose(2, 3).matmul(diff)).sum(dim=2)
    # Projection onto diff
    proj = inner_prods.unsqueeze(-1).unsqueeze(-1) * diff
    perturb -= proj
    t = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ones_normalized = t(torch.ones_like(perturb)[0]).repeat(perturb.shape[0], 1, 1, 1)
    zeros_normalized = t(torch.zeros_like(perturb)[0]).repeat(perturb.shape[0], 1, 1, 1)
    overflow = (prev_samples + perturb) - ones_normalized
    perturb -= overflow * (overflow > 0).type(torch.float32)
    underflow = (prev_samples + perturb) - zeros_normalized
    perturb -= underflow * (underflow < 0).type(torch.float32)
    return perturb.to(device)


def forward_perturbation(epsilons, prev_samples, target_samples, device):
    """
    Calculate the perturbation needed towards target sample
    :param device: cpu or cuda
    :param epsilons: collection of epsilon of each entry in the batch size = (batch)
    :param prev_samples: previous samples
    :param target_samples: target samples
    :return: the perturbation of size (batch_size, c, h, w)
    """
    prev_samples, target_samples = prev_samples.to(device), target_samples.to(device)
    perturb = (target_samples - prev_samples).type(torch.float32)
    perturb *= 1. / get_diff(target_samples, prev_samples, device).unsqueeze(-1).unsqueeze(-1)
    perturb *= epsilons.unsqueeze(-1).unsqueeze(-1)
    return perturb.to(device)


def get_diff(samples_1, samples_2, device):
    """
    Get the frobenius norm of difference between sample_1 and sample_2
    :param device: cpu or cuda
    :param samples_1: (batch_size, c, h, w) or (batch_size, h, w, c)
    :param samples_2: (batch_size, c, h, w) or (batch_size, h, w, c)
    :return: (batch_size, 3) dimension tensor of difference at each dimension
    """
    samples_1, samples_2 = samples_1.to(device), samples_2.to(device)
    if samples_1.shape[1] != 3:
        samples_1 = samples_1.clone().permute(0, 3, 1, 2).to(device)
    if samples_2.shape[1] != 3:
        samples_2 = samples_2.clone().permute(0, 3, 1, 2).to(device)
    batch_size = samples_1.shape[0]
    num_channel = samples_1.shape[1]
    diff = samples_1 - samples_2
    return torch.norm(diff.view(batch_size, num_channel, -1), dim=2).to(device)


def generate_target_samples(data, labels, fundus_dataset=None, target_indices=(4, 5, 300, 6), device='cuda'):
    """
    Generate target samples for decision boundary attack from the given data. Basically, for each input label, we take
    a sample of different label in the data as a target sample. If all the labels are the same, we take a distinct label
    from the target_indices which contains indices of the fundus dataset where labels are 0 - 4 and use the selected label
    as well as the corresponding image to construnct a target image batch.
    :param device:
    :param data: input images
    :param labels: target labels of data
    :param fundus_dataset: fundus dataset object
    :param target_indices: 5 positions in the fundus dataset where the labels are respectively 0 - 4
    :return: target samples along with their labels used for decision boundary attack
    """
    # If all the labels are the same
    batch_size = data.shape[0]
    all_zero = (labels != labels[0]).bitwise_not().all()
    zero_and_the_other = len(torch.unique(labels)) == 2 and 0 in torch.unique(labels)
    if all_zero or zero_and_the_other:
        data_all = torch.Tensor()
        labels_all = []
        for index in target_indices:
            data_all = torch.cat((data_all, fundus_dataset[index][0].unsqueeze(0)))
            labels_all.append(torch.tensor(fundus_dataset[index][1]))
        labels_all = torch.stack(labels_all).to(device)
        if all_zero:
            result_indices = torch.where((labels_all != labels[0].to(device)))
        elif zero_and_the_other:
            result_indices = torch.where((labels_all != torch.unique(labels)[1].to(device)))
        result_indices = result_indices[torch.randperm(len(result_indices))]
        target_labels = labels_all[result_indices][0].repeat(batch_size, 1)
        target_samples = data_all[result_indices][0].repeat(batch_size, 1, 1, 1)
        return target_samples, target_labels.view(batch_size)

    else:
        result_indices = []
        for label in labels:
            distinct_indices = torch.where((labels != label) * (labels != 0))
            result_indices.append(distinct_indices[torch.randperm(len(distinct_indices))][0])
        result_indices = torch.stack(result_indices)
        target_labels = labels[result_indices].clone()
        target_samples = data[result_indices].clone()
        return target_samples, target_labels


def generate_initial_samples(data, labels, model, device, max_iter=100, epsilon=3.0 / 255.0):
    data, labels = data.to(device), labels.to(device)
    init_samples = data.detach().clone()
    n_iter = 0
    correct = torch.max(model(init_samples), 1)[1] == labels
    while correct.any() and n_iter < max_iter:
        init_samples = torch.rand_like(init_samples)
        correct = torch.max(model(init_samples), 1)[1] == labels
        n_iter += 1
    print("generate {} initial samples".format(correct.bitwise_not().type(torch.int).sum()))
    return init_samples[correct.bitwise_not()], correct.bitwise_not()


def move_to_boundary(model, epsilons, adversarial_samples, target_samples, init_preds, d_step_max, n_calls, device):
    """
    Move first step to the boundary: first coincide with the target sample and gradually reduce step size
    wrong/correct_indices is used for navigating in the global tensor (tensor with size of qualified candidates)
    wrong/correct is used for navigating in the wrongly classified images that need to be treated (trial samples)
    """
    d_step_1 = 0

    while True:
        # Initialize trial indices
        if d_step_1 == 0:
            trial_indices = torch.arange(len(adversarial_samples)).to(device)

        step_size = epsilons[trial_indices].unsqueeze(-1) * get_diff(adversarial_samples[trial_indices],
                                                                     target_samples[trial_indices], device)

        trial_samples = adversarial_samples[trial_indices] + forward_perturbation(step_size, adversarial_samples[
            trial_indices], target_samples[trial_indices], device)
        trial_outputs = model(trial_samples)
        n_calls += 1
        d_step_1 += 1

        correct = torch.max(trial_outputs, 1)[1] == init_preds[trial_indices]
        wrong = correct.bitwise_not()

        # Calculate corresponding indices in the whole adversarial batch
        correct_indices = trial_indices[correct]
        wrong_indices = trial_indices[wrong]

        # Update adversarial examples and step sizes
        adversarial_samples[correct_indices] = trial_samples[correct]
        epsilons[wrong_indices] *= 0.8

        # Update trial indices
        trial_indices = trial_indices[wrong]

        if correct.all() or d_step_1 > d_step_max:
            return epsilons, adversarial_samples, n_calls


def move_and_tuning(model, adversarial_samples, target_samples, init_preds, n_calls, device, parameter, move_type,
                    num_trial, step_max, reduce_threshold=0.2, increase_threshold=0.7,
                    increase=0.9, decrease=0.9):
    """
    make a move and adjust the step sizes(parameter) according to statistics of num_trial trials.
    :param num_trial: number of trials
    :param step_max: maximum number of steps
    :param reduce_threshold: decrease the step size if the ratio of valid samples is smaller than this value
    :param increase_threshold: increase the step size if the ratio of valid samples is smaller than this value
    :param increase: increase the step size by 1 / increase times
    :param decrease: decrease the step size by decrease times
    """
    if move_type == 'forward':
        print("\tForward step...")
    if move_type == 'orthogonal':
        print("\tOrthogonal step...")
    step = 0
    while True:
        step += 1
        print("\t#{}".format(step))
        # Stop updating correct samples
        if step == 1:
            trial_indices = torch.arange(len(adversarial_samples)).to(device)

        trial_samples = adversarial_samples[trial_indices].repeat(num_trial, 1, 1, 1).to(device)
        trial_target_samples = target_samples[trial_indices].repeat(num_trial, 1, 1, 1).to(device)
        trial_parameter = parameter[trial_indices].repeat(num_trial).to(device)

        if move_type == 'orthogonal':
            trial_samples += orthogonal_perturbation(trial_parameter, trial_samples, trial_target_samples, device)
        if move_type == 'forward':
            step_sizes = trial_parameter.unsqueeze(-1) * get_diff(trial_samples, trial_target_samples, device)
            trial_samples += forward_perturbation(step_sizes, trial_samples, trial_target_samples, device)

        trial_outputs = model(trial_samples)
        n_calls += num_trial * len(trial_indices)

        # predictions of size (batch * num_trial)
        trial_preds = torch.max(trial_outputs, 1)[1]
        # print("trial predictions:{}".format(trial_preds))
        # print("initial predictions:{}".format(init_preds))

        d_scores = torch.mean((trial_preds.view(num_trial, -1) == init_preds[trial_indices]).type(torch.float32), dim=0)
        # print("d_scores: {}".format(d_scores))
        non_zero = d_scores > 0.0
        case1 = non_zero * (d_scores < reduce_threshold)
        case2 = d_scores > increase_threshold
        zero = non_zero.bitwise_not()

        # Calculate corresponding indices in the whole adversarial example batch
        case1_indices = trial_indices[case1]
        case2_indices = trial_indices[case2]
        non_zero_indices = trial_indices[non_zero]
        zero_indices = trial_indices[zero]

        # Update step sizes
        parameter[case1_indices] *= decrease
        parameter[case2_indices] /= increase
        parameter[zero_indices] *= decrease
        # print("Parameter: {}".format(parameter))

        # Take one of the valid orthogonal perturbation
        non_zero_row_indices = []

        # Take out non zero elements
        correct_pred_positions = torch.where(
            trial_preds.view(num_trial, -1)[:, non_zero] == init_preds[non_zero_indices])

        # Loop over non zero elements and take one valid sample
        for index in range(non_zero.type(torch.int).sum()):
            first_col_to_be_index = torch.where(index == correct_pred_positions[1])[0][0]
            non_zero_row_indices.append(correct_pred_positions[0][first_col_to_be_index])

        # Update adversarial samples
        if len(non_zero_row_indices) != 0:
            non_zero_row_indices = torch.stack(non_zero_row_indices)
            adversarial_samples[non_zero_indices] = torch.stack(trial_samples.chunk(num_trial, dim=0))[
                (non_zero_row_indices, torch.where(non_zero)[0])]

        # Update trial indices
        trial_indices = trial_indices[zero]
        # Break the loop if all samples are within the correct region.
        if non_zero.all() or step > step_max:
            return parameter, adversarial_samples, n_calls


def boundary_attack_image(model, device,
                          data, labels, untarget=False, skip_zero=False, fundus_dataset=None,
                          target_indices=(4, 5, 300, 6),
                          epsilon=1., delta=0.1, seed=None,
                          n_step_max=250, e_step_max=20, diff_tol=10, d_step_max=20, unqualified_sample_ratio_tol=0.2):
    """
    Batch implementation of decision boundary attack which allows to produce adversarial examples from input data.
    For a input batch, we shuffle it to construct a target batch and optimize the input images towards it. The images which
    the model cannot correctly classify will be directly returned. The adversarial examples whose maximum difference from
    the target examples of the three channels is greater than the diff_tol are considered as "bad samples" and will be
    discarded at return.
    Based on https://arxiv.org/pdf/1712.04248.pdf
    :param unqualified_sample_ratio_tol: return if the ratio of the "bad adv samples" is smaller than this threshold
    :param d_step_max: maximum number of delta steps
    :param target_indices: Indices of data of labels 1 - 5
    :param n_step_max: maximum number of
    :param labels: target labels of input data
    :param data: input images
    :param fundus_dataset: Fundus_Dataset object used when all the labels are the same
    :param diff_tol: return if difference between target sample and adversarial sample smaller than diff_tol
    :param e_step_max: maximum number of epsilon steps
    :param delta: size of delta step (orthogonal move)
    :param epsilon: size of epsilon step (step towards target sample)
    :param model: model to be evaluated
    :param device: cpu or cuda
    :return: adversarial examples along with the corresponding target labels
    """
    if seed:
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
    # Load the data, labels to device
    data, labels = data.to(device), labels.to(device)

    if untarget:
        init_samples, success = generate_initial_samples(data.detach(), labels.detach(), model, device)
        target_samples, target_labels = data.detach()[success], labels.detach()[success]
        # Forward pass the data through the model
        init_outputs = model(data[success])
        init_preds = torch.max(init_outputs, 1)[1]  # get the index of the max log-probability
        correctly_classified = init_preds == labels[success]

    else:
        init_samples = data
        # Generate target samples from data
        target_samples, target_labels = generate_target_samples(data.detach(), labels.detach(),
                                                                fundus_dataset=fundus_dataset,
                                                                target_indices=target_indices, device=device)
        # Forward pass the data through the model
        init_outputs = model(data)
        init_preds = torch.max(init_outputs, 1)[1]  # get the index of the max log-probability
        correctly_classified = init_preds == labels

    # Load target_samples, target_labels to device
    target_samples, target_labels = target_samples.to(device), target_labels.to(device)

    # Generate epsilons of size batch_size
    batch_size = data.detach().shape[0]

    with torch.no_grad():

        # If the classifier cannot classify correctly the initial training data,
        # no need to generate adversarial examples
        qualified_candidates = correctly_classified

        # If skip zero, we skip images with label 0
        if skip_zero:
            qualified_candidates *= labels != 0

        num_qualified_candidates = qualified_candidates.type(torch.int).sum()
        target_samples = target_samples[qualified_candidates]
        target_labels = target_labels[qualified_candidates]
        adversarial_samples = init_samples[qualified_candidates].clone().to(device)
        init_preds = init_preds[qualified_candidates]
        epsilons = (torch.ones(num_qualified_candidates) * epsilon).to(device)
        deltas = (torch.ones(num_qualified_candidates) * delta).to(device)
        print("Initial Diff :")
        print(get_diff(adversarial_samples, target_samples, device))

        if adversarial_samples.shape[0] == 0:
            return data[correctly_classified.bitwise_not()].clone().to(device), \
                   labels[correctly_classified.bitwise_not()].clone().to(device)

        n_steps = 0
        n_calls = 0
        epsilons, adversarial_samples, n_calls = move_to_boundary(model, epsilons, adversarial_samples, target_samples,
                                                                  init_preds,
                                                                  d_step_max, n_calls, device)
        print("After first move:")
        print(get_diff(adversarial_samples, target_samples, device))

        while True:
            print("Step #{}...".format(n_steps))
            deltas, adversarial_samples, n_calls = move_and_tuning(model, adversarial_samples, target_samples,
                                                                   init_preds, n_calls, device,
                                                                   move_type='orthogonal', parameter=deltas,
                                                                   step_max=d_step_max, num_trial=20,
                                                                   reduce_threshold=0.2, increase_threshold=0.8,
                                                                   increase=0.9, decrease=0.9)
            print("After orthgonal move:")
            # print("deltas: {}".format(deltas))
            print(get_diff(adversarial_samples, target_samples, device))
            epsilons, adversarial_samples, n_calls = move_and_tuning(model, adversarial_samples, target_samples,
                                                                     init_preds, n_calls, device,
                                                                     move_type='forward', parameter=epsilons,
                                                                     step_max=e_step_max, num_trial=10,
                                                                     reduce_threshold=0.2, increase_threshold=0.8,
                                                                     increase=0.5, decrease=0.5)
            print("After forward move:")
            print(get_diff(adversarial_samples, target_samples, device))

            n_steps += 1
            diff = get_diff(adversarial_samples, target_samples, device)
            print(diff)
            print("{} steps".format(n_steps))
            print("Mean Squared Error: {}".format(torch.mean(diff).item()))
            unqualified_samples_num = (torch.max(diff, dim=1).values > diff_tol).type(torch.int).sum()
            if diff.mean().item() <= diff_tol or n_steps > n_step_max \
                    or unqualified_samples_num < unqualified_sample_ratio_tol * num_qualified_candidates:
                break

        # We only return the valid samples
        adversarial_samples = adversarial_samples[(torch.max(diff, dim=1).values < diff_tol)]
        target_labels = target_labels[(torch.max(diff, dim=1).values < diff_tol)]

        # append wrongly classified samples for further training
        adversarial_samples = torch.cat(
            (adversarial_samples, data[correctly_classified.bitwise_not()].clone().to(device)))
        target_labels = torch.cat((target_labels, labels[correctly_classified.bitwise_not()].clone().to(device)))

        print("Generate {} adversarial samples".format(len(adversarial_samples)))
        print("Total number of calls: {}".format(n_calls))
    return adversarial_samples, target_labels
