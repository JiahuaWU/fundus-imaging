import numpy as np
import torch
import scipy as sp


# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps=50, device=None, baseline=None,
                         path=None):
    if device and "cpu" in inputs.device.type:
        inputs = inputs.to(device)
    if baseline is None:
        baseline = torch.rand_like(inputs, device=device)
    elif device and "cpu" in baseline.device.type:
        baseline = baseline.to(device)
    if path is None:
        # scale inputs and compute gradients
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    else:
        scaled_inputs = [p.to(device) for p in path]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, device)
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    integrated_grad = (inputs.cpu() - baseline.cpu()).squeeze(0).permute(1, 2, 0).numpy() * avg_grads
    return integrated_grad, grads, baseline


# TODO: play around with other integration methods/paths
def integrate_gradients(inputs, model, target_label_idx, predict_and_gradients, steps=50, device=None, baseline=None,
                        path=None):
    if baseline is None:
        baseline = torch.rand_like(inputs)

    def path_step(alpha):
        diff = (inputs - baseline)
        pimg = baseline + alpha * diff
        grads, _ = predict_and_gradients(pimg, model, target_label_idx, device)

    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, device)
    # average along scaling path, with equal weighting => riemman approx as in paper
    avg_grads = np.average(grads[:-1], axis=0)
    # move channel from first to last for display
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    integrated_grad = (inputs - baseline).squeeze(0).permute(1, 2, 0).numpy() * avg_grads
    return integrated_grad, grads, baseline


def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps,
                                         num_random_trials, device, baselines=None, paths=None):
    all_intgrads = []
    trial_grads = []
    if baselines is None:
        bl = []
    else:
        bl = baselines
    for i in range(num_random_trials):
        if baselines is None:
            b = None
        else:
            b = baselines[i]
        if paths is None:
            p = None
        else:
            p = paths[i]
        integrated_grad, grads, baseline = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients,
                                                                steps=steps, device=device, baseline=b, path=p)
        all_intgrads.append(integrated_grad)
        trial_grads.append(grads)
        if baselines is None:
            bl.append(baseline)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads, trial_grads, bl
