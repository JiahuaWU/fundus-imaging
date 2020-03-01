import torch
from zeiss_umbrella.resnet import resnet18
from zeiss_umbrella.integrated_gradient.integrated_gradients import integrated_gradients
from zeiss_umbrella.integrated_gradient.utils import calculate_outputs_and_gradients
import numpy as np
import torch.nn.functional as F


def testIntegratedGradients():
    model = resnet18(pretrained=True)
    cuda = torch.cuda.is_available()
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    seed = torch.randint(high=10000000, size=(1, 1), device=device).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    model.eval()
    x_baseline = torch.randn((1, 3, 256, 256))
    x_input = torch.randn((1, 3, 256, 256))
    output = model(x_input.to(device))
    output_baseline = model(x_baseline.to(device))
    output_index = torch.argmax(output, dim=1).item()
    output_baseline_index = torch.argmax(output_baseline, dim=1).item()
    y_input = F.softmax(output)[0][output_index]
    y_baseline = F.softmax(output_baseline)[0][output_baseline_index]
    expected_val = y_input.item() - y_baseline.item()
    integrated_grad, _, _ = integrated_gradients(x_input, model, None, calculate_outputs_and_gradients,
                                                 steps=1000, cuda=cuda, baseline=x_baseline, path=None)
    print(integrated_grad.sum())
    print(expected_val)
    diff = abs(integrated_grad.sum() - expected_val) / abs(expected_val)
    if diff < 1e-4:
        print('Integrated Gradients Test past')
    else:
        print('Integrated Gradients not passed, error: {}'.format(diff))


if __name__ == '__main__':
    testIntegratedGradients()
