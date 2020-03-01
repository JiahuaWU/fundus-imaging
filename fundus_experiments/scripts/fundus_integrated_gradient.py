from zeiss_umbrella.config import FILE_OBSERVER_BASE_PATH, FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH
import sacred
from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver
import torch.nn as nn
import numpy as np
import torchvision
from zeiss_umbrella.integrated_gradient.utils import calculate_outputs_and_gradients
from zeiss_umbrella.integrated_gradient.integrated_gradients import random_baseline_integrated_gradients
from zeiss_umbrella.integrated_gradient.visualization import visualize
from zeiss_umbrella.fundus.setting_parser import get_baseline, get_optimizer, get_loss
from zeiss_umbrella.fundus.train import *
from zeiss_umbrella.fundus.adversarial import fgsm_k_image, fgsm_image, pgd
from zeiss_umbrella.fundus.data import get_fundus_train
from datetime import datetime as dt

ex = Experiment('integrated gradients')
template = ""
ex.observers.append(FileStorageObserver(FILE_OBSERVER_BASE_PATH,
                                        FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH, template))


# uncomment if you use progress bars
# from sacred.utils import apply_backspaces_and_linefeeds
# ex.captured_out_filter = apply_backspaces_and_linefeeds
# for more info see https://sacred.readthedocs.io/en/latest/collected_information.html#live-information


@ex.config
def my_config():
    adv_training_config = {'type': 'fgsm'}
    experiments_path = '/home/jiwu/interpretable-fundus/fundus_experiments'
    root_dir = 'data/fundus_preprocessed_512/train/'
    dataset = 'trainLabels.csv'
    grad_steps = 100
    savefigs = True
    trials = 1
    num_examples = 50
    target_label = 0
    polarity = "both"  # both or positive
    model_type = 'efficientnetb0'
    preprocessing = {'type': 'normalize'}
    device = 'cuda:0'
    weights_dir = 'corruption_experiments/efficientnetb0_corruption_imbalance_3/' + \
                  'train_efficientnetb0_normalize_baseline_unfreezed_crossentropy_parallel_corrupted'
    root_dir = os.path.join(experiments_path, root_dir)
    weights_dir = os.path.join(experiments_path, weights_dir)
    valid_rate = 0.3
    seed = 19660602

# adapted from pcam_experiments/pcam_integrated_gradient.py
@ex.automain
def run(_run: sacred.run.Run, adv_training_config, grad_steps, root_dir, dataset, trials, num_examples, polarity,
        model_type, savefigs, preprocessing, device, weights_dir, valid_rate, seed, target_label):
    import os
    now = dt.now()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loaders1, datasets, sizes, _, _ = get_fundus_train(root_dir=root_dir,
                                                       transform_dic=preprocessing,
                                                       original_csv_name=dataset,
                                                       seed=seed,
                                                       shuffle=False,
                                                       batch_size=num_examples,
                                                       valid_rate=valid_rate,
                                                       train_len=None, valid_len=None)

    loaders2, datasets2, sizes, _, _ = get_fundus_train(root_dir=root_dir,
                                                       transform_dic={'type': 'default'},
                                                       original_csv_name=dataset,
                                                       seed=seed,
                                                       shuffle=False,
                                                       batch_size=num_examples,
                                                       valid_rate=valid_rate,
                                                       train_len=None, valid_len=None,
                                                       valid_transform_dic = {'type': 'default'})
    print(f"Loading {num_examples} Examples")
    x_list, y_list, x_original_list = [], [], []
    n = 0
    it1, it2 = iter(datasets['valid']), iter(datasets2['valid'])
    while n < num_examples:
        x, y = next(it1)
        x_original, _ = next(it2)
        if y == target_label:
            x_list.append(x)
            y_list.append(y)
            x_original_list.append(x_original)
            n += 1
            print("get {} image(s) of label {}".format(n, y))
    x, y, x_original = torch.stack(x_list), torch.tensor(y_list), torch.stack(x_original_list)

    criterion = torch.nn.CrossEntropyLoss()
    model = get_baseline(model_type, weights_dir=weights_dir)
    model.eval()
    model = model.to(device)
    print("Loaded model {} in ".format(weights_dir), dt.now() - now)

    EPSILON_fgsm = adv_training_config.get('epsilon_fgsm', 1.0 / 255.0)
    ALPHA_fgsm = adv_training_config.get('alpha_fgsm', None)
    STEPS = adv_training_config.get('steps', None)
    if adv_training_config['type'] == "fgsm":
        advers_exp = fgsm_image
    elif adv_training_config['type'] == "fgsm_k_image":
        advers_exp = fgsm_k_image
    elif adv_training_config['type'] == 'pgd':
        advers_exp = pgd

    for i in range(num_examples):
        img = x[i:i + 1, :, :, :]
        img_original = x_original[i:i + 1, :, :, :]
        level = y[i:i + 1]
        adv_img, path = advers_exp(img, level, model, criterion, device, EPSILON_fgsm, steps=STEPS, alpha=ALPHA_fgsm,
                                   return_path=True)
        paths = None if path is None else [[t.cpu().detach() for t in reversed(path)]]
        baselines = [adv_img.detach()]
        adv_pred = torch.argmax(model(baselines[0].to(device)))
        print("Got eval image ")

        print("Calculating integrated gradients")
        now = dt.now()
        # calculate the gradient and the label index
        gradients, label_index = calculate_outputs_and_gradients([img], model, None, device=device)
        gradients = np.transpose(gradients[0], (1, 2, 0))
        img_gradient_overlay = visualize(gradients, img_original, clip_above_percentile=99, clip_below_percentile=0,
                                         overlay=True, mask_mode=True, polarity=polarity)
        img_gradient = visualize(gradients, img_original, clip_above_percentile=99, clip_below_percentile=0,
                                 overlay=False,
                                 polarity=polarity)
        imgdisp = img.permute(0, 2, 3, 1).squeeze()

        attributions, trial_grads, baselines = random_baseline_integrated_gradients(img, model, label_index,
                                                                                    calculate_outputs_and_gradients,
                                                                                    steps=grad_steps,
                                                                                    num_random_trials=1, device=device,
                                                                                    baselines=baselines,
                                                                                    paths=paths)
        # attributions, n2 = integrated_gradients(img, model, label_index, calculate_outputs_and_gradients,None,50,True)
        img_integrated_gradient_overlay = visualize(attributions, img_original, clip_above_percentile=99,
                                                    clip_below_percentile=0, overlay=True, mask_mode=True,
                                                    polarity=polarity)
        img_integrated_gradient = visualize(attributions, img_original, clip_above_percentile=99,
                                            clip_below_percentile=0,
                                            overlay=False, polarity=polarity)

        attributions0, trial_grads0, baselines0 = random_baseline_integrated_gradients(img, model, label_index,
                                                                                       calculate_outputs_and_gradients,
                                                                                       steps=grad_steps,
                                                                                       num_random_trials=trials,
                                                                                       device=device)
        # attributions, n2 = integrated_gradients(img, model, label_index, calculate_outputs_and_gradients,None,50,True)
        img_integrated_gradient_overlay0 = visualize(attributions0, img_original, clip_above_percentile=99,
                                                     clip_below_percentile=0,
                                                     overlay=True, mask_mode=True, polarity=polarity)
        img_integrated_gradient0 = visualize(attributions0, img_original, clip_above_percentile=99,
                                             clip_below_percentile=0,
                                             overlay=False, polarity=polarity)

        print("Got gradients and vizualizations in ", dt.now() - now)
        norms = [np.array([np.linalg.norm(g) for g in grads]) for grads in trial_grads]
        norms0 = [np.array([np.linalg.norm(g) for g in grads]) for grads in trial_grads0]

        def plot_output_image(img, img_gradient, img_gradient_overlay, img_integrated_gradient,
                              img_integrated_gradient_overlay, norms, b):
            def mk_arr(x):
                if type(x) != np.ndarray:
                    return x.cpu().detach().numpy()
                else:
                    return x

            b, img, img_gradient, img_gradient_overlay, img_integrated_gradient, img_integrated_gradient_overlay = [
                mk_arr(x) for x in [b, img, img_gradient, img_gradient_overlay, img_integrated_gradient,
                                    img_integrated_gradient_overlay]]

            fig, ax = plt.subplots(3, 4, figsize=(20, 10))
            fig.suptitle("Results for {}".format(weights_dir))
            ax[0, 0].imshow(img)
            ax[0, 0].set_title('Ground truth(level={},pred={})'.format(level[0], label_index[0]))
            ax[0, 1].imshow(img_gradient_overlay)
            ax[0, 1].set_title('Gradient overlay')
            ax[0, 2].imshow(img_gradient)
            ax[0, 2].set_title('Image gradient')
            for i, n in enumerate(norms):
                ax[1, 0].plot(n, label=f"Trial{i}")
            # ax[1,0].legend()
            attack = adv_training_config['type']
            ax[1, 0].set_title("Gradient norm moving from\n {} base to GT".format(attack))
            ax[1, 1].imshow(img_integrated_gradient_overlay)
            ax[1, 1].set_title('Integrated gradient overlay\n({} base)'.format(attack))
            ax[1, 2].imshow(img_integrated_gradient)
            ax[1, 2].set_title('Image integrated gradient\n ({} base)'.format(attack))
            ax[0, 3].imshow(np.transpose(b.squeeze(), (1, 2, 0)))
            ax[0, 3].set_title('Adversarial baseline (pred={})'.format(adv_pred.item()))

            for i, n in enumerate(norms0):
                ax[2, 0].plot(n, label=f"Trial{i}")
            ax[2, 0].legend()
            ax[2, 0].set_title("Gradient norm moving from random baseline to GT")
            ax[2, 1].imshow(img_integrated_gradient_overlay0)
            ax[2, 1].set_title('Integrated gradient overlay \n(rand base)')
            ax[2, 2].imshow(img_integrated_gradient0)
            ax[2, 2].set_title('Image integrated gradient \n(rand base)')
            fig.delaxes(ax[1, 3])
            fig.delaxes(ax[2, 3])
            return fig, ax

        figout, _ = plot_output_image(imgdisp, img_gradient, img_gradient_overlay, img_integrated_gradient,
                                      img_integrated_gradient_overlay, norms, baselines[0])
        if savefigs:
            if "adversarial" in weights_dir:
                robust = "_robust"
            else:
                robust = ""
            if "full_res" in weights_dir:
                res = "_fullres"
            else:
                res = ""
            fig_name = "fundus_integrated_adv_{}_eps{:0.4f}_{}_{}{}{}.png".format(adv_training_config['type'],
                                                                             EPSILON_fgsm, i, model_type, robust, res)
            figout.savefig(fig_name)
            _run.add_artifact(fig_name)
            os.remove(fig_name)
        if i != num_examples - 1:
            plt.close(figout)

    plt.show(block=True)
