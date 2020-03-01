import numpy as np
import matplotlib.pyplot as plt
import torch

G = [0, 1.0, 0]
R = [1.0, 0, 0]


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2,
                     plot_distribution=False):
    m = compute_threshold_by_top_percentage(attributions, percentage=100 - clip_above_percentile,
                                            plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(attributions, percentage=100 - clip_below_percentile,
                                            plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (np.abs(transformed) >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.abs(np.sum(flat_attributions))
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        # Generate a plot of sorted intgrad scores.
        values_to_plot = np.where(cum_sum >= 95)[0][0]
        values_to_plot = max(values_to_plot, threshold_idx)
        plt.plot(np.arange(values_to_plot), sorted_attributions[:values_to_plot])
        plt.axvline(x=threshold_idx)
        plt.show()
    return threshold


def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        pos = np.clip(attributions, 0, 1)
        neg = np.clip(attributions, -1, 0)
        neg = np.abs(neg)
        return pos, neg


def overlay_function(attributions, image):
    # sanity check
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    return np.clip(0.7 * image + 0.5 * attributions, 0, 1)


def visualize(attributions, image, positive_channel=G, negative_channel=R, polarity='positive',
              clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False,
              structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True,
              mask_mode=False, plot_distribution=False, channels_first=True):
    if channels_first:
        image = image.permute(0, 2, 3, 1).squeeze(0)
    if len(attributions.shape) == 4:
        attributions = attributions.squeeze(0)
    if polarity == 'both':
        pos_attr, neg_attr = polarity_function(attributions, polarity=polarity)
        attributions = np.zeros_like(pos_attr)
        attributions_mask = np.zeros_like(pos_attr)
        for attr, chan in zip([pos_attr, neg_attr], [positive_channel, negative_channel]):
            # convert the attributions to the gray scale
            attr = convert_to_gray_scale(attr)
            attr = linear_transform(attr, clip_above_percentile, clip_below_percentile, 0,
                                    plot_distribution=plot_distribution)
            amask = attr.copy()
            if morphological_cleanup:
                raise NotImplementedError
            if outlines:
                raise NotImplementedError
            attr = np.expand_dims(attr, 2) * chan
            attributions += attr
            attributions_mask += np.expand_dims(amask, 2)

        if overlay:
            if not mask_mode:
                attributions = overlay_function(attributions, image)
            else:
                # attributions =attributions_mask
                imgd = image.detach().numpy()
                attributions = np.clip(attributions + imgd * 0.7, 0, 1)
                # attributions = attributions[:, :, (2, 1, 0)]
    else:
        if polarity == 'positive':
            attributions = polarity_function(attributions, polarity=polarity)
            channel = positive_channel
        elif polarity == 'negative':
            attributions = polarity_function(attributions, polarity=polarity)
            channel = negative_channel

        # convert the attributions to the gray scale
        attributions = convert_to_gray_scale(attributions)
        attributions = linear_transform(attributions, clip_above_percentile, clip_below_percentile, 0.0,
                                        plot_distribution=plot_distribution)
        attributions_mask = attributions.copy()
        if morphological_cleanup:
            raise NotImplementedError
        if outlines:
            raise NotImplementedError
        attributions = np.expand_dims(attributions, 2) * channel
        if overlay:
            if not mask_mode:
                attributions = overlay_function(attributions, image)
            else:
                imgd = image.detach().numpy()
                attributions = np.expand_dims(attributions_mask, 2)
                attributions = np.clip(attributions * imgd, 0, 1)
                # attributions = attributions[:, :, (2, 1, 0)]
    return attributions
