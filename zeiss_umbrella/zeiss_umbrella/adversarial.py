import torch
import numpy as np
# FGSM attack code from  https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def fgsm_k_image(data,target,epsilon,model,criterion,device,alpha=1.0/255.0,steps=None,return_path=False):
    # from https://arxiv.org/pdf/1611.01236.pdf adapted for range 0 1 instead of 0 255
    if steps is None:
        steps=int(np.round(min(epsilon+4./255,1.25*epsilon)*255))
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    perturbed_image=data.detach()
    # Set requires_grad attribute of tensor. Important for Attack
    perturbed_image.requires_grad = True
    path=[perturbed_image]
    for k in range(steps):

        #print("step",k)
        # Forward pass the data through the model
        output = model(perturbed_image)
        init_pred = torch.max(output,1)[1] # get the index of the max log-probability

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
            perturbed_image = perturbed_image.detach() + alpha*sign_data_grad
            # Adding clipping to maintain [0,1] range

            perturbed_image = torch.min(torch.max(perturbed_image, data-epsilon), data+epsilon)
            # Return the perturbed image
            perturbed_image=torch.clamp(perturbed_image,0,1)
        if return_path:
            path.append(perturbed_image.detach())
        perturbed_image.requires_grad=True
    if return_path:
        return perturbed_image.detach(),path
    else:
        return perturbed_image.detach()

def fgsm_image(data,target,epsilon,model,criterion,device,skip_wrong=False,**kwargs):
    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = torch.max(output,1)[1] # get the index of the max log-probability

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
    data.requires_grad=False

    if "return_path" in kwargs:
        return perturbed_data,None
    else:
        return perturbed_data
