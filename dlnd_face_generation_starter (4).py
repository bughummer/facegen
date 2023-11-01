#!/usr/bin/env python
# coding: utf-8

# # Face Generation
# 
# In this project, you'll define and train a Generative Adverserial network of your own creation on a dataset of faces. Your goal is to get a generator network to generate *new* images of faces that look as realistic as possible!
# 
# The project will be broken down into a series of tasks from **defining new architectures training adversarial networks**. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.
# 
# ### Get the Data
# 
# You'll be using the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train your adversarial networks.
# 
# This dataset has higher resolution images than datasets you have previously worked with (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.
# 
# ### Pre-processed Data
# 
# Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.
# 
# <img src='assets/processed_face_data.png' width=60% />
# 
# > If you are working locally, you can download this data [by clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)
# 
# This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data `processed-celeba-small/`.

# In[2]:


# run this once to unzip the file
get_ipython().system('unzip processed-celeba-small.zip')


# In[1]:


from glob import glob
from typing import Tuple, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

import tests


# In[2]:


data_dir = 'processed_celeba_small/celeba/'


# ## Data pipeline
# 
# The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.
# 
# ### Pre-process and Load the Data
# 
# Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This *pre-processed* dataset is a smaller subset of the very large CelebA dataset and contains roughly 30,000 images. 
# 
# Your first task consists in building the dataloader. To do so, you need to do the following:
# * implement the get_transforms function
# * create a custom Dataset class that reads the CelebA data

# ### Exercise: implement the get_transforms function
# 
# The `get_transforms` function should output a [`torchvision.transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose) of different transformations. You have two constraints:
# * the function takes a tuple of size as input and should **resize the images** to the input size
# * the output images should have values **ranging from -1 to 1**

# In[3]:


from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def get_transforms(size: Tuple[int, int]) -> Callable:
    """Transforms to apply to the image."""
    # Define a list of transformations to apply
    transforms = [
        Resize(size),  # Resize the image to the specified size
        ToTensor(),     # Convert the image to a PyTorch tensor
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to range from -1 to 1
    ]
    
    return Compose(transforms)


# ### Exercise: implement the DatasetDirectory class
# 
# 
# The `DatasetDirectory` class is a torch Dataset that reads from the above data directory. The `__getitem__` method should output a transformed tensor and the `__len__` method should output the number of files in our dataset. You can look at [this custom dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) for ideas. 

# In[4]:


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Callable

class DatasetDirectory(Dataset):
    """
    A custom dataset class that loads images from a folder.
    Args:
    - directory: location of the images
    - transform: transform function to apply to the images
    - extension: file format
    """
    def __init__(self, 
                 directory: str, 
                 transforms: Callable = None, 
                 extension: str = '.jpg'):
        # Initialize the dataset by storing the directory, transform function, and extension
        self.directory = directory
        self.transforms = transforms
        self.extension = extension
        
        # Create a list of image file paths in the specified directory with the given extension
        self.image_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        # Return the number of elements in the dataset, which is the length of the list of image file paths
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load an image and apply transformation."""
        # Get the image path at the specified index
        image_path = self.image_paths[index]
        
        # Open the image using PIL
        image = Image.open(image_path)
        
        # Apply the transformation if it's provided
        if self.transforms is not None:
            image = self.transforms(image)
        
        # Return the transformed image as a PyTorch tensor
        return image


# In[5]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to verify your dataset implementation
dataset = DatasetDirectory(data_dir, get_transforms((64, 64)))
tests.check_dataset_outputs(dataset)


# The functions below will help you visualize images from the dataset.

# In[6]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""

def denormalize(images):
    """Transform images from [-1.0, 1.0] to [0, 255] and cast them to uint8."""
    return ((images + 1.) / 2. * 255).astype(np.uint8)

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
    img = dataset[idx].numpy()
    img = np.transpose(img, (1, 2, 0))
    img = denormalize(img)
    ax.imshow(img)


# ## Model implementation
# 
# As you know, a GAN is comprised of two adversarial networks, a discriminator and a generator. Now that we have a working data pipeline, we need to implement the discriminator and the generator. 
# 
# Feel free to implement any additional class or function.

# ### Exercise: Create the discriminator
# 
# The discriminator's job is to score real and fake images. You have two constraints here:
# * the discriminator takes as input a **batch of 64x64x3 images**
# * the output should be a single value (=score)
# 
# Feel free to get inspiration from the different architectures we talked about in the course, such as DCGAN, WGAN-GP or DRAGAN.
# 
# #### Some tips
# * To scale down from the input image, you can either use `Conv2d` layers with the correct hyperparameters or Pooling layers.
# * If you plan on using gradient penalty, do not use Batch Normalization layers in the discriminator.

# In[7]:


from torch.nn import Module


# In[8]:


import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Define the layers of the discriminator
        self.layers = nn.Sequential(
            # Input: 64x64x3
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16x128
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8x256
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4x512
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Output a single value between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the discriminator layers
        return self.layers(x)


# In[9]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to check your discriminator implementation
discriminator = Discriminator()
tests.check_discriminator(discriminator)


# ### Exercise: create the generator
# 
# The generator's job creates the "fake images" and learns the dataset distribution. You have three constraints here:
# * the generator takes as input a vector of dimension `[batch_size, latent_dimension, 1, 1]`
# * the generator must outputs **64x64x3 images**
# 
# Feel free to get inspiration from the different architectures we talked about in the course, such as DCGAN, WGAN-GP or DRAGAN.
# 
# #### Some tips:
# * to scale up from the latent vector input, you can use `ConvTranspose2d` layers
# * as often with Gan, **Batch Normalization** helps with training

# In[10]:


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()

        # Define the layers of the generator
        self.layers = nn.Sequential(
            # Input: [batch_size, latent_dim, 1, 1]
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 4x4x512
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8x256
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32x64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output a 64x64x3 image with pixel values in the range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the generator layers
        return self.layers(x)

# Instantiate the generator
latent_dim = 100  # You can choose the dimensionality of the latent vector
generator = Generator(latent_dim)


# In[11]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to verify your generator implementation
latent_dim = 128
generator = Generator(latent_dim)
tests.check_generator(generator, latent_dim)


# ## Optimizer
# 
# In the following section, we create the optimizers for the generator and discriminator. You may want to experiment with different optimizers, learning rates and other hyperparameters as they tend to impact the output quality.

# ### Exercise: implement the optimizers

# In[12]:


import torch.optim as optim
from torch.nn import Module

def create_optimizers(generator: Module, discriminator: Module):
    """This function should return the optimizers of the generator and the discriminator."""
    
    # Define the optimizer for the generator
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Define the optimizer for the discriminator
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    return g_optimizer, d_optimizer


# ## Losses implementation
# 
# In this section, we are going to implement the loss function for the generator and the discriminator. You can and should experiment with different loss function.
# 
# Some tips:
# * You can choose the commonly used the binary cross entropy loss or select other losses we have discovered in the course, such as the Wasserstein distance.
# * You may want to implement a gradient penalty function as discussed in the course. It is not required and the code will work whether you implement it or not.

# ### Exercise: implement the generator loss
# 
# The generator's goal is to get the discriminator to think its generated images (= "fake" images) are real.

# In[13]:


import torch
import torch.nn.functional as F

def generator_loss(fake_logits):
    """Generator loss, takes the fake scores as inputs."""
    
    # Calculate the BCE loss for the generator
    # The target is 1 because we want the discriminator to classify fake images as real (outputting 1)
    loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    
    return loss


# ### Exercise: implement the discriminator loss
# 
# We want the discriminator to give high scores to real images and low scores to fake ones and the discriminator loss should reflect that.

# In[14]:


import torch
import torch.nn.functional as F

def discriminator_loss(real_logits, fake_logits):
    """Discriminator loss, takes the real and fake logits as inputs."""
    
    # Calculate the BCE loss for the real and fake samples separately
    # The target for real samples is 1, and for fake samples is 0
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    
    # The total loss is the sum of the losses for real and fake samples
    loss = real_loss + fake_loss
    
    return loss


# ### Exercise (Optional): Implement the gradient Penalty
# 
# In the course, we discussed the importance of gradient penalty in training certain types of Gans. Implementing this function is not required and depends on some of the design decision you made (discriminator architecture, loss functions).

# In[15]:


# def gradient_penalty(discriminator, real_samples, fake_samples):
#     """ This function enforces """
#     gp = 0
#     # TODO (Optional): implement the gradient penalty
#     return gp


# In[16]:


import torch
import torch.autograd as autograd

def gradient_penalty(discriminator, real_samples, fake_samples):
    """This function enforces gradient penalty for WGAN-GP."""
    
    # Generate random epsilon values for interpolation between real and fake samples
    epsilon = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    
    # Create interpolated samples by combining real and fake samples
    interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
    
    # Calculate the discriminator's scores for the interpolated samples
    interpolated_scores = discriminator(interpolated_samples)
    
    # Compute the gradients of the interpolated scores with respect to the interpolated samples
    gradients = autograd.grad(outputs=interpolated_scores, inputs=interpolated_samples,
                             grad_outputs=torch.ones_like(interpolated_scores),
                             create_graph=True, retain_graph=True)[0]
    
    # Calculate the gradient penalty as the L2 norm of the gradients
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


# ## Training
# 
# 
# Training will involve alternating between training the discriminator and the generator. You'll use your functions real_loss and fake_loss to help you calculate the discriminator losses.
# 
# * You should train the discriminator by alternating on real and fake images
# * Then the generator, which tries to trick the discriminator and should have an opposing loss function

# ### Exercise: implement the generator step and the discriminator step functions
# 
# Each function should do the following:
# * calculate the loss
# * backpropagate the gradient
# * perform one optimizer step

# In[17]:


import torch
import torch.optim as optim
from typing import Dict
from torch.autograd import Variable
from torch.nn import functional as F

# Assuming you have already created optimizers for the generator and discriminator
g_optimizer, d_optimizer = create_optimizers(generator, discriminator)

def generator_step(batch_size: int, latent_dim: int) -> Dict:
    """One training step of the generator."""
    
    # Generate random noise as the input to the generator
    noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
    
    # Generate fake images from the noise
    fake_images = generator(noise)
    
    # Calculate the discriminator's scores for the fake images
    fake_scores = discriminator(fake_images)
    
    # Calculate the generator loss
    g_loss = -torch.mean(fake_scores)  # Generator wants the discriminator to output 1
    
    # Backpropagate the gradient and update the generator's parameters
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    
    return {'loss': g_loss.item()}

def discriminator_step(batch_size: int, latent_dim: int, real_images: torch.Tensor) -> Dict:
    """One training step of the discriminator."""
    
    # Generate random noise as the input to the generator
    noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
    
    # Generate fake images from the noise
    fake_images = generator(noise)
    
    # Calculate the discriminator's scores for real and fake images
    real_scores = discriminator(real_images)
    fake_scores = discriminator(fake_images.detach())
    
    # Calculate the discriminator loss
    d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
    
    # Compute the gradient penalty (optional)
    gp = gradient_penalty(discriminator, real_images, fake_images)
    
    # Add the gradient penalty to the discriminator loss (if used)
    d_loss += gp
    
    # Backpropagate the gradient and update the discriminator's parameters
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    
    return {'loss': d_loss.item(), 'gp': gp.item()}

# Example usage:
# You would typically call these functions in a training loop, alternating between generator and discriminator steps.
# In the training loop, you'd provide the batch_size, latent_dim, and real_images.


# ### Main training loop
# 
# You don't have to implement anything here but you can experiment with different hyperparameters.

# In[18]:


from datetime import datetime


# In[19]:


# you can experiment with different dimensions of latent spaces
latent_dim = 128

# update to cpu if you do not have access to a gpu
device = 'cuda'

# number of epochs to train your model
n_epochs = 1

# number of images in each batch
batch_size = 64


# In[20]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
print_every = 50

# Create optimizers for the discriminator D and generator G
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
g_optimizer, d_optimizer = create_optimizers(generator, discriminator)

dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        num_workers=4, 
                        drop_last=True,
                        pin_memory=False)


# In[21]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""

def display(fixed_latent_vector: torch.Tensor):
    """ helper function to display images during training """
    fig = plt.figure(figsize=(14, 4))
    plot_size = 16
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
        img = fixed_latent_vector[idx, ...].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = denormalize(img)
        ax.imshow(img)
    plt.show()


# ### Exercise: implement the training strategy
# 
# You should experiment with different training strategies. For example:
# 
# * train the generator more often than the discriminator. 
# * added noise to the input image
# * use label smoothing
# 
# Implement with your training strategy below.

# In[22]:


fixed_latent_vector = torch.randn(16, latent_dim, 1, 1).float().cuda()

losses = []
for epoch in range(n_epochs):
    for batch_i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        
        ####################################
        
        # TODO: implement the training strategy
        
        ####################################
        
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            d = d_loss['loss'].item()
            g = g_loss['loss'].item()
            losses.append((d, g))
            # print discriminator and generator loss
            time = str(datetime.now()).split('.')[0]
            print(f'{time} | Epoch [{epoch+1}/{n_epochs}] | Batch {batch_i}/{len(dataloader)} | d_loss: {d:.4f} | g_loss: {g:.4f}')
    
    # display images during training
    generator.eval()
    generated_images = generator(fixed_latent_vector)
    display(generated_images)
    generator.train()


# In[ ]:


import torch
from datetime import datetime

# Define the number of epochs and other hyperparameters
n_epochs = 100
print_every = 200  # Print loss every 'print_every' batches
train_generator_more = True  # Train the generator more often

# Assuming you have already defined the 'fixed_latent_vector', 'generator', and 'dataloader'

losses = []
for epoch in range(n_epochs):
    for batch_i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        
        ####################################
        # Training Strategy
        ####################################
        
        if train_generator_more:
            # Train the generator twice for every discriminator step
            for _ in range(2):
                g_loss = generator_step(real_images.size(0), latent_dim)
        
        # Train the discriminator
        d_loss = discriminator_step(real_images.size(0), latent_dim, real_images)
        
        if batch_i % print_every == 0:
            # Append discriminator loss and generator loss
            d = d_loss['loss']
            g = g_loss['loss']
            losses.append((d, g))
            
            # Print discriminator and generator loss
            time = str(datetime.now()).split('.')[0]
            print(f'{time} | Epoch [{epoch+1}/{n_epochs}] | Batch {batch_i}/{len(dataloader)} | d_loss: {d:.4f} | g_loss: {g:.4f}')
    
    # Display generated images during training
    generator.eval()
    generated_images = generator(fixed_latent_vector)
    display(generated_images)
    generator.train()


# ### Training losses
# 
# Plot the training losses for the generator and discriminator.

# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


# ### Question: What do you notice about your generated samples and how might you improve this model?
# When you answer this question, consider the following factors:
# * The dataset is biased; it is made of "celebrity" faces that are mostly white
# * Model size; larger models have the opportunity to learn more features in a data feature space
# * Optimization strategy; optimizers and number of epochs affect your final result
# * Loss functions

# **Answer:** (Write your answer in this cell)

# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb".  
# 
# Submit the notebook using the ***SUBMIT*** button in the bottom right corner of the Project Workspace.

# In[ ]:




