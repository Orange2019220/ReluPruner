import torch
import numpy as np
from . import augmentations


mixture_width = 3 #Number of augmentation chains to mix per augmented example
mixture_depth = -1 # Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
aug_severity = 3 # Severity of base augmentation operators

def aug(image, preprocess, all_ops=False):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=True, is_instance=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.is_instance = is_instance

  def __getitem__(self, i):
    if self.is_instance:
      x, y, _ = self.dataset[i]
    else:
      x, y = self.dataset[i]
    if self.no_jsd:
      if self.is_instance:
        return aug(x, self.preprocess), y, i
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)
  

  