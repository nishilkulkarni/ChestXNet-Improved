"""
Contains functionality for crreating PyTorh DataLoaders for the image classifiation task.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import urllib.request
import tarfile
import shutil
import pandas as pd



def download_data():
# Downloads dataset from the required location
# URLs for the .gz files
  links = [
      'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
      'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
      'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
      'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
      'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
      'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
      'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
      'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
      'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
      'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
      'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
      'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
      ]

 

  # Create data directory if it does not exist
  if not os.path.exists('data'):
      os.makedirs('data')

  for idx, link in enumerate(links):
      fn = 'images_%02d.tar.gz' % (idx + 1)
      print(f'Downloading {fn}...')
      urllib.request.urlretrieve(link, fn)
      print(f'{fn} downloaded.')

      # Extract the downloaded tar.gz file
      print(f'Extracting {fn}...')
      with tarfile.open(fn, 'r:gz') as tar:
          tar.extractall(path='data')
      print(f'{fn} extracted.')

      # Optionally, remove the downloaded tar.gz file after extraction
      os.remove(fn)
      print(f'{fn} removed.')
  
  # Load the CSV file
  files = ['trainlabels.csv', 'vallabels.csv', 'testlabels.csv']
  for i in range(3):
      
    csv_file = files[i]
    df = pd.read_csv(csv_file)

    # Directory where the images are currently stored
    data_dir = 'data/images'

    # Create directories for each class and move images
    for index, row in df.iterrows():
        img_file = row['id']
        
        # Determine the class for the current image
        img_class = None
        for col in df.columns[1:]:
            if row[col] == 1:
                img_class = col
                break

        # If a class was found, proceed with moving the file
        if img_class is not None:
            # Create class directory if it does not exist

            temp = None
            if i == 2:
                temp = 'test'
            else:
                temp = 'train'

            class_dir = os.path.join(data_dir, temp, img_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # Define source and destination paths
            src_path = os.path.join(data_dir, img_file)
            dst_path = os.path.join(class_dir, img_file)
            
            # Move the image to the corresponding class directory
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                print(f'Moved {img_file} to {class_dir}')
            else:
                print(f'{img_file} not found in {data_dir}')
        else:
            print(f'No class found for {img_file}')


def create_dataloaders(
        train_dir : str,
        test_dir : str,
        transform : transforms.Compose,
        batch_size: int 
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    
    #Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    #Get class names
    class_names = train_data.classes

    #Turn images into data loades
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle = True
                                    )
    test_dataloader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle = True
                                    )
    
    return train_dataloader, test_dataloader, class_names