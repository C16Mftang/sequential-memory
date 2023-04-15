from __future__ import print_function
import os
import os.path
import errno
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import numpy as np
import math

from src.utils import *

class DataWrapper(Dataset):
    """
    Class to wrap a dataset. Assumes X and y are already
    torch tensors and have the right data type and shape.
    
    Parameters
    ----------
    X : torch.Tensor
        Features tensor.
    y : torch.Tensor
        Labels tensor.
    """
    def __init__(self, X):
        self.features = X
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], []
    
def generate_correlated_binary_patterns(P, N, b, device, seed=1):
    np.random.seed(seed)
    X = np.zeros((int(P), int(N)))
    template = np.random.choice([-1, 1], size=N)
    prob = (1 + b) / 2
    for i in range(P):
        for j in range(N):
            if np.random.binomial(1, prob) == 1:
                X[i, j] = template[j]
            else:
                X[i, j] = -template[j]
            
        # revert the sign
        if np.random.binomial(1, 0.5) == 1:
            X[i, j] *= -1

    return to_torch(X, device)

def load_aliased_mnist(seed):
     # Set random seed for PyTorch random number generator
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Set up MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter dataset to include only digits 1, 2, and 3
    mnist_123 = [img for img, label in mnist if label in [1, 2, 3]]

    # Sample 5 random indices from the filtered dataset
    indices = torch.randperm(len(mnist_123))[:5]

    # Extract images corresponding to the sampled indices
    sequence = [mnist_123[i] for i in indices]

    # Replace the last two images with the first two images
    sequence[3], sequence[4] = sequence[1], sequence[0]

    # Convert images to PyTorch tensors and stack into a sequence tensor
    sequence_tensor = torch.stack(sequence).squeeze()

    return sequence_tensor

def load_sequence_mnist(seed, seq_len, order=True, binary=True):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 28, 28))

    if order:
        # Loop through each digit class and randomly sample one image from each class
        for i in range(seq_len):
            indices = torch.where(mnist.targets == i)[0]
            idx = torch.randint(0, indices.size()[0], (1,))
            img, _ = mnist[indices[idx][0]]
            sequence[i] = img.squeeze()

    else:
        # Sample `seq_len` random images from the MNIST dataset
        indices = torch.randint(0, len(mnist), (seq_len,))
        for i, idx in enumerate(indices):
            img, _ = mnist[idx]
            sequence[i] = img.squeeze()

    if binary:
        sequence[sequence > 0.5] = 1
        sequence[sequence <= 0.5] = -1

    return sequence

def load_sequence_emnist(seed, seq_len):
    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    emnist = datasets.EMNIST(root='./data', train=True, split='balanced', download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 28, 28))

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    i = 0
    while i < seq_len:
        idx = torch.randint(len(emnist), (1,))
        image, target = emnist[idx[0]]
        if target >= 10:  # Ignore digits
            sequence[i] = image.squeeze()
            i += 1

    # Sample `seq_len` random images from the MNIST dataset
    # indices = torch.randint(0, len(emnist), (seq_len,))
    # for i, idx in enumerate(indices):
    #     img, _ = emnist[idx]
    #     sequence[i] = img.squeeze()

    return sequence

def load_sequence_cifar(seed, seq_len):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the CIFAR10 dataset
    cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 3, 32, 32))

    # Sample `seq_len` random images from the MNIST dataset
    indices = torch.randint(0, len(cifar), (seq_len,))
    for i, idx in enumerate(indices):
        img, _ = cifar[idx]
        sequence[i] = img

    return sequence

def get_seq_mnist(datapath, seq_len, sample_size, batch_size, seed, device):
    """Get batches of sequence mnist
    
    The data should be of shape [sample_size, seq_len, h, w]
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    # test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # each sample is a sequence of randomly sampled mnist digits
    # we could thus sample samplesize x seq_len images
    random.seed(seed)
    train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size * seq_len))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size * seq_len, shuffle=False)

    return train_loader


def get_mnist(datapath, sample_size, sample_size_test, batch_size, seed, device, binary=False, classes=None):
    # classes: a list of specific class to sample from
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, 28, 28
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, 28, 28
    y_test = torch.cat(y_test, dim=0).to(device)

    if binary:
        X[X > 0.5] = 1
        X[X < 0.5] = 0
        X_test[X_test > 0.5] = 1
        X_test[X_test < 0.5] = 0

    print(X.shape)
    return (X, y), (X_test, y_test)


def get_rotating_mnist(datapath, seq_len, sample_size, batch_size, seed, angle):
    """digit: digit used to train the model
    
    test_digit: digit used to test the generalization of the model

    angle: rotating angle at each step
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)

    # randomly sample 
    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # get data from particular classes
    # idx = (train.targets != test_digit).bool()
    # test_idx = (train.targets == test_digit).bool()
    train_data = train.data / 255.
    # test_data = train.data[test_idx] / 255.

    random.seed(seed)
    train_data = train_data[random.sample(range(len(train_data)), sample_size)] # [sample_size, h, w]
    # test_data = test_data[random.sample(range(len(test_data)), test_size)]
    h, w = train_data.shape[-2], train_data.shape[-1]
    # rotate images
    train_sequences = torch.zeros((sample_size, seq_len, h, w))

    for l in range(seq_len):
        train_sequences[:, l] = TF.rotate(train_data, angle * l)

    train_loader = DataLoader(DataWrapper(train_sequences), batch_size=batch_size)
    
    return train_loader

def load_ucf_frames(datapath):
    # load UCF101 movies directly from folder

    # Set desired output tensor dimensions
    num_frames, height, width = 10, 64, 64

    # Define data transformation pipeline to resize and convert images to tensors
    data_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Initialize an empty tensor to store the image tensors
    image_tensors = torch.empty(num_frames, 3, height, width)

    # Loop through the JPEG images in the directory and convert them to PyTorch tensors
    for i in range(num_frames):
        # Set the path to the JPEG image file
        image_file = os.path.join(datapath, f'frame_{i:02d}.jpg')
        
        # Load the JPEG image as a PIL Image object
        image = Image.open(image_file)
        
        # Apply the data transformation pipeline to the image and convert it to a tensor
        image_tensor = data_transforms(image)
        
        # Store the tensor in the output tensor
        image_tensors[i] = image_tensor

    return image_tensors



class MovingMNIST(Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

