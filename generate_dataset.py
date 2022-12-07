import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
# from pyrfm.random_feature.random_fourier import RandomFourier
from sklearn.kernel_approximation import RBFSampler
import random

# see old for other dataset generators



def load_data(dataset='EMNIST', n=2**13, d=2**8, gamma=0.002,kappa=1e2,high_coherence=False):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'EMNIST':
        trainset = torchvision.datasets.EMNIST(root='./data',split='byclass',train=True, download=True, transform=transform)
    else:
        A = np.random.randn(n,d)
        u, sigma, vh = np.linalg.svd(A, full_matrices=False)
        sigma = np.linspace(1,kappa,d)
        if dataset.startswith("High Coherence"):
            z = 1./np.sqrt(np.random.gamma(2,0.5,n))
            u = u * z[:,None]
        A = u @ (np.diag(sigma) @ vh)
        x = 1./np.sqrt(d)*np.random.randn(d,1)
        if dataset.endswith("Least Squares"):
            # Least squares version
            b = A @ x + 0.1*np.random.randn(n,1)
        else:
            # logistic regression version
            b = np.sign(A @ x)
        
        A = torch.tensor(A)
        b = torch.tensor(b)
        print(dataset + " with dimensions " + str(A.shape) + " and condition number: " + str(kappa))
        return A, b

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=n, shuffle=True, num_workers=2)
    
    A_, b_ = iter(trainloader).next()
    A_ = torch.tensor(A_.reshape((A_.shape[0], -1)), dtype=torch.float64)
    print(dataset + " before transform: " + str(A_.shape))
    A = RBFSampler(gamma=gamma,n_components=d).fit_transform(A_.numpy())
    A = torch.tensor(A)
    # label is -1 for even-numbered classes, else 1?
    b = torch.tensor([-1 if b_[ii] % 2 == 0 else 1 for ii in range(len(b_))], dtype=torch.float64).reshape((-1,1))
    
    print(dataset + " after transform: " + str(A.shape))
    del A_, b_
    return A, b

        
    
    
    
    
    
    
    
    
    
    
    
    
