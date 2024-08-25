import scipy.io
from os.path import join
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.nn.functional import normalize
import os
import torch
import util as ut

import matplotlib.pyplot as plt

# Ces deux classes définissent des datasets personnalisés pour PyTorch.
# Dataset est pour les données générales (entrée X et labels y).
# HeatmapDataset est pour des données qui incluent également des coordonnées z.
class Dataset(torch.utils.data.Dataset):
  def __init__(self, x_data, y_labels):
        self.y = y_labels
        self.x = x_data

  def __len__(self):
        return len(self.x)

  def __getitem__(self, index):
        X = self.x[index]
        y = self.y[index]
        return X, y

class HeatmapDataset(torch.utils.data.Dataset):
  def __init__(self, x_data, y_labels, z_coordinates):
        self.y = y_labels
        self.x = x_data
        self.z = z_coordinates

  def __len__(self):
        return len(self.x)

  def __getitem__(self, index):
        X = self.x[index]
        y = self.y[index]
        z = self.z[index]
        return X, y, z

# Cette fonction ajoute du bruit à l'image IQ en utilisant un filtre gaussien
# et d'autres paramètres tels que la puissance, l'impédance, et les niveaux de bruit.
def PALA_AddNoiseInIQ(IQ, power, impedance, sigmaGauss, clutterdB, amplCullerdB):
    max_IQ = np.max(IQ)
    return IQ + ndimage.gaussian_filter(max_IQ * 10**(clutterdB / 20) + np.reshape(np.random.normal(size=np.prod(IQ.shape), scale=np.abs(power * impedance)), IQ.shape) * max_IQ * 10**((amplCullerdB + clutterdB) / 20), sigma=sigmaGauss)

# Cette fonction charge et prétraite les données de la séquence "PALA_InSilicoFlow_sequence.mat"
# et les fichiers IQ. Elle ajoute du bruit aux images IQ et organise les positions xy des points.
# Elle retourne les données normalisées, les positions xy, l'origine, la taille des données et le
# nombre maximum de bulles.
def load_silicoFlow_data(args):
    transform = transforms.ToTensor()
    sequence = scipy.io.loadmat(join(args.pathData,"PALA_InSilicoFlow_sequence.mat"))
    Origin = sequence["PData"]["Origin"].flatten()[0][0]
    data_size = sequence["PData"]["Size"].flatten()[0][0]
    NoiseParam = {}
    NoiseParam["power"]        = -2;   # [dBW]
    NoiseParam["impedance"]    = .2;   # [ohms]
    NoiseParam["sigmaGauss"]   = 1.5;  # Gaussian filtering
    NoiseParam["clutterdB"]    = -20;  # Clutter level in dB (will be changed later)
    NoiseParam["amplCullerdB"] = 10;   # dB amplitude of clutter
    IQs, xy_pos, max_bulles = None, None, None
    for file in os.listdir(join(args.pathData,"IQ")):
        temp = scipy.io.loadmat(join(args.pathData,"IQ",file))
        if max_bulles is None:
            max_bulles = temp["ListPos"].shape[0]
        listpos = temp["ListPos"][:,[0, 2],:]
        indices_listpos = np.argsort(listpos[:,0,:], axis=0) #trier en fonction de la coordonnée y
        sorted_listpos = np.take_along_axis(listpos, indices_listpos[:, None, :], axis=0)
        xy_pos = torch.cat((xy_pos, transform(sorted_listpos)), dim=0) if xy_pos is not None else transform(sorted_listpos)
        IQs = torch.cat((IQs, transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))), dim=0) if IQs is not None else transform(PALA_AddNoiseInIQ(np.abs(temp["IQ"]), **NoiseParam))
    return normalize(IQs), xy_pos, Origin, data_size, max_bulles

# Cette fonction charge les datasets d'entraînement et de test.
# Elle gère trois types de formation :
#    trainType == 2 : Convertit les coordonnées en masque.
#    trainType == 3 : Convertit les coordonnées en heatmap et utilise un HeatmapDataset.
#    Autres types : Utilise un dataset général.
# Elle crée des loaders pour PyTorch avec les datasets appropriés
# et les paramètres spécifiés (nombre de travailleurs, mémoire).
def load_dataset(args):
    X, Y, origin, data_size, max_bulles = load_silicoFlow_data(args)
    if args.trainType == 2:
        Y = ut.coordinates_to_mask(Y, X.shape, origin, data_size)
    if args.trainType == 3:
        Z = Y.clone()
        Y = ut.coordinates_to_heatmap(X.shape, data_size, origin, Y, args.std)
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size=args.testSize, shuffle=args.shuffle)
        dataset_train = HeatmapDataset(torch.unsqueeze(x_train, 1), torch.unsqueeze(y_train, 1), torch.unsqueeze(z_train, 1))
        dataset_test = HeatmapDataset(torch.unsqueeze(x_test, 1), torch.unsqueeze(y_test, 1), torch.unsqueeze(z_test, 1))
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.testSize, shuffle=args.shuffle)
        dataset_train = Dataset(torch.unsqueeze(x_train, 1), torch.unsqueeze(y_train, 1))
        dataset_test = Dataset(torch.unsqueeze(x_test, 1), torch.unsqueeze(y_test, 1))

    kwargs = {'num_workers': args.numWorkers, 'pin_memory': True} if args.device=='cuda' else {}
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchSize, shuffle=args.shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchSize, shuffle=args.shuffle, **kwargs)
    return train_loader, test_loader, origin, data_size, max_bulles
