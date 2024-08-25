import torch
from dataclasses import dataclass
from load_data import load_dataset
from networks.UnetPosition import UnetPosition
from networks.UnetBulle import UnetBulle
from networks.UnetMap import UnetMap
import train as tr
import torch.nn as nn
from tkinter import Tk
from tkinter import filedialog

@dataclass
class Args():
    pathData: str #chemin vers les données
    pathSave: str #dossier dans lequel sauvegarder les epochs
    device: torch.device 
    testSize: float #pourcentage du dataset attribué au test [0;1]
    batchSize: int #nombre de mini batchs utilisés pendant l'entraînement
    numWorkers: int #nombre de threads pour les dataloaders
    shuffle: bool #mélanger les données
    weightDecay: float
    epochs: int
    trainType: int #0 = position, 1 = nbBulles, #2 = positionMap, #3 = localisation par centre de masse
    loss: torch.nn.Module #DynamicMSELoss() ou torch.nn.MSELoss()
    std: float #écart type de la heatmap (seulement pour trainType = 3), valeur préférentiellement autour de 1

args = Args(pathData = "D:\\ChefOeuvre\\data\\added_noise",
            pathSave = "D:\\ChefOeuvre\\save",
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            testSize = 0.15,
            batchSize = 16,
            numWorkers = 1,
            shuffle = True,
            weightDecay = 0.01,
            epochs = 30,
            trainType = 3,
            loss = nn.MSELoss(),
            std = 0.6)

#ClutterList = [-60 -40 -30 -25 -20 -15 -10]

train_loader, test_loader, origin, data_size, max_bulles = load_dataset(args)

if args.trainType == 0:
    model = UnetPosition(max_bulles)
    model = model.to(args.device)
    tr.train_position_model(model, args, train_loader, test_loader, origin, data_size, max_bulles)
elif args.trainType == 1:
    model = UnetBulle()
    model = model.to(args.device)
    tr.train_bulle_model(model, args, train_loader, test_loader, max_bulles)
elif args.trainType == 2:
    model = UnetMap()
    model = model.to(args.device)
    tr.train_position_map_model(model, args, train_loader, test_loader)
else:
    Tk().withdraw()
    print("Veuillez choisir un modele pour effectuer la prédiction des probabilités (utilisé pour mesurer la RMSE):")
    filename_map = filedialog.askopenfilename()
    print("Modele choisi: ", filename_map)
    checkpoint_map = torch.load(filename_map, map_location=args.device)
    if checkpoint_map['model_name'] == 'UnetMap':
        model_map = UnetMap()
        model_map.load_state_dict(checkpoint_map['model_state_dict'])
        model_map.to(args.device)
    else:
        raise Exception("Veuillez choisir un modèle Map pour effectuer la prédiction des coordonnées")
    model = UnetMap()
    model = model.to(args.device)
    tr.train_heatmap_model(model, model_map, args, train_loader, test_loader, origin, data_size)
         