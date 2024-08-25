import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage import filters
import torchvision.transforms as T


# Cette fonction ajoute des bords (padding) autour d'une image pour s'assurer que
# ses dimensions sont multiples de stride. Elle retourne l'image paddée et les
# valeurs de padding utilisées.
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding par défaut, voir https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

# Cette fonction enlève le padding ajouté par pad_to.
# Elle utilise les valeurs de padding pour recadrer l'image à sa taille d'origine.
def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

# Ces fonctions plot_loss et plot_metrics créent et sauvegardent des graphiques pour
# visualiser la perte (loss) et d'autres métriques au cours des epochs d'entraînement.
def plot_loss(epochs, losses, pathSave):
    plt.title("Loss")
    plt.plot(epochs, losses, label="Loss par epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(pathSave + '/loss.png')
    plt.clf()

def plot_metrics(epochs, train, test, title, pathSave):
    plt.title(title)
    plt.plot(epochs, train, label="Train")
    plt.plot(epochs, test, label="Test")
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend(loc='best')
    plt.savefig(pathSave + '/' + title + '.png')
    plt.clf()

# Cette fonction crée des masques binaires à partir des coordonnées des points.
# Elle ajuste les coordonnées en fonction de l'origine et de la taille des données,
# puis marque les positions valides dans le masque.
def coordinates_to_mask(coordinates, shape, origin, data_size):
    Y_mask = torch.zeros(shape)
    for i, coordonnees in enumerate(coordinates):
        coordonnees = coordonnees[torch.isfinite(coordonnees)]
        coordonnees = torch.reshape(coordonnees, (-1, 2))
        coordonnees[:, 0] = coordonnees[:, 0] - origin[0]
        coordonnees[:, 1] = coordonnees[:, 1] - origin[2]
        coordonnees = coordonnees[~torch.any(coordonnees<0, axis=1)]
        coordonnees = coordonnees[torch.logical_and(coordonnees[:, 0] < data_size[1]-1, coordonnees[:, 1] < data_size[0]-1)]
        for xy in coordonnees:
            xy = torch.round(xy).to(torch.int)
            Y_mask[i, xy[1], xy[0]] = 1
    return Y_mask

# Cette fonction génère des heatmaps à partir des coordonnées des points.
# Elle crée une heatmap gaussienne pour chaque point et les normalise.
def coordinates_to_heatmap(shape, data_size, origin, coordinates, std=1):
    Y_heatmap = torch.zeros(shape)
    for i, coords in enumerate(coordinates):
        coords = process_data(coords, origin, data_size)
        coords = torch.flip(coords, (1,))
        heatmap = torch.zeros((data_size[0], data_size[1]), dtype=torch.float32)
        x = torch.arange(data_size[1]).float()
        y = torch.arange(data_size[0]).float()
        xx, yy = torch.meshgrid(y, x, indexing='ij')
        for x, y in coords:
            gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * std**2))
            heatmap += gaussian
        heatmap = heatmap / heatmap.max() #normalisation de la heatmap
        Y_heatmap[i] = heatmap
    return Y_heatmap

# Cette fonction utilise le seuil d'Otsu pour détecter des bulles dans une
# image et retourne les coordonnées des pixels qui sont au-dessus du seuil.
def find_bubbles(image):
    thresh = filters.threshold_otsu(image)
    mask = image > thresh
    non_black_indices = np.where(mask > 0)
    coordinates = np.column_stack((non_black_indices[1], non_black_indices[0]))
    return coordinates[:, ::-1]

# Cette fonction convertit une heatmap en coordonnées en utilisant une
# transformation de distance, des marqueurs, et l'algorithme de segmentation
# watershed pour trouver les centres des bulles.
def heatmap_to_coordinates(heatmap, probabiliy_img):
    thresh = filters.threshold_otsu(heatmap)
    distance = ndi.distance_transform_edt(heatmap >= thresh)
    coords = find_bubbles(probabiliy_img)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=heatmap >= thresh)
    centers = []
    for label in np.unique(labels):
        if label == 0:  #ignore l'arrière-plan
            continue
        region = labels == label
        if heatmap[region].sum() > 0: #la région contient des pixels non nuls
            center = ndi.center_of_mass(heatmap, labels, label)
            centers.append(center)
    return np.array(centers)[:, ::-1] if centers else np.array([])

# Cette fonction calcule la matrice de coût basée sur la distance
# euclidienne entre les prédictions et les vérités terrain.
def compute_cost_matrix(list_pred, list_gt):
    max_size = max(len(list_pred), len(list_gt))
    penality = 1e9
    cost_matrix = np.full((max_size, max_size), penality)
    for i, pred in enumerate(list_pred):
        for j, vt in enumerate(list_gt):
            cost_matrix[i][j] = np.sqrt((pred[0] - vt[0])**2 + (pred[1] - vt[1])**2)
    return cost_matrix

# Cette fonction calcule l'erreur quadratique moyenne ajustée en utilisant
# l'algorithme d'affectation linéaire pour appairer les prédictions et les vérités terrain.
def compute_rmse_adjusted_matching(list_pred, list_gt):
    cost_matrix = compute_cost_matrix(list_pred, list_gt)
    row, col = linear_sum_assignment(cost_matrix)
    total_mse = sum(cost_matrix[i, j]**2 for i, j in zip(row, col) if cost_matrix[i, j] != 1e9)
    nb_valid_matches = sum(1 for i, j in zip(row, col) if cost_matrix[i, j] != 1e9) #exclure les penalites
    rmse = np.sqrt(total_mse / nb_valid_matches) if nb_valid_matches > 0 else 0
    return rmse

# Cette fonction prétraite les positions.
def process_data(positions, origin, data_size):
    positions = positions[torch.isfinite(positions)]
    positions = torch.reshape(positions, (-1, 2))
    positions[:, 0] = positions[:, 0] - origin[0]
    positions[:, 1] = positions[:, 1] - origin[2]
    positions = positions[~torch.any(positions<0, axis=1)] #enlève les valeurs inférieures à 0
    positions = positions[torch.logical_and(positions[:, 0] <= data_size[1], positions[:, 1] <= data_size[0])] #enlève les valeurs supérieures aux bordures de l'image
    return positions
