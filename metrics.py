import torch
import numpy as np
import train as tr
import util as ut

def get_bubble_accuracy(model, dataloader, device, max_bulles):
    model.eval()  # Met le modèle en mode évaluation
    erreur_bulles = 0  # Initialise l'erreur totale du nombre de bulles à 0
    for IQ, ground_truth in dataloader.dataset:  # Parcourt chaque échantillon dans le dataset
        IQ, ground_truth = IQ.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)  # Transfère les données sur l'appareil spécifié et les convertit en flottant
        out_numBubbles = model(torch.unsqueeze(IQ, 0)) * max_bulles  # Prédit le nombre de bulles et le multiplie par max_bulles
        Nb_ref = int(len(ground_truth[torch.isfinite(ground_truth)])/2)  # Calcule le nombre de bulles réel dans la vérité terrain
        out_numBubbles = torch.squeeze(out_numBubbles).cpu().detach().numpy() if device==torch.device("cuda") else out_numBubbles.detach().numpy()  # Convertit la prédiction en numpy array
        erreur_bulles += abs(np.round(out_numBubbles) - Nb_ref)  # Calcule l'erreur absolue et l'ajoute à l'erreur totale
    erreur_bulles = erreur_bulles / len(dataloader.dataset)  # Normalise l'erreur totale par le nombre d'échantillons
    return erreur_bulles  # Retourne l'erreur moyenne du nombre de bulles


def get_heatmap_accuracy(model, model_map, dataloader, origin, data_size, device):
    model.eval()  # Met le modèle de position en mode évaluation
    model_map.eval()  # Met le modèle de probabilité en mode évaluation
    rmse, erreur_nb_bulles, nbElem = 0, 0, len(dataloader.dataset)  # Initialise les métriques à 0 et obtient le nombre d'éléments dans le dataset
    for IQ, _, ground_truth in dataloader.dataset:  # Parcourt chaque échantillon dans le dataset
        IQ, ground_truth = IQ.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)  # Transfère les données sur l'appareil spécifié et les convertit en flottant
        out_xy = model(torch.unsqueeze(IQ, 0))  # Prédit les positions des bulles
        out_xy = torch.squeeze(out_xy).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(out_xy).detach().numpy()  # Convertit la prédiction en numpy array
        out_probability_img = model_map(torch.unsqueeze(IQ, 0))  # Prédit l'image de probabilité des bulles
        out_probability_img = torch.squeeze(out_probability_img).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(out_probability_img).detach().numpy()  # Convertit la prédiction en numpy array
        ground_truth = torch.squeeze(ground_truth).cpu().detach() if device==torch.device("cuda") else torch.squeeze(ground_truth).detach()  # Convertit la vérité terrain en numpy array
        ground_truth = ut.process_data(ground_truth, origin, data_size).numpy()  # Traite les données de vérité terrain
        coordinates_prediction = ut.heatmap_to_coordinates(out_xy, out_probability_img)  # Convertit la heatmap prédite en coordonnées de bulles
        nb_bubbles_predicted = len(coordinates_prediction)  # Compte le nombre de bulles prédites
        nb_ref = len(ground_truth)  # Compte le nombre de bulles dans la vérité terrain
        erreur_nb_bulles += abs(nb_ref - nb_bubbles_predicted)  # Calcule l'erreur absolue du nombre de bulles et l'ajoute à l'erreur totale
        rmse += ut.compute_rmse_adjusted_matching(coordinates_prediction, ground_truth)  # Calcule le RMSE et l'ajoute au RMSE total
    return rmse / nbElem, erreur_nb_bulles / nbElem  # Retourne le RMSE moyen et l'erreur moyenne du nombre de bulles


def get_position_map_accuracy(model, dataloader, device):
    model.eval()  # Met le modèle en mode évaluation
    jaccard, recall, precision, nbElem = 0, 0, 0, len(dataloader.dataset)  # Initialise les métriques à 0 et obtient le nombre d'éléments dans le dataset
    for IQ, ground_truth in dataloader.dataset:  # Parcourt chaque échantillon dans le dataset
        IQ, ground_truth = IQ.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)  # Transfère les données sur l'appareil spécifié et les convertit en flottant
        out_xy = model(torch.unsqueeze(IQ, 0))  # Prédit les positions des bulles
        out_xy = torch.squeeze(out_xy, 0).cpu().detach().numpy() if device==torch.device("cuda") else torch.squeeze(out_xy, 0).detach().numpy()  # Convertit la prédiction en numpy array
        ground_truth = ground_truth.cpu().detach().numpy() if device==torch.device("cuda") else ground_truth.detach().numpy()  # Convertit la vérité terrain en numpy array
        truth_prediction = out_xy > 0.5  # Binarise la prédiction
        ground_label = ground_truth > 0.5  # Binarise la vérité terrain
        TP = np.sum(truth_prediction*ground_label)  # Calcule les vrais positifs
        FP = np.sum(truth_prediction*~ground_label)  # Calcule les faux positifs
        FN = np.sum(~truth_prediction*ground_label)  # Calcule les faux négatifs
        jaccard += TP/max(TP+FP+FN, 1)  # Calcule le coefficient de Jaccard
        recall += TP/max(TP+FN, 1)  # Calcule le rappel
        precision += TP/max(TP+FP, 1)  # Calcule la précision
    return jaccard / nbElem, recall / nbElem, precision / nbElem  # Retourne le Jaccard moyen, le rappel moyen et la précision moyenne


def get_position_accuracy(model, dataloader, origin, data_size, device):
    model.eval()  # Met le modèle en mode évaluation
    accuracy, nb_ref_total, abs_diff_total = 0, 0, 0  # Initialise les métriques de précision et d'erreur
    
    for IQ, ground_truth in dataloader.dataset:  # Boucle à travers chaque échantillon dans le dataset
        IQ, ground_truth = IQ.to(device=device, dtype=torch.float), ground_truth.to(device=device, dtype=torch.float)  # Transfère les données sur le dispositif (CPU/GPU) et les convertit en float
        out_xy = model(torch.unsqueeze(IQ, 0))  # Passe l'image IQ à travers le modèle pour prédire les positions des bulles
        out_xy = torch.squeeze(out_xy)  # Supprime la dimension supplémentaire ajoutée

        ground_truth = torch.unsqueeze(ground_truth, 0)  # Ajoute une dimension à la vérité terrain
        ground_truth = tr.process_data(ground_truth, out_xy, origin, data_size, device)  # Traite les données de vérité terrain

        # Convertit les prédictions et la vérité terrain en numpy arrays et les transfère sur le CPU si nécessaire
        out_xy = out_xy.cpu().detach().numpy() if device == torch.device("cuda") else out_xy.detach().numpy()
        ground_truth = torch.squeeze(ground_truth).cpu().detach().numpy() if device == torch.device("cuda") else torch.squeeze(ground_truth).detach().numpy()

        # Filtre les coordonnées non valides dans la vérité terrain
        ground_truth = ground_truth[~np.any(ground_truth <= 0, axis=1)]

        # Met à l'échelle les coordonnées de la vérité terrain
        ground_truth[:, 0] *= data_size[1]
        ground_truth[:, 1] *= data_size[0]

        Nb_ref = ground_truth.shape[0]  # Calcule le nombre de bulles dans la vérité terrain

        # Tronque les prédictions pour qu'elles correspondent au nombre de bulles dans la vérité terrain
        out_xy = out_xy[:Nb_ref, :]

        # Met à l'échelle les coordonnées des prédictions
        out_xy[:, 0] *= data_size[1]
        out_xy[:, 1] *= data_size[0]

        # Calcule la différence absolue et la distance euclidienne pour chaque prédiction
        abs_diff = out_xy - ground_truth
        abs_diff = np.sqrt(abs_diff[:, 0]**2 + abs_diff[:, 1]**2)

        abs_diff_total += np.sum(abs_diff)  # Ajoute la somme des différences absolues
        accuracy += np.sum(abs_diff < 0.35)  # Compte les prédictions correctes (erreur < 0.35)
        nb_ref_total += Nb_ref  # Ajoute le nombre de références

    # Calcule les métriques moyennes
    accuracy = accuracy / nb_ref_total  # Exactitude moyenne
    abs_diff_total = abs_diff_total / nb_ref_total  # Différence absolue moyenne

    return accuracy, abs_diff_total  # Retourne l'exactitude et la différence absolue moyennes
