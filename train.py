import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from prodigyopt import Prodigy
import metrics as mt
import util as ut

# DynamicMSELoss: Une fonction de perte personnalisée qui calcule l'erreur
# quadratique moyenne (MSE) uniquement pour les coordonnées de vérité terrain
# non nulles.
class DynamicMSELoss(nn.Module):
    # Initialise la classe.
    def __init__(self):
        super(DynamicMSELoss, self).__init__()

    # Calcule la perte. Elle redimensionne les prédictions et les vérités
    # terrain, masque les entrées nulles et applique la perte MSE.
    def forward(self, pred, ground_truth):
        ground_truth = ground_truth.view(-1, 2)
        pred = pred.view(-1, 2)
        mask = (ground_truth.sum(dim=1) != 0)
        masked_prediction = pred[mask]
        masked_ground_truth = ground_truth[mask]
        mse_loss = nn.MSELoss()(masked_prediction, masked_ground_truth) #MSELoss
        return mse_loss

# La fonction ajuste les formes des tenseurs de positions pour qu'ils aient tous
# la même taille en ajoutant du remplissage (padding).
def adjust_tensor_shapes(pred, pos, device):
    # max_size : Calcule la taille maximale des positions dans le lot.
    max_size = max(max(torch.numel(t)/2 for t in pos), (torch.numel(pred)/len(pos))/2) #len(pos)=batch_size
    # result_list : Liste pour stocker les tenseurs ajustés.
    result_list = []
    # Parcourt chaque tenseur de positions.
    for tensor in pos:
        # padding_size: Calcule la taille de remplissage nécessaire.
        padding_size = int(max_size - torch.numel(tensor)/2)
        # Si le remplissage est nécessaire, concatène les zéros pour obtenir la taille souhaitée.
        if padding_size > 0:
            padded_tensor = torch.cat((tensor, torch.zeros(padding_size, 2).to(device=device, dtype=torch.float)), dim=0)
            # Ajoute le tenseur rempli à la liste.
            result_list.append(padded_tensor)
            del padded_tensor
        else:
            result_list.append(tensor)
    # Empile les tenseurs ajustés en un seul tenseur.
    result_tensor = torch.stack(result_list)
    del padding_size
    del max_size
    del result_list
    # Retourne le tenseur ajusté.
    return result_tensor

# La fonction Traite les positions des bulles pour normaliser et ajuster leurs formes.
def process_data(positions, pred, origin, data_size, device):
    result = []
    # Parcourt chaque position de bulle dans le lot.
    for bubble_positions in positions: #batch_size
        # Filtre les positions valides.
        pos = bubble_positions[torch.isfinite(bubble_positions)]
        # Redimensionne les positions.
        pos = torch.reshape(pos, (-1, 2))
        # Ajuste les positions par rapport à l'origine.
        pos[:, 0] = pos[:, 0] - origin[0]
        pos[:, 1] = pos[:, 1] - origin[2]
        # Filtre les valeurs supérieures aux bordures de l'image.
        pos = pos[~torch.any(pos<0, axis=1)] #enlève les valeurs inférieures à 0
        pos = pos[torch.logical_and(pos[:, 0] <= data_size[1], pos[:, 1] <= data_size[0])] #enlève les valeurs supérieures aux bordures de l'image
        # Normalise les positions.
        pos[:, 0] /= data_size[1] #normalisation
        pos[:, 1] /= data_size[0]
        # Ajoute les positions traitées à la liste result.
        result.append(pos)
    del pos
    # Ajuste les formes des tenseurs et retourne le résultat.
    return adjust_tensor_shapes(pred, result, device)

# La fonction Calcule le nombre de bulles dans la vérité terrain, normalisé par max_bulles.
def get_nbBubbles_groundTruth(positions, max_bulles):
    result = []
    # Parcourt chaque position de bulle.
    for bubble_positions in positions: #batch_size
        # Ajoute le nombre normalisé de bulles à la liste result.
        result.append((len(bubble_positions[torch.isfinite(bubble_positions)])/2)/max_bulles)
    # Retourne le nombre de bulles sous forme de tenseur.
    return torch.reshape(torch.tensor([(i) for i in result]), (-1, 1))

# Fonction d'Entraînement pour le Modèle de Carte de Chaleur
# Entraîne un modèle de carte de chaleur pour prédire les bulles dans les images
def train_heatmap_model(model, model_map, args, train_loader, test_loader, origin, data_size):
    # Initialisation de l'optimiseur Prodigy avec les paramètres du modèle.
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    # Utilisation d'un planificateur de taux d'apprentissage CosineAnnealingLR.
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Listes pour sauvegarder les métriques
    train_rmse_save, test_rmse_save, train_avg_missing_save, test_avg_missing_save, losses, epochs = [], [], [], [], [], []
    # Boucle d'entraînement
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Pour chaque époque, initialisation de train_loss à 0.
        train_loss = 0
        # Pour chaque lot (batch) dans train_loader
        for IQs, ground_truth, _ in tqdm(train_loader):
            # Déplacez IQs et ground_truth vers le dispositif spécifié (args.device).
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            # Mise du modèle en mode entraînement avec model.train().
            model.train()
            # Remise à zéro des gradients de l'optimiseur.
            optimizer.zero_grad()
            # Prédiction avec model(IQs).
            out_xy = model(IQs)
            # Calcul de la perte avec args.loss.
            loss_xy = args.loss(out_xy, ground_truth)
            # Effectuation la rétropropagation avec loss_xy.backward().
            loss_xy.backward()
            # Mise à jour les paramètres du modèle avec optimizer.step().
            train_loss += loss_xy.item()
            # Ajout la perte du lot à train_loss.
            optimizer.step()

        # Calcul des métriques et sauvegarde des modèles:

        # Calcul des métriques d'entraînement (train_rmse, train_avg_missing) avec mt.get_heatmap_accuracy.
        train_rmse, train_avg_missing = mt.get_heatmap_accuracy(model, model_map, train_loader, origin, data_size, args.device)
        train_rmse_save.append(train_rmse)
        train_avg_missing_save.append(train_avg_missing)
        # De même pour les métriques de test (test_rmse, test_avg_missing).
        test_rmse, test_avg_missing = mt.get_heatmap_accuracy(model, model_map, test_loader, origin, data_size, args.device)
        test_rmse_save.append(test_rmse)
        test_avg_missing_save.append(test_avg_missing)
        # Ajout les métriques et la perte moyenne de l'époque aux listes correspondantes.
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        # Mise à jour le taux d'apprentissage avec scheduler.step().
        if scheduler is not None:
            scheduler.step()
        # Sauvegarde du modèle à la fin de chaque époque.
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetHeatmap', 'std': args.std}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train: RMSE: {train_rmse}, Average missing bubbles: {train_avg_missing}")
        print(f"Test: RMSE: {test_rmse}, Average missing bubbles: {test_avg_missing}")

    # Visualisation des résultats:

    ut.plot_loss(epochs, losses, args.pathSave)
    # Traçage des courbes de perte avec ut.plot_loss.
    ut.plot_metrics(epochs, train_rmse_save, test_rmse_save, "RMSE", args.pathSave)
    # Traçage des métriques avec ut.plot_metrics.
    ut.plot_metrics(epochs, train_avg_missing_save, test_avg_missing_save, "Avg Miss", args.pathSave)

# Cette fonction entraîne un modèle pour prédire les cartes de position des bulles.
def train_position_map_model(model, args, train_loader, test_loader):
    # Initialisation: Comme la fonction précédente
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_jaccard_save, train_recall_save, train_precision_save, test_jaccard_save, test_recall_save, test_precision_save, losses, epochs = [], [], [], [], [], [], [], []
    # Boucle d'entraînement: Similaire à la fonction précédente, mais ici on entraîne un modèle de carte de position.
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = 0
        for IQs, ground_truth in tqdm(train_loader):
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()
            out_xy = model(IQs)
            loss_xy = args.loss(out_xy, ground_truth)
            loss_xy.backward()
            train_loss += loss_xy.item()
            optimizer.step()

        # Calcul des métriques et sauvegarde des modèles:

        # Calcul des métriques de Jaccard, de rappel (recall) et de précision pour l'entraînement et le test.
        train_jaccard, train_recall, train_precision = mt.get_position_map_accuracy(model, train_loader, args.device)
        train_jaccard_save.append(train_jaccard)
        train_recall_save.append(train_recall)
        train_precision_save.append(train_precision)
        test_jaccard, test_recall, test_precision = mt.get_position_map_accuracy(model, test_loader, args.device)
        test_jaccard_save.append(test_jaccard)
        test_recall_save.append(test_recall)
        test_precision_save.append(test_precision)
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        if scheduler is not None:
            scheduler.step()
        # Sauvegarde des modèles
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetMap'}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train: Jaccard: {train_jaccard}, Recall: {train_recall}, Precision: {train_precision}")
        print(f"Test: Jaccard: {test_jaccard}, Recall: {test_recall}, Precision: {test_precision}")
    # Traçage les métriques comme précédemment.
    ut.plot_loss(epochs, losses, args.pathSave)
    ut.plot_metrics(epochs, train_jaccard_save, test_jaccard_save, "Jaccard", args.pathSave)
    ut.plot_metrics(epochs, train_recall_save, test_recall_save, "Recall", args.pathSave)
    ut.plot_metrics(epochs, train_precision_save, test_precision_save, "Precision", args.pathSave)

# Cette fonction entraîne un modèle pour prédire les positions précises des bulles.
def train_position_model(model, args, train_loader, test_loader, origin, data_size, max_bulles):
    # Initialisation

    # Initialisation de l'optimiseur Prodigy avec les paramètres du modèle.
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    # Utilisation un planificateur de taux d'apprentissage CosineAnnealingLR.
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Listes pour sauvegarder les métriques
    train_accuracy_save, train_abs_diff_total_save, test_accuracy_save, test_abs_diff_total_save, losses, epochs = [], [], [], [], [], []
    # Boucle d'entraînement:
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Initialisation train_loss à 0.
        train_loss = 0
        # Pour chaque lot (batch) dans train_loader:
        for IQs, ground_truth in tqdm(train_loader):
            # Déplacement de IQs et ground_truth vers le dispositif spécifié (args.device).
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            # Mise du modèle en mode entraînement avec model.train().
            model.train()
            # Remise à zéro des gradients de l'optimiseur.
            optimizer.zero_grad()
            # Effectuation d'une prédiction avec model(IQs).
            out_xy = model(IQs)
            # Ajustement des données de vérité terrain (ground_truth) pour
            # correspondre à la sortie du modèle en utilisant process_data.
            ground_truth_padded = process_data(ground_truth, out_xy, origin, data_size, args.device)
            # Calcul de la perte avec args.loss
            loss_xy = args.loss(out_xy, ground_truth_padded)
            # Effectuation de la rétropropagation avec loss_xy.backward()
            loss_xy.backward()
            # Mise à jour des paramètres du modèle avec optimizer.step()
            train_loss += loss_xy.item()
            # Ajout de la perte du lot à train_loss
            optimizer.step()

        # Calcul des métriques et sauvegarde des modèles:

        # Calcul des métriques d'exactitude (accuracy) et de différence absolue
        # totale (abs_diff_total) pour l'entraînement avec mt.get_position_accuracy
        train_accuracy, train_abs_diff_total = mt.get_position_accuracy(model, train_loader, origin, data_size, args.device)
        train_accuracy_save.append(train_accuracy)
        train_abs_diff_total_save.append(train_abs_diff_total)
        # De même pour les métriques de test.
        test_accuracy, test_abs_diff_total = mt.get_position_accuracy(model, test_loader, origin, data_size, args.device)
        test_accuracy_save.append(test_accuracy)
        test_abs_diff_total_save.append(test_abs_diff_total)
        # Ajout des métriques et de la perte moyenne de l'époque aux listes correspondantes.
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        # Mise à jour du taux d'apprentissage avec scheduler.step()
        if scheduler is not None:
            scheduler.step()
        # Sauvegarde du modèle à la fin de chaque époque.
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetPosition', 'max_bulles': max_bulles}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train: Accuracy: {train_accuracy}, RMSE: {train_abs_diff_total}")
        print(f"Test: Accuracy: {test_accuracy}, RMSE: {test_abs_diff_total}")

    # Visualisation des résultats:

    # Traçage des courbes de perte avec ut.plot_loss
    ut.plot_loss(epochs, losses, args.pathSave)
    # Traçage des métriques avec ut.plot_metrics.
    ut.plot_metrics(epochs, train_accuracy_save, test_accuracy_save, "Accuracy", args.pathSave)
    ut.plot_metrics(epochs, train_abs_diff_total_save, test_abs_diff_total_save, "RMSE", args.pathSave)

# Cette fonction entraîne un modèle pour prédire le nombre de bulles dans une image.
def train_bulle_model(model, args, train_loader, test_loader, max_bulles):
    # Initialisation: Identique aux fonctions précédentes.
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weightDecay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_erreur_bulles_save, test_erreur_bulles_save, losses, epochs = [], [], [], []
    # Boucle d'entraînement:
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Pour chaque époque, initialisation train_loss à 0.
        train_loss = 0
        # Pour chaque lot (batch) dans train_loader:
        for IQs, ground_truth in tqdm(train_loader):
            # Déplacement de IQs et ground_truth vers le dispositif spécifié (args.device).
            IQs, ground_truth = IQs.to(device=args.device, dtype=torch.float), ground_truth.to(device=args.device, dtype=torch.float)
            # Mise du modèle en mode entraînement avec model.train()
            model.train()
            # Remise à zéro des gradients de l'optimiseur
            optimizer.zero_grad()
            # Effectuation d'une prédiction avec model(IQs) pour obtenir le nombre de bulles prédit
            out_numBubbles = model(IQs)
            # Préparation les données de vérité terrain pour le nombre de bulles en utilisant get_nbBubbles_groundTruth
            ground_truth_bubbles = get_nbBubbles_groundTruth(ground_truth, max_bulles).to(device=args.device, dtype=torch.float)
            # Calcul de la perte avec args.loss
            loss_numBubbles = args.loss(out_numBubbles, ground_truth_bubbles)
            # Effectuation de la rétropropagation avec loss_numBubbles.backward()
            loss_numBubbles.backward()
            # Ajout de la perte du lot à train_loss
            train_loss += loss_numBubbles.item()
            # Mise à jour des paramètres du modèle avec optimizer.step()
            optimizer.step()

        # Calcul des métriques et sauvegarde des modèles:

        # Calcul de l'erreur du nombre de bulles (erreur_bulles) pour l'entraînement avec mt.get_bubble_accuracy
        train_erreur_bulles = mt.get_bubble_accuracy(model, train_loader, args.device, max_bulles)
        train_erreur_bulles_save.append(train_erreur_bulles)
        # De même pour les métriques de test
        test_erreur_bulles = mt.get_bubble_accuracy(model, test_loader, args.device, max_bulles)
        test_erreur_bulles_save.append(test_erreur_bulles)
        # Ajout des métriques et de la perte moyenne de l'époque aux listes correspondantes
        epochs.append(epoch)
        losses.append(train_loss/len(train_loader.dataset))
        # Mise à jour du taux d'apprentissage avec scheduler.step()
        if scheduler is not None:
            scheduler.step()
        # Sauvegarde du modèle à la fin de chaque époque
        print("File saved as: ", args.pathSave + '\\epoch_' + str(epoch+1) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'model_name': 'UnetBulle', 'max_bulles': max_bulles}, args.pathSave + '/epoch_' + str(epoch+1) + '.pt')
        print("loss:", train_loss/len(train_loader.dataset))
        print(f"Train BullesErreur: {train_erreur_bulles}")
        print(f"Test BullesErreur: {test_erreur_bulles}")

    # Visualisation des résultats:
    # Traçage des courbes de perte avec ut.plot_loss
    ut.plot_loss(epochs, losses, args.pathSave)
    # Traçage des métriques avec ut.plot_metrics
    ut.plot_metrics(epochs, train_erreur_bulles_save, test_erreur_bulles_save, "BubbleError", args.pathSave)
