import os
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt
from load_data import PALA_AddNoiseInIQ 
from concurrent.futures import ProcessPoolExecutor
from PALA_multiULM import PALA_multiULM
from ULM_LOCALIZATION2D import ULM_localization2D
from matplotlib import cm
from ULM_Track2MatOut import ULM_Track2MatOut
#from matplotlib.tight_layout import tight_layout
from matplotlib.axes import Axes

# Paramètres de configuration
PALA_data_folder = '/content/drive/MyDrive/Data'  # Modifie selon le chemin correct

# Sélectionner les dossiers IQ et Media
print('Running PALA_SilicoFlow.py')
t_start = time.time()

# Chemin de travail
workingdir = os.path.join(PALA_data_folder, 'PALA_data_InSilicoFlow')
print('Chemin de répertoire de travail:', workingdir)

# Vérifie si le chemin de répertoire est valide
if os.path.isdir(workingdir):
    os.chdir(workingdir)
else:
    raise NotADirectoryError(f'{workingdir} n\'est pas un répertoire valide.')

filename = 'PALA_InSilicoFlow'

myfilepath = os.path.join(workingdir, filename)
myfilepath_data = os.path.join(workingdir, 'IQ', filename)

# Charger les variables à partir du fichier .mat mochkil
mat_seq_file = myfilepath + '_sequence.mat'
print('Chemin du fichier sequence.mat:', mat_seq_file)

if os.path.isfile(mat_seq_file):
    mat_contents = sio.loadmat(mat_seq_file)
    P = mat_contents['P']
    PData = mat_contents['PData']
    Trans = mat_contents['Trans']
    Media = mat_contents['Media']
    UF = mat_contents['UF']
    Resource = mat_contents['Resource']
    Receive = mat_contents['Receive']
    filetitle = mat_contents['filetitle']
else:
    raise FileNotFoundError(f'Le fichier {mat_seq_file} est introuvable.')

# Chemin complet pour le fichier de configuration
config_file_path = os.path.join(PALA_data_folder, 'PALA_InSilicoFlow_v3_config.mat')
print('Chemin du fichier de configuration:', config_file_path)

if os.path.isfile(config_file_path):
    config_file = sio.loadmat(config_file_path)
    MatOut_target = config_file['MatOut']
    try:
        nb_points_max = config_file['MyMedia']['nbPointsMax'][0, 0]  # Extraire la valeur
    except KeyError:
        raise KeyError("La clé 'nbPointsMax' est manquante dans 'MyMedia'.")
    except Exception as e:
        print(f'Erreur lors de l\'extraction de nbPointsMax: {e}')
else:
    raise FileNotFoundError(f'Le fichier {config_file_path} est introuvable.')

# Créer les répertoires pour les résultats
myfilepath_res = os.path.join(workingdir, 'Results')
os.makedirs(myfilepath_res, exist_ok=True)
myfilepath_res = os.path.join(myfilepath_res, filename)

# Définir les paramètres ULM
NFrames = P['BlocSize'][0, 0] * P['numBloc'][0, 0]
framerate = P['FrameRate'][0, 0]
res = 10

ULM = {
    'numberOfParticles': 40,
    'res': 10,
    'max_linking_distance': 2,
    'min_length': 15,
    'fwhm': [3, 3],
    'max_gap_closing': 0,
    'size': [PData['Size'][0, 0][0, 0], PData['Size'][0, 0][0, 1], NFrames],
    'scale': [1, 1, 1 / framerate],
    'numberOfFramesProcessed': NFrames,
    'interp_factor': 1 / res
}
# PData
listAlgo = ['no_shift', 'wa', 'interp_cubic', 'interp_lanczos', 'interp_spline', 'gaussian_fit', 'radial']
Nalgo = len(listAlgo)

# Charger les données IQ depuis le fichier .mat
iq_file_path = os.path.join(workingdir, 'IQ', 'PALA_InSilicoFlow_IQ010.mat')
if os.path.isfile(iq_file_path):
    temp = sio.loadmat(iq_file_path)
    IQ = temp.get('IQ')  # Utilisation de .get() pour éviter une KeyError
    if IQ is None:
        raise ValueError('La clé "IQ" n\'existe pas dans le fichier.')
else:
    raise FileNotFoundError(f'Le fichier {iq_file_path} est introuvable.')

# Paramètres de bruit (comme définis précédemment)
NoiseParam = {
    'power': -2,  # dBW
    'impedance': 0.2,  # ohms
    'sigmaGauss': 1.5,  # Filtrage gaussien
    'clutterdB': -20,  # Niveau de clutter en dB
    'amplCullerdB': 10  # Amplitude du clutter en dB
}

# Ajouter du bruit aux données IQ en utilisant la fonction de load_data.py
noisy_IQ = PALA_AddNoiseInIQ(IQ, **NoiseParam)

# Statistiques des données IQ avant ajout de bruit
print("Avant ajout de bruit:")
print(f"Min: {np.min(IQ)}, Max: {np.max(IQ)}, Mean: {np.mean(IQ)}, Std: {np.std(IQ)}")

# Statistiques des données IQ après ajout de bruit
print("Après ajout de bruit:")
print(f"Min: {np.min(noisy_IQ)}, Max: {np.max(noisy_IQ)}, Mean: {np.mean(noisy_IQ)}, Std: {np.std(noisy_IQ)}")

# Calculer dB 60
dB = 20 * np.log10(np.abs(noisy_IQ))
print("Avant normalisation:")
print(f"Min: {np.min(dB)}, Max: {np.max(dB)}, Mean: {np.mean(dB)}, Std: {np.std(dB)}")

dB -= np.max(dB)  # Normalisation

print("Après normalisation:")
print(f"Min: {np.min(dB)}, Max: 0.0, Mean: {np.mean(dB)}, Std: {np.std(dB)}")

# Afficher un histogramme des valeurs de dB
plt.hist(dB.ravel(), bins=100)
plt.title("Histogramme des valeurs de dB")
plt.xlabel("Valeurs de dB")
plt.ylabel("Fréquence")
plt.show()

# Afficher les données
vmin, vmax = np.min(dB), np.max(dB)
plt.ion()  # Mode interactif pour la mise à jour en temps réel
fig, ax = plt.subplots()
cax = ax.imshow(dB[:, :, 0], cmap='gray', vmin=vmin, vmax=vmax)
fig.colorbar(cax)

for ii in range(min(5, dB.shape[2])):
    cax.set_data(dB[:, :, ii])
    ax.set_title(f'Frame {ii}')
    plt.draw()
    plt.pause(0.01)  # Pause pour l'animation

plt.ioff()  # Désactiver le mode interactif
plt.show()

# Enregistrer quelques images pour les examiner
for ii in range(min(5, dB.shape[2])):
    plt.imshow(dB[:, :, ii], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f'Frame {ii}')
    plt.colorbar()
    plt.savefig(f'/content/drive/MyDrive/fframes/frame_{ii:03d}.png')
    plt.show()

# Afficher un seul cadre pour vérification mochkil
plt.imshow(dB[:, :, 0], cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title("Affichage d'un seul cadre de dB")
plt.show()


print( 'Pdata est :' ,PData )

###########################################################################

ClutterList = [-60 -40 -30 -25 -20 -15 -10]
Nrepeat = 1


# %% %% Load data and perform detection/localization and tracking algorithms %%%
print('--- DETECTION AND LOCALIZATION --- \n\n')

Track_tot = []
Track_tot_interp = []
IQ_speckle = None
IQ = None
temp = None
dB = None

t1 = time.time()
for iclutter in range(len(ClutterList)):
    print(f'Computation: SNR at {abs(ClutterList[iclutter])}dB')
    NoiseParam['clutterdB'] = ClutterList[iclutter]
    
    # Extract the integer value from P['Nrepeat']
    #Nrepeat = int(P['Nrepeat'].flatten()[0])
    
    for hhh in range(1, Nrepeat + 1):
        print(f'Bloc {hhh}/{Nrepeat}')
        temp = sio.loadmat(f'{myfilepath_data}_IQ{hhh:03d}.mat', variable_names=['IQ', 'Media', 'ListPos'])
        Media = temp['Media']
        ListPos = temp['ListPos']
        IQ = PALA_AddNoiseInIQ(
            np.abs(temp['IQ']),
            NoiseParam['power'],         # Assurez-vous que 'power' est dans NoiseParam
            NoiseParam['impedance'],
            NoiseParam['sigmaGauss'],
            NoiseParam['clutterdB'],
            NoiseParam['amplCullerdB']
        )
        
        # Save IQ with noise (uncomment if needed)
        # sio.savemat(f'{myfilepath_res}_IQnoise_{abs(ClutterList[iclutter])}dB.mat', {'IQ': IQ, 'ListPos': ListPos, 'Media': Media})
        print(f"PData['PDelta'] contents: {PData['PDelta']}")
        print(f"PData['PDelta'] length: {len(PData['PDelta'])}")
        args = ['tracking', 1, 'savingfilename', 'output_file.mat']
        track_tot, track_tot_interp, varargout = PALA_multiULM(IQ, listAlgo, ULM, PData, *args)
        Track_tot.append(track_tot)
        Track_tot_interp.append(track_tot_interp)
    
    sio.savemat(f'{myfilepath_res}_Tracks_multi_{abs(ClutterList[iclutter])}dB.mat', {
        'Track_tot': Track_tot,
        'Track_tot_interp': Track_tot_interp,
        'ULM': ULM,
        'P': P,
        'listAlgo': listAlgo,
        'Nalgo': Nalgo,
        'ClutterList': ClutterList,
        'NoiseParam': NoiseParam
    })
    print('Saved')

t2 = time.time()
elapsed_time = t2 - t1
print(f'Detection and localization done in {elapsed_time // 3600} hours {elapsed_time % 3600 / 60:.1f} minutes.')

############################################################################

print('--- CREATING MATOUTS --- \n\n')
for iclutter in range(len(ClutterList)):
    # Load tracks and generate MatOut for each Clutter levels
    print(f'MatOut: SNR at {abs(ClutterList[iclutter])}dB')
    
    # Chargement des données
    data = sio.loadmat(f'{myfilepath_res}_Tracks_multi_{abs(ClutterList[iclutter])}dB.mat')
    Track_tot = data['Track_tot']
    Track_tot_interp = data['Track_tot_interp']
    ULM = data['ULM']
    P = data['P']
    listAlgo = data['listAlgo']
    
    MatOut = []
    MatOut_vel = []

    for ialgo in range(Nalgo):
        Track_tot_matout = np.concatenate(Track_tot_interp[:, ialgo], axis=0)
        aa = -PData['Origin'][0, 0][0, [2, 0]] + np.array([1, 1])
        # Extraction des données
        #pdelta = PData[0][2]  # Accède à l'élément PDelta (index 2 en Python)
        # Utilisation des indices [2, 0] pour accéder aux éléments
        #bb = 1. / pdelta[[2, 0]] * ULM['res']
        bb = 1. / PData['PDelta'][0, 0][0,[2, 0]] * ULM['res']  # fix the size of pixel
        aa = np.append(aa, 1)
        bb = np.append(bb, 2)  # for velocity
        
        Track_matout = [((x[:, [0, 1, 2]] + aa) * bb) for x in Track_tot_matout]
        #PData['Size'][0, 0][0, 0], PData['Size'][0, 0][0, 1]
        #matlab : PData.Size(1),PData.Size(2)
        #ULM_Track2MatOut(Track_matout,ULM.res*[PData(1).Size(1) PData(1).Size(2)]+[1 1]
        # Supposons que ULM est comme suit :
        #print(aa,bb,"######################")
        #print("hadi",ULM['res'] * [PData['Size'][0, 0][0, 1], PData['Size'][0, 0][0, 2]] + np.array([1, 1]))
        #print(ULM['res'] * [PData['Size'][0, 0][0, 1], PData['Size'][0, 0][1, 2]] + np.array([1, 1]))
        #print(ULM['res'] * [PData['Size'][0, 0][0, 1], PData['Size'][0, 1][0, 2]] + np.array([1, 1]))
        #print(ULM['res'] * [PData['Size'][0, 0][0, 1],PData['Size'][0, 0][0, 1] + PData['Size'][0, 0][0, 1] + PData['Size'][0, 0][0, 2]] + np.array([1, 1]))
        #test =  PData['Size'][0, 0][0, 1] + PData['Size'][0, 0][0, 1] + PData['Size'][0, 0][0, 2]
        #test = test + test + test + test + test + test + PData['Size'][0, 0][0, 1]
        
        MatOut_ialgo, MatOut_vel_ialgo = ULM_Track2MatOut(Track_matout,ULM['res'] * [PData['Size'][0, 0][0, 1], PData['Size'][0, 0][0, 2]] * np.array([1, 200]) , mode= '2D_velmean')
        # Vérifie les résultats 
        print(f'MatOut_ialgo: {MatOut_ialgo}')  # Vérifie le contenu de MatOut_ialgo
        print(f'MatOut_vel_ialgo: {MatOut_vel_ialgo}')  # Vérifie le contenu de MatOut_vel_ialgo
        
        MatOut.append(MatOut_ialgo)
        MatOut_vel.append(MatOut_vel_ialgo)
        
    MatOut.append(MatOut_target) 
    
    # Display MatOut round Tracks MatOut_ialgo
    plt.figure(figsize=(15.5, 5.4))
    axes = plt.subplots(2, 4, figsize=(15.5, 5.4), tight_layout=True)[1].flatten()
    dossier_sauvegarde = '/content/drive/MyDrive/testdeux'
    for ii in range(len(listAlgo)):
        ax = axes[ii]
        cax = ax.imshow(np.sqrt(MatOut[ii]), cmap='hot')
        ax.axis('off')
        ax.set_title(listAlgo[ii] if ii < len(MatOut) - 1 else 'Target')
        CountMatOut = np.sum(MatOut[ii])
        # Créez le chemin complet du fichier de sauvegarde
        chemin_fichier = f'{dossier_sauvegarde}/image_{ii}.png'
    
        # Sauvegardez l'image
        plt.savefig(chemin_fichier, bbox_inches='tight', pad_inches=0) 
    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
    plt.show()
    #/content/drive/MyDrive/testdeux


    #plt.savefig(f'/content/drive/MyDrive/fframes/frame_{ii:03d}.png')
    # Save data
    sio.savemat(f'{myfilepath_res}_MatOut_multi_{abs(ClutterList[iclutter])}dB.mat', {
        'MatOut': MatOut,
        'MatOut_vel': MatOut_vel,
        'ULM': ULM,
        'P': P,
        'PData': PData,
        'listAlgo': listAlgo,
        'Nalgo': Nalgo,
        'CountMatOut': CountMatOut,
        'ClutterList': ClutterList
    })



# ghi bach t3raf hhhh localizeRadialSymmetry