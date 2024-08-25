import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
#from simpletracker import simpletracker  # Assurez-vous d'avoir importé ces fonctions correctement
#from munkres import Munkres
#import trackpy as tp
from scipy.optimize import linear_sum_assignment
from scipy.sparse import lil_matrix
#from scipy.sparse.csgraph import hungarian_algorithm
from typing import List, Dict, Tuple, Optional
from tracking import hungarian_linker
from tracking import nearest_neighbor_linker
from tracking import simpletracker

def is_zero(value):
    if np.isscalar(value):
        return value == 0
    #else:
     #   return (value == 0).all()



def ULM_tracking2D(MatTracking, ULM, mode):
    # Définir les paramètres locaux
    max_linking_distance_zero = is_zero(ULM['max_linking_distance'])
    res_zero = is_zero(ULM['res'])
    
    if max_linking_distance_zero or res_zero:
        interp_factor = 0
    else:
        interp_factor = 1 / ULM['max_linking_distance'] / ULM['res'] * .8
    #print('mochkil hna', interp_factor)
    # Assurez-vous que interp_factor est positif et non nul
    #if interp_factor <= 0:
      #  print("Erreur : interp_factor doit être supérieur à zéro.")
      #  return [[0, 0, 0, 0]]

    smooth_factor = 20
    numberOfFrames = ULM['size'][2]

    # Renormaliser pour prendre en compte que toutes les matrices ne commencent pas avec le numéro de frame 1
    minFrame = np.min(MatTracking[:, 3])
    MatTracking[:, 3] = MatTracking[:, 3] - minFrame + 1
    numberOfFrames = int(numberOfFrames)
    index_frames = [np.where(MatTracking[:, 3].astype(int) == i)[0] for i in range(1, numberOfFrames + 1)]
    
    # Impression des index_frames pour le débogage
    #print(f"index_frames: {index_frames}")

    Points = []
    for idx in index_frames:
        if len(idx) > 0:
            Points.append(MatTracking[idx][:, [1, 2]])

    Simple_Tracks, Adjacency_Tracks = simpletracker(
        Points,
        ULM['max_linking_distance'],
        ULM['max_gap_closing']
    )

    n_tracks = len(Simple_Tracks)
    all_points = np.vstack(Points)

    count = 0
    Tracks_raw = []
    for i_track in range(n_tracks):
        track_id = Adjacency_Tracks[i_track]
        idFrame = MatTracking[track_id, 3]
        track_points = np.hstack((all_points[track_id], idFrame.reshape(-1, 1)))
        if len(track_points) > ULM['min_length']:
            Tracks_raw.append(track_points)
            count += 1

    if count == 0:
        print(f"Was not able to find tracks at {minFrame}")
        return [[0, 0, 0, 0]], [[0, 0, 0, 0]] if mode == 'pala' else [[0, 0, 0, 0]]

    # Post-traitement des tracks
    Tracks_out = []
    if mode == 'nointerp':
        for track_points in Tracks_raw:
            xi = track_points[:, 1]
            zi = track_points[:, 0]
            iFrame = track_points[:, 2]
            if len(zi) > ULM['min_length']:
                Tracks_out.append(np.vstack((zi, xi, iFrame)).T)

    elif mode == 'interp':
        for track_points in Tracks_raw:
            xi = track_points[:, 1]
            zi = track_points[:, 0]
            zi_smooth = uniform_filter1d(zi, smooth_factor)
            xi_smooth = uniform_filter1d(xi, smooth_factor)
            num_points = len(zi_smooth)
            if num_points <= 1:
                print("Erreur : pas assez de points pour l'interpolation.")
                continue
            indices = np.arange(0, num_points, interp_factor)
            if len(indices) == 0:
                print("Erreur : les indices d'interpolation sont vides.")
                continue
            zi_interp = interp1d(range(num_points), zi_smooth, kind='linear')(indices)
            xi_interp = interp1d(range(num_points), xi_smooth, kind='linear')(indices)
            if not zi_interp.any() or not xi_interp.any():
                print("Erreur : l'interpolation contient des valeurs invalides.")
                continue
            if len(zi) > ULM['min_length']:
                Tracks_out.append(np.vstack((zi_interp, xi_interp)).T)

    elif mode == 'velocityinterp':
        for track_points in Tracks_raw:
            xi = track_points[:, 1]
            zi = track_points[:, 0]
            TimeAbs = np.arange(len(zi)) * ULM['scale'][2]
            zi_smooth = uniform_filter1d(zi, smooth_factor)
            xi_smooth = uniform_filter1d(xi, smooth_factor)
            num_points = len(zi_smooth)
            if num_points <= 1:
                print("Erreur : pas assez de points pour l'interpolation.")
                continue
            indices = np.arange(0, num_points, interp_factor)
            if len(indices) == 0:
                print("Erreur : les indices d'interpolation sont vides.")
                continue
            zi_interp = interp1d(range(num_points), zi_smooth, kind='linear')(indices)
            xi_interp = interp1d(range(num_points), xi_smooth, kind='linear')(indices)
            if not zi_interp.any() or not xi_interp.any():
                print("Erreur : l'interpolation contient des valeurs invalides.")
                continue
            TimeAbs_interp = interp1d(range(len(TimeAbs)), TimeAbs, kind='linear')(np.arange(0, len(TimeAbs), interp_factor))
            vzi = np.diff(zi_interp) / np.diff(TimeAbs_interp)
            vxi = np.diff(xi_interp) / np.diff(TimeAbs_interp)
            vzi = np.insert(vzi, 0, vzi[0])
            vxi = np.insert(vxi, 0, vxi[0])
            if len(zi) > ULM['min_length']:
                Tracks_out.append(np.vstack((zi_interp, xi_interp, vzi, vxi, TimeAbs_interp)).T)

    elif mode == 'pala':
        Tracks_interp = []
        for track_points in Tracks_raw:
            xi = track_points[:, 1]
            zi = track_points[:, 0]
            iFrame = track_points[:, 2]
            if len(zi) > ULM['min_length']:
                Tracks_out.append(np.vstack((zi, xi, iFrame)).T)
            zi_smooth = uniform_filter1d(zi, smooth_factor)
            xi_smooth = uniform_filter1d(xi, smooth_factor)
            num_points = len(zi_smooth)
            indices = np.arange(0, num_points, 1000)
            zi_interp = interp1d(range(num_points), zi_smooth, kind='linear')(indices)
            xi_interp = interp1d(range(num_points), xi_smooth, kind='linear')(indices)
            dd = np.sqrt(np.diff(xi_interp)**2 + np.diff(zi_interp)**2)
            vmean = np.sum(dd) / (len(zi)) / ULM['scale'][2]
            if len(zi) > ULM['min_length']:
                Tracks_interp.append(np.vstack((zi_interp, xi_interp, vmean * np.ones_like(zi_interp))).T)
        return Tracks_out, Tracks_interp

    Tracks_out = [track for track in Tracks_out if track.shape[0] > 0]
    return Tracks_out





# t9dar

