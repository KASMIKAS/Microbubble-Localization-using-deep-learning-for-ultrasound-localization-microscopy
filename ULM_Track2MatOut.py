import numpy as np

def ULM_Track2MatOut(Tracks, sizeOut, **kwargs):
    # Convertir sizeOut en tuple d'entiers
    if isinstance(sizeOut, list):
        sizeOut = tuple(map(int, sizeOut))
    elif isinstance(sizeOut, np.ndarray):
        sizeOut = tuple(map(int, sizeOut.flatten()))
    elif isinstance(sizeOut, tuple):
        sizeOut = tuple(map(int, sizeOut))
    else:
        raise ValueError("sizeOut must be a list, numpy array, or tuple of integers.")
    
    # Vérifier que sizeOut contient des entiers positifs
    if any(not isinstance(i, int) or i <= 0 for i in sizeOut):
        raise ValueError("All elements of sizeOut must be positive integers.")
    
    # Tracks
    print(f'sizeOut: {sizeOut}')  # Pour déboguer
    
    # Sélectionner le mode
    mode = kwargs.get('mode', '2D_tracks' if isinstance(Tracks, list) else '2D_allin')
    
    # Initialiser MatOut
    MatOut = np.zeros(sizeOut, dtype=float)  # Initialisation de l'image de densité
    MatOut_vel = np.zeros(sizeOut, dtype=float)  # Initialiser MatOut_vel même si ce n'est pas utilisé

    if mode == '2D_allin':
        # 2D case - 1 cell
        print(f'Building 2D ULM image (size : {sizeOut}).')
        for track in Tracks:
            track = np.asarray(track, dtype=float)
            pos_z = track[:, 0]
            pos_x = track[:, 1]
            pos_z_round = np.round(pos_z).astype(int)
            pos_x_round = np.round(pos_x).astype(int)

            # Debugging statements
            print("Pos z (arrondis) :", pos_z_round)
            print("Pos x (arrondis) :", pos_x_round)

            # Ensure indices are within valid range
            pos_z_round = np.clip(pos_z_round, 0, sizeOut[0] - 1)
            pos_x_round = np.clip(pos_x_round, 0, sizeOut[1] - 1)

            # Remove out of grid bubbles
            outP = (pos_z < 0) | (pos_z >= sizeOut[0]) | (pos_x < 0) | (pos_x >= sizeOut[1])
            valid_indices = ~outP

            pos_z_round = pos_z_round[valid_indices]
            pos_x_round = pos_x_round[valid_indices]

            # Increment the intensity count of the pixel crossed by a track by +1
            for z, x in zip(pos_z_round, pos_x_round):
                MatOut[z, x] += 1

    elif mode == '2D_tracks':
        # 2D case - 1 cell per track
        print(f'Building 2D ULM image (size : {sizeOut}).')
        for itrack in range(len(Tracks)):
            pos_z = np.asarray(Tracks[itrack][:, 0], dtype=float)
            pos_x = np.asarray(Tracks[itrack][:, 1], dtype=float)
            pos_z_round = np.round(pos_z).astype(int)
            pos_x_round = np.round(pos_x).astype(int)

            # Debugging statements
            print("Pos z (arrondis) :", pos_z_round)
            print("Pos x (arrondis) :", pos_x_round)

            # Ensure indices are within valid range
            pos_z_round = np.clip(pos_z_round, 0, sizeOut[0] - 1)
            pos_x_round = np.clip(pos_x_round, 0, sizeOut[1] - 1)

            # Remove out of grid bubbles
            outP = (pos_z < 0) | (pos_z >= sizeOut[0]) | (pos_x < 0) | (pos_x >= sizeOut[1])
            valid_indices = ~outP

            pos_z_round = pos_z_round[valid_indices]
            pos_x_round = pos_x_round[valid_indices]

            # Increment the intensity count of the pixel crossed by a track by +1
            for z, x in zip(pos_z_round, pos_x_round):
                MatOut[z, x] += 1

    elif mode in {'2D_vel_z', '2D_velnorm', '2D_velmean'}:
        print(f'Building 2D ULM image (size: {sizeOut}).')
    
        for track in Tracks:
            track = np.array(track, dtype=float)
            print("Track avant le calcul de la norme :", track)
            pos_z_round = np.rint(track[:, 0]).astype(int)
            pos_x_round = np.rint(track[:, 1]).astype(int)

            # Ensure indices are within valid range
            pos_z_round = np.clip(pos_z_round, 0, sizeOut[0] - 1)
            pos_x_round = np.clip(pos_x_round, 0, sizeOut[1] - 1)

            # Calculer les vitesses
            if mode == '2D_velmean':
                velnorm = track[:, 1]
            else:
                velnorm = np.linalg.norm(track[:, 2:4], axis=1)

            print("Vélocités calculées :", velnorm) 

            if mode == '2D_vel_z':
                sign_mean_velocity = np.sign(np.mean(track[:, 2]))
                print("Sign de la moyenne des vitesses :", sign_mean_velocity)
                velnorm *= sign_mean_velocity

            # Remove out of grid bubbles
            outP = (pos_z_round < 0) | (pos_z_round >= sizeOut[0]) | (pos_x_round < 0) | (pos_x_round >= sizeOut[1])
            valid_indices = ~outP
            print("valid_indices :",valid_indices)
            pos_z_round = pos_z_round[valid_indices]
            pos_x_round = pos_x_round[valid_indices]
            velnorm = velnorm[valid_indices]

            print("Indices valides pour velnorm :", pos_z_round, pos_x_round)
            print("Velocities pour indices valides :", velnorm)

            # Convert indices to linear indices for flat array indexing
            linear_indices = np.ravel_multi_index((pos_z_round, pos_x_round), sizeOut)
            print("affichage de linear_indices",linear_indices)
            print("test test",MatOut,MatOut_vel)
            for ind, vel in zip(linear_indices, velnorm):
                MatOut.ravel()[ind] += 1
                MatOut_vel.ravel()[ind] += vel
            print(MatOut.ravel()[ind])  # Affiche: 1.0
            print(MatOut_vel.ravel()[ind])  # Affiche: 1592.084121
    else:
        raise ValueError('Wrong mode selected')

    if np.any(MatOut > 0):
        print("ha anaaaaaya")
        # Average velocity MatOut_vel_ialgo
        MatOut_vel[MatOut > 0] /= MatOut[MatOut > 0]
    

    return MatOut, MatOut_vel 
