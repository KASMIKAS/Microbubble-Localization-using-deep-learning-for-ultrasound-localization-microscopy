import numpy as np
import time
import scipy.io as sio
from ULM_LOCALIZATION2D import ULM_localization2D
from ULM_tracking2D import ULM_tracking2D
from ULM_tracking2D import simpletracker

def PALA_multiULM(IQ, listAlgo, ULM, PData, *args):
   
    if 'parameters' not in ULM:
        ULM['parameters'] = {}
    if 'NLocalMax' not in ULM['parameters']:
        ULM['parameters']['NLocalMax'] = 3  # safeguard

    tracking = 1
    SaveData = False
    savingFileName = None

    for i in range(len(args)):
        if args[i] == 'tracking':
            tracking = args[i + 1]
        elif args[i] == 'savingfilename':
            savingFileName = args[i + 1]
            SaveData = True

    ULM['max_linking_distance'] *= PData['PDelta'][0]

    IQ = np.abs(IQ)
    Nalgo = len(listAlgo)
    ProcessingTime = np.zeros(Nalgo)

    print('Processing algo: ', end='')
    Track_raw = [None] * Nalgo
    Track_interp = [None] * Nalgo

    for ialgo in range(Nalgo):
        print(f'{ialgo + 1} ', end='')
        
        t0 = time.time()

        if listAlgo[ialgo].lower() in ['wa', 'weighted_average']:
            ULM['LocMethod'] = 'wa'
        elif listAlgo[ialgo].lower() in ['radial', 'radial_vivo', 'radial_silicio']:
            ULM['LocMethod'] = 'radial'
        elif listAlgo[ialgo].lower() == 'radial_sg':
            ULM['LocMethod'] = 'radial_sg'
        elif listAlgo[ialgo].lower() == 'interp_cubic':
            ULM['LocMethod'] = 'interp'
            ULM['parameters']['InterpMethod'] = 'cubic'
        elif listAlgo[ialgo].lower() == 'interp_lanczos':
            ULM['LocMethod'] = 'interp'
            ULM['parameters']['InterpMethod'] = 'lanczos3'
        elif listAlgo[ialgo].lower() in ['interp_spline']:
            ULM['LocMethod'] = 'interp'
            ULM['parameters']['InterpMethod'] = 'spline'
        elif listAlgo[ialgo].lower() == 'gaussian_fit':
            ULM['LocMethod'] = 'curvefitting'
        elif listAlgo[ialgo].lower() in ['interp_bilinear', 'no_localization', 'no_shift']:
            ULM['LocMethod'] = 'nolocalization'
        else:
            raise ValueError('Wrong method selected')

        MatTracking = ULM_localization2D(IQ, ULM)

        ProcessingTime[ialgo] = time.time() - t0

        # Checking and adjusting dimensions to avoid IndexError
        # Assurez-vous que PData['PDelta'] contient des éléments avec des indices suffisants.
        if len(PData['PDelta'][0][0]) >= 3 and len(PData['Origin']) >= 3:
            MatTracking[:, 1:3] = (MatTracking[:, 1:3] - [1, 1]) * [PData['PDelta'][0][0][2], PData['PDelta'][0][0][0]] + [PData['Origin'][2], PData['Origin'][0]]
        #else:
         #   raise IndexError("PData['PDelta'] or PData['Origin'] does not have sufficient dimensions. Required length is at least 3.")

        if tracking:
            Track_raw[ialgo], Track_interp[ialgo] = ULM_tracking2D(MatTracking.astype(float), ULM, 'pala')
        else:
            Track_raw[ialgo] = MatTracking.astype(np.float32)
            Track_interp[ialgo] = []

        Track_interp[ialgo] = [track.astype(np.float32) for track in Track_interp[ialgo]]
        Track_raw[ialgo] = [track.astype(np.float32) for track in Track_raw[ialgo]]

    varargout = []
    if len(args) > 2:
        varargout.append(ProcessingTime)

    if SaveData:
        print('saving... ', end='')
        ProTime = ProcessingTime
        sio.savemat(savingFileName, {
            'Track_raw': Track_raw, 
            'Track_interp': Track_interp, 
            'ProTime': ProTime, 
            'ULM': ULM, 
            'PData': PData, 
            'listAlgo': listAlgo, 
            'Nalgo': Nalgo
        }, do_compression=True)
    
    print('end.')
    
    return Track_raw, Track_interp, varargout if varargout else None
