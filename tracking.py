import numpy as np
import matplotlib
import os
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import math
import scipy.io
from scipy.interpolate import interp1d
import json
np.set_printoptions(threshold=2000,suppress = True)

def nearest_neightbor_linker(source, target, max_distance):
    n_source_points = len(source)
    n_target_points = len(target)
    D = np.full((n_source_points,n_target_points),fill_value=np.nan)

    for i in range(n_source_points):
        current_point = source[i]
        for j in range(n_target_points):
            target_point = target[j]
            
            D[i,j] = (current_point[1]-target_point[1])**2 + (current_point[2]-target_point[2])**2
            
            if D[i,j] > max_distance**2:
                D[i,j] = np.inf
    
    target_indices = np.full((n_source_points),-1)
    target_distances = np.full((n_source_points),np.nan)

    while not (np.isinf(D).all()):
        closest_targets = D.argmin(axis=1)

        min_distances = D.min(axis=1)
        sorted_index = np.argsort(min_distances)

        for i in range(len(sorted_index)):
            source_index = sorted_index[i]
            target_index = closest_targets[sorted_index[i]]

            if any(target_indices == target_index):
                break
            
            target_indices[source_index] = target_index
            target_distances[source_index] = np.sqrt(min_distances[sorted_index[i]])

            D[:,target_index] = np.inf
            D[source_index,:] = np.inf

            if np.isinf(D).all():
                break
    
    unassigned_targets = np.setdiff1d(list(range(n_target_points)),target_indices)
    unassigned_sources = np.argwhere(target_indices == -1)

    return target_indices, target_distances, unassigned_sources, unassigned_targets


def outerplus(M, x, y):
    ny = M.shape[1]
    minval = np.inf
    for c in range(ny):
        M[:, c] = M[:, c] - (x + y[c])
        minval = min(minval, np.min(M[:, c]))
    rIdxcIdx = np.argwhere(M == minval)
    return minval, rIdxcIdx[:,0], rIdxcIdx[:,1]


def munkres(costMat):
    #setup
    assignment = np.full(costMat.shape[0],-1)
    cost = 0

    validMat = np.logical_and(~np.isnan(costMat), costMat < np.inf)

    bigM = 10**(np.ceil(np.log10(sum(costMat[validMat])+1))+1)
    costMat[np.where(~validMat)] = bigM

    validCol = np.sum(validMat, axis=0) > 0
    validRow = np.sum(validMat, axis=1) > 0
    nRows = sum(validRow)
    nCols = sum(validCol)
    n = max(nRows,nCols)

    try:
        maxv = 10*max(costMat[validMat])
    except:
        return np.full(costMat.shape[0],-1),0
    dMat = np.full((n,n), maxv)

    dMat[:nRows, :nCols] = costMat[validRow,:][:,validCol]


    #step 1
    minR = np.min(dMat,axis=1)[:,np.newaxis]
    minC = np.min(dMat - minR, axis=0)

    #step 2
    zP = dMat == (minC + minR)

    starZ = np.full((n,1),-1,dtype=int)
    while zP.any():
        rc = np.argwhere(zP)
        r,c = rc[0]
        starZ[r] = c
        zP[r,:] = False
        zP[:,c] = False

    #step 3
    while True:
        if (starZ>=0).all():
            break
        
        coverColumn = np.full(n,False)
        coverColumn[starZ[starZ > -1]] = True
        coverRow = np.full((n,1),False)
        primeZ = np.full((n,1),-1,dtype=int)

        rIdxcIdx = np.argwhere(dMat[~coverRow[:,0]][:, ~coverColumn] == minR[~coverRow][:,np.newaxis] + minC[~coverColumn])
        rIdx, cIdx = rIdxcIdx[:,0][:,np.newaxis],rIdxcIdx[:,1][:,np.newaxis]

        #step 4
        while True:
            
            cR = np.argwhere(~coverRow)[:,0][:,np.newaxis]
            cC = np.argwhere(~coverColumn)[:,0]

            rIdx = cR[rIdx][:,0]
            cIdx = cC[cIdx]
            if len(cIdx.shape) > 1 and cC.shape != (1,):
                cIdx = cIdx[:,0]
            Step = 6
            while not cIdx.size == 0:
                uZr = rIdx[0]
                uZc = cIdx[0]

                primeZ[uZr] = uZc

                stz = starZ[uZr]
                if len(stz.shape) == 1:
                    stz = stz[:,np.newaxis]
                if stz == -1:
                    Step = 5
                    break
                coverRow[uZr] = True
                coverColumn[stz] = False
                
                z = rIdx==uZr

                rIdx = np.delete(rIdx, np.argwhere(z))
                cIdx = np.delete(cIdx, np.argwhere(z))

                cR = np.argwhere(~coverRow)[:,0][:,np.newaxis]

                z = dMat[~coverRow[:,0]][:,stz][:,0] == minR[~coverRow][:,np.newaxis] + minC[stz]

                rIdx = np.concatenate([rIdx, cR[z]])
                
                if len(z.shape) == 2 and z.shape[1] > 0:
                    zsize = sum(z[:,0])
                else:
                    zsize = sum(z[0])
                right =  np.full((zsize,1),True)
                if right.size != 0:
                    if len(stz.shape) > 0:
                        stz = stz[0]
                    right = np.where(right, stz, 0)
                    #right = stz[right]
                    if right.ndim > cIdx.ndim:
                        cIdx = cIdx[:,np.newaxis]
                    cIdx = np.concatenate([cIdx,right])
                
            if Step == 6:
                minval, rIdx, cIdx = outerplus(dMat[~coverRow[:,0]][:, ~coverColumn], minR[~coverRow], minC[~coverColumn])
                minC[~coverColumn] += minval
                minR[coverRow] -= minval
            else:
                break
        
        #step 5
        rowZ1 = np.argwhere(starZ == uZc)
        if rowZ1.shape[0] != 0:
            rowZ1 = np.argwhere(starZ == uZc)[0,0]
        else:
            rowZ1 = np.array([])

        starZ[uZr] = uZc
        
        while rowZ1.size != 0:
            starZ[rowZ1] = -1
            uZc = primeZ[rowZ1]
            uZr = np.array([rowZ1])
            rowZ1 = np.argwhere(starZ == uZc)[:,0]
            starZ[uZr] = uZc

    rowIdx = np.argwhere(validRow)
    colIdx = np.argwhere(validCol)
    starZ = starZ[:nRows]
    vIdx = starZ < nCols

    assignment[rowIdx[vIdx]] = colIdx[starZ[vIdx]][:,0]
    pass_indices = assignment[assignment >= 0]
    pass_indices[~np.diag(validMat[assignment >= 0][:,pass_indices])] = -1
    assignment[assignment >= 0] = pass_indices
    cost = np.trace(costMat[assignment >= 0][:,assignment[assignment >= 0]])

    return assignment,cost





def hungarian_linker(source, target, max_distance):
    n_source_points = len(source)
    n_target_points = len(target)
    D = np.full((n_source_points,n_target_points),fill_value=np.nan)

    for i in range(n_source_points):
        current_point = source[i]
        for j in range(n_target_points):
            #print("hung")
            target_point = target[j]
            
            D[i,j] = (current_point[1]-target_point[1])**2 + (current_point[2]-target_point[2])**2
            
            if D[i,j] > max_distance**2:
                D[i,j] = np.inf

    target_indices, total_cost = munkres(D)
    target_indices[np.argwhere(target_indices==-1)] = -1

    target_distances = np.full((n_source_points),np.nan)

    for i in range(len(target_indices)):
        if target_indices[i] < 0:
            continue
        target_distances[i] = np.sqrt(D[i,target_indices[i]])

    unassigned_targets = np.setdiff1d(list(range(n_target_points)),target_indices)
    unassigned_sources = np.argwhere(target_indices == -1)[:,0]
    return target_indices, target_distances, unassigned_sources, unassigned_targets, total_cost





def simple_tracker(slices, max_linking_distance, max_gap_closing):
    n_slices = len(slices)
    unassigned_sources, unassigned_targets, target_indices, target_indices_used = [],[[]],[],[]
    

    #Using Hungarian
    for i in range(n_slices-1):
        #print("s track hung")
        source = slices[i]
        target = slices[i+1]
        ti, _, us, ut, _ = hungarian_linker(source, target, max_linking_distance)
        unassigned_targets.append(list(ut))
        unassigned_sources.append(list(us))
        target_indices.append([(t,i+1) for t in ti])
        target_indices_used.append([False for t in ti])
    target_indices_used.append([False for t in slices[n_slices-1]])
    target_indices.append([(-1, n_slices) for t in slices[n_slices-1]])
    unassigned_sources.append([])

    #Trying to link gaps
    for i in range(n_slices-1):
        if len(unassigned_sources[i]) == 0:
            continue
        for j in range(i+2,min(n_slices,i+max_gap_closing)):
            if len(unassigned_targets[j]) == 0:
                continue
            idx_A = [el for el in range(len(slices[i])) if el in unassigned_sources[i]]
            idx_B = [el for el in range(len(slices[j])) if el in unassigned_targets[j]]
            A = [slices[i][el] for el in range(len(slices[i])) if el in unassigned_sources[i]]
            B = [slices[j][el] for el in range(len(slices[j])) if el in unassigned_targets[j]]
            ti, _, _, _ = nearest_neightbor_linker(A, B, max_linking_distance)
            for k in range(len(ti)):
                #print("s track link")
                if ti[k] == -1:
                    continue
                target_indices[i][idx_A[k]] = (idx_B[ti[k]],j)
                del unassigned_sources[i][unassigned_sources[i].index(idx_A[k])]
                del unassigned_targets[j][unassigned_targets[j].index(idx_B[ti[k]])]
    
    #Slices to tracks
    tracks = []
    index_tracks = []
    for i in range(n_slices):
        for j in range(len(target_indices[i])):
            if target_indices_used[i][j]:
                continue
            target_indices_used[i][j] = True
            track = [slices[i][j]]
            index_track = [j]
            next_s = target_indices[i][j]
            index_track += [-1]*(next_s[1]-i-1)
            k = next_s[1]
            while k < n_slices and next_s[0] != -1:
                track.append(slices[next_s[1]][next_s[0]])
                index_track.append(next_s[0])
                target_indices_used[next_s[1]][next_s[0]] = True
                next_s = target_indices[next_s[1]][next_s[0]]
                index_track += [-1]*(next_s[1]-k-1)
                k = next_s[1]
            tracks.append(track)
            index_tracks.append(index_track)

    return tracks,index_tracks


def generate_random_points_uniform(n_points=1, nb_frames=10, max_speed=0.2):
    tracks = [[] for f in range(n_points)]
    for p in range(n_points):
        point = (0,round(random.uniform(0,1),3),round(random.uniform(0,1),3))
        tracks[p].append(point)
        for f in range(1,nb_frames):
            point = (f,round(tracks[p][-1][1] + random.uniform(-max_speed,max_speed),3),round(tracks[p][-1][2] + random.uniform(-max_speed,max_speed),3))
            tracks[p].append(point)
    return tracks

def generate_random_points_directional(n_points=1, nb_frames=10, speed_variance=0.2, angular_variance=2*math.pi/10):
    tracks = [[] for f in range(n_points)]
    for p in range(n_points):
        current_angle = random.uniform(0,2*math.pi)
        point = (0,round(random.uniform(0,1),3),round(random.uniform(0,1),3))
        tracks[p].append(point)
        for f in range(1,nb_frames):
            current_speed = random.gauss(current_angle, speed_variance)
            current_angle = random.gauss(current_angle, angular_variance)
            point = (f,round(tracks[p][-1][1] + math.cos(current_angle)*current_speed,3),round(tracks[p][-1][2] + math.sin(current_angle)*current_speed,3))
            tracks[p].append(point)
    return tracks

def generate_tracking_example(n_points=1, nb_frames=10, probability_disapear=0, function=generate_random_points_uniform):
    tracks = function(n_points=n_points, nb_frames=nb_frames)
    slices = [[] for _ in range(nb_frames)]
    cut_track = [[] for _ in range(n_points)]
    for s in range(nb_frames):
        for p in range(n_points):
            if random.uniform(0,1) > probability_disapear:
                slices[s].append(tracks[p][s])
                cut_track[p].append(tracks[p][s])
    for s in slices:
        random.shuffle(s)

    return slices,tracks,cut_track

def show_follow(tracking, original, rand_original_cut, nb_frames=10):
    fig, axs = plt.subplots(1,3)
    axs[0].set_title("Tracked")
    axs[1].set_title("Original + Cut")
    axs[2].set_title("Original")
    for track_ind,current_tracking in enumerate([tracking, rand_original_cut, original]):
        for track in current_tracking:
            r,g=random.uniform(0.25,0.75), random.uniform(0.25,0.75)
            color = (r, g, 1-(g+r)/2)
            for i in range(len(track)-1):
                linex,liney = [track[i][1],track[i+1][1]], [track[i][2],track[i+1][2]]
                if track[i+1][0] > track[i][0] + 1:
                    axs[track_ind].plot(linex,liney, color=color, linestyle='--')
                else:
                    axs[track_ind].plot(linex,liney, color=color)
            for i in range(len(track)):
                axs[track_ind].plot(track[i][1],track[i][2], marker = 'o', color=(1,track[i][0]/nb_frames,track[i][0]/nb_frames))
                axs[track_ind].annotate(track[i][0],(track[i][1],track[i][2]))
        
    plt.show(block=True)

# def get_error(tracks, expected_tracks):
#     points = [p for l in expected_tracks for p in l]
#     for point in points:
#         for track in expected_tracks:
#             if point in track:
#                 expected_track = track
#         for track in tracks:
#             if point in track:
#                 predicted_track = track
            



def track_interp_smooth(x, factor):
    returned = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        #print("int s")
        dist_factor_left = min(i,factor//2)
        dist_factor_right = min(x.shape[0] - i-1,factor//2)
        dist_factor = min(dist_factor_left,dist_factor_right)
        returned[i] = np.average(x[i-dist_factor:i+dist_factor+1])
    return returned

def track_interp_diff(x):
    returned = np.zeros(x.shape[0])
    for i in range(returned.shape[0]-1):
        #print("int diff")
        returned[i+1] = x[i+1] - x[i]
    returned[0] = returned[1]
    return returned

def tracking(slices, max_linking_distance, max_gap_closing, min_length, time_per_frame, res):
    interp_factor = 1/max_linking_distance/res*.8
    smooth_factor = 20
    
    simple_tracks,a = simple_tracker(slices, max_linking_distance, max_gap_closing)

    # for x in a:
    #     print(np.array(x)+1)

    returned = [None]*len(simple_tracks)

    for i_track,track in enumerate(simple_tracks):
       #print("track")
        if len(track) <= 1:
            returned[i_track]=[[],[],[],[],[]]
            continue
        
        xi = np.array([p[1] for p in track])
        zi = np.array([p[2] for p in track])
        #print(xi)
        #print(zi)
        timeAbs = np.arange(zi.shape[0]) * time_per_frame
        zu = interp1d(np.arange(zi.shape[0]),track_interp_smooth(zi,smooth_factor))(np.around(np.arange(0,zi.shape[0]-1+1e-10,interp_factor),10))
        xu = interp1d(np.arange(xi.shape[0]),track_interp_smooth(xi,smooth_factor))(np.around(np.arange(0,xi.shape[0]-1+1e-10,interp_factor),10))
        TimeAbs_interp = interp1d(np.arange(timeAbs.shape[0]),track_interp_smooth(timeAbs,smooth_factor))(np.around(np.arange(0,timeAbs.shape[0]-1+1e-10,interp_factor),10))
        #print(xu)
        #print(zu)
        vzu=track_interp_diff(zu)/track_interp_diff(TimeAbs_interp)
        vxu=track_interp_diff(xu)/track_interp_diff(TimeAbs_interp)

        #print(vxu)
        #print(vzu)
        #print("===")

        if zi.shape[0]>min_length:
            returned[i_track]=np.array([zu,xu,vzu,vxu,TimeAbs_interp])

    return returned


# nb_frames=5
# for i in range(100):
#     rand,rand_original,rand_original_cut = generate_tracking_example(n_points=2, nb_frames=nb_frames, probability_disapear=0.2, function=generate_random_points_directional)
#     track = simple_tracker(rand, 10, 10)
#     # get_error(track, rand_original_cut)
#     show_follow(track, rand_original, rand_original_cut, nb_frames=nb_frames)
    
# rand,rand_original,rand_original_cut = generate_tracking_example(n_points=10, nb_frames=20, probability_disapear=0.2, function=generate_random_points_directional)
# print(rand)

def get_matlab_tracking_input_matrix(exp):
    s = "["
    for sl_i,sl in enumerate(exp):
        for p in sl:
            s += " 1, "+str(p[2])+", "+str(p[1])+", "+str(sl_i)+";\n"
    s += "]"
    return s


#### RENDERING FUNCTIONS ####
def create_tracking_InVivo(IQ):
    pos = []
    for frame in range(IQ.shape[2]):
        l = []
        for x in range(IQ.shape[0]):
            for y in range(IQ.shape[1]):
                if IQ[x, y, frame] == 1:
                    l.append((frame,round(x,3),round(y,3)))
        pos.append(l)
    return pos

def create_tracking_InSilico(IQ):
    pos = []
    for i,frame in enumerate(IQ):
        l = []
        for curr in IQ[frame]:
            l.append((i,curr[1],curr[0]))
        pos.append(l)
    return pos

# UNUSED 
def reconstruct_img(track,IQ):
    h = IQ.shape[0]
    w = IQ.shape[1]
    img_reconstructed = np.zeros((h,w))
    for frame in range(len(track)):
        if track[frame][0] != []:
            print(track[frame])
            for curr in range(len(track[frame])):
                print(len(track[frame]))
                print(curr)
                x = int(track[frame][1][curr])
                y = int(track[frame][0][curr])
                img_reconstructed[x,y] += 0.05
    return img_reconstructed

########################################
# EXEMPLE
"""exp = [[(0, 0.881, 0.636), (0, 0.026, 0.058), (0, 0.075, 0.748), (0, 0.495, 0.847), (0, 0.294, 0.692), (0, 0.051, 0.419), (0, 0.397, 0.602), (0, 0.544, 0.485)], [(1, -0.428, 2.149), (1, 0.289, 0.118), (1, -1.977, 2.262), (1, -0.016, 2.49), (1, 5.427, -2.314), (1, -3.814, 0.786), (1, 3.834, -4.295)], [(2, -7.083, 0.401), 
(2, 0.021, 3.728), (2, -4.379, 3.591), (2, 10.373, -5.749), (2, -4.611, -1.725), (2, 1.482, -8.886)], [(3, 15.891, -5.729), (3, -5.46, -0.306), (3, -1.135, 5.92), (3, -10.255, -0.933), (3, 0.44, 4.875), (3, 0.195, -0.002), (3, -6.444, 5.364)], [(4, 21.739, -3.23), (4, -7.834, 7.31), (4, -5.589, -11.036), (4, -8.39, -3.357), (4, -3.509, -15.647), (4, -0.028, 0.447), (4, 1.099, 5.778), (4, -2.223, 7.716)], [(5, -2.03, -19.423), (5, -9.052, -10.671), (5, -14.712, 2.204), (5, -0.119, 1.312), (5, -4.36, 8.213), (5, -8.522, 9.399)], [(6, -9.038, 10.98), (6, 1.112, 8.288), (6, -14.742, -0.91), (6, -0.177, 2.388), (6, -6.94, 9.167), (6, -11.319, -8.559), (6, -5.367, -11.673), (6, -14.731, 4.162), (6, -0.658, -24.013)], [(7, -7.613, 11.536), (7, -15.419, 0.949), (7, 39.757, -5.217), (7, 0.26, 9.711), (7, -15.23, 5.612), (7, -13.222, -10.238), (7, 0.585, 3.551)], [(8, 0.939, 11.911), (8, 2.535, 3.293), (8, -12.684, 13.624), (8, -13.225, -16.121), (8, 5.483, -26.881), (8, -15.629, -13.2), (8, 45.677, -5.358), (8, -7.799, 13.322)], [(9, 4.483, 0.823), (9, -17.254, 4.332), (9, -0.126, 13.126), (9, -9.647, 14.091), (9, -19.545, -12.16), (9, -14.937, 15.325)], [(10, -2.267, 13.943), (10, -22.488, 4.434), (10, -18.426, 6.444), (10, 17.915, -30.243)], [(11, 56.838, -17.605), (11, -15.58, 13.932), (11, -25.432, 3.145), (11, -17.107, 19.077), (11, -1.664, 16.633), (11, -25.353, -11.828), (11, 23.496, -28.678), (11, -16.689, -27.935), (11, -20.085, 7.734), (11, 4.488, -8.133)], [(12, 58.541, -23.019), (12, -20.665, -29.876), (12, -22.395, 8.045), (12, -29.261, 2.775), (12, 
24.298, -22.349), (12, -18.372, 15.56), (12, -27.161, -15.011), (12, -16.58, 21.81), (12, 5.672, -13.029)], [(13, -19.833, 17.566), (13, -1.922, 19.675), (13, -24.19, -30.745), (13, -25.036, 9.019), (13, -31.017, -16.127), (13, 3.016, -16.707), (13, 21.824, -14.767)], [(14, 65.973, -25.251), (14, -27.02, 10.963), (14, -1.833, 21.846), (14, -32.424, 6.885), (14, -18.907, 19.251), (14, -34.345, -16.323), (14, -0.956, -20.259), (14, -15.403, 23.333)], [(15, -37.302, -18.025), (15, -30.255, -35.444), (15, -14.757, 23.558), (15, 18.139, 0.44), (15, 72.525, -26.812), (15, -29.695, 11.705), (15, -33.693, 7.833)], [(16, -32.31, 13.02), (16, 76.605, -21.72), (16, -0.304, 23.761), (16, -18.387, 21.816), (16, -40.51, -19.894), (16, 12.546, 6.07)], [(17, -43.729, -21.458), (17, 0.156, 23.955), (17, -36.546, -37.791), (17, -12.661, -32.119), (17, -15.079, 23.752), (17, -32.449, 11.264), (17, 4.844, 9.672)], [(18, -15.783, 24.326), (18, -34.594, 17.659), (18, -0.086, 24.658), (18, -46.161, -24.36), (18, -37.009, -41.579), (18, -16.928, 23.313), (18, -12.498, -37.732)], [(19, 87.317, -2.758), (19, -46.618, -28.148), (19, -10.083, 18.76), (19, -34.276, 19.493), (19, -0.211, 26.429), (19, -16.152, 24.86), (19, -16.751, 23.569), (19, -13.731, -42.27), (19, -31.988, 11.344)]]
#print(get_matlab_tracking_input_matrix(exp))"""
#result = tracking(exp, 5, 5, 0, 1, 1)


def track_render(mode, file_path):

    print("The tracking and the rendering should take something like a minute or two, please be patient!")
    if mode == "InVivo":
        ###-- INVIVO --###
        mat_file = scipy.io.loadmat(file_path)

        # mat_file is a dict
        IQ = mat_file["RBULMrs_1"]

        RB1_Bmode = scipy.io.loadmat('RB1_IQ_filt.mat')['IQ_filt']


        ### BEGIN TRACKING INVIVO ###
        track = create_tracking_InVivo(IQ)

        exact = tracking(track, 2, 0, 0, 1, 1)
        ########################################

        ### SAVE TO USE IT IN MATLAB ###
        data = {}
        data['x'] = [element[0] for element in exact]
        data['y'] = [element[1] for element in exact]

        scipy.io.savemat("resultsInVivo.mat",data)
        ########################################


        ### BEGIN PLOTTING INVIVO ###

        ### PLOT THE SUM OF TRACKING ###
        for i in range(len(exact)):

            x = exact[i][0]
            y = exact[i][1]
            plt.plot([-i for i in x],[-ii for ii in y], color = "black",linewidth=0.1)
        plt.show()


        NumFrame = 1 #can be changed, how many frames do you want to see
        IntPower = 1/2

        ### PLOT WITH OUR RESULTS ###
        plt.imshow(RB1_Bmode[:,:,0]**IntPower ,cmap="gray")
        for i in range(len(exact)-1,-1,-1):
            #print(exact[i][0])
            x = exact[i][0]
            y = exact[i][1]
            plt.plot([i for i in x],[ii for ii in y], color = "white",linewidth=0.25)
        plt.title("Image originale avec bulles suivies algorithmiquement")
        plt.show()

        ### PLOT WITH GIVEN RESULTS ###
        for kk in range(NumFrame):
            plt.imshow(RB1_Bmode[:,:,kk]**IntPower ,cmap="gray")
            lig,col = np.nonzero(IQ[:,:,kk])
            plt.scatter(col,lig, color = "white",linewidth=0.25)
            plt.title("Image originale")
            plt.show()
        ########################################
    
    if mode == "InSilico":
        ###-- INSILICO --###
        f = open(file_path) # NEEDS TO BE A JSON !!!!!!!!!!!!
        results = json.load(f)

        ### BEGIN TRACKING INSILICO ###
        track = create_tracking_InSilico(results)

        exact = tracking(track, 1, 0, 0, 1, 1)
        ########################################

        ### SAVE TO USE IT IN MATLAB ###
        data = {}
        data['x'] = [element[0] for element in exact]
        data['y'] = [element[1] for element in exact]

        scipy.io.savemat("resultsInSilico.mat",data)
        ########################################


        ### BEGIN PLOTTING INSILICO ###
        ### PLOT THE SUM OF TRACKING ###
        for i in range(len(exact)-1,-1,-1):

            #print(exact[i][0])
            x = exact[i][0]
            y = exact[i][1]
            plt.plot([i for i in x],[-ii for ii in y], color = "orange",linewidth=0.1)
        plt.title("Tracking sur résultats du CNN")
        plt.show()


        
        fond = np.zeros((84,143))
        x = [element[0] for element in exact]
        y = [element[1] for element in exact]
        plt.imshow(fond**0.5 ,cmap="gray")
        for i in range(len(exact)-1,-1,-1):
            #print(exact[i][0])
            x = exact[i][0]
            y = exact[i][1]
            plt.plot([i for i in x],[ii for ii in y], color = "orange",linewidth=0.1)
        plt.title("Tracking")
        plt.show()

        ### FOR COMPARAISON ###
        MatOut_RS = scipy.io.loadmat('SilicoULM_RS.mat')['SilicoULM_RS']
        plt.imshow(MatOut_RS**0.5,cmap="hot")
        #plt.colorbar()
        plt.title('Radial Symmetry and tracking')
        plt.show()
    else:
        ###-- NOT GOOD ZONE --##
        print("Usage: track_render(\"InVivo\"|\"InSilico\",IQ_path)")
        exit(1)
        ########################################

track_render("InVivo","RBULMrs_1.mat")