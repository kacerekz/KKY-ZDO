from sklearn.cluster import KMeans
import numpy as np
import math

# Filter positions using k-means algorithm
def FilterKmeans(positions):
    positions2 = []
    shape = (len(positions) // 2, 2)
    positions3 = np.zeros(shape)
    for p in range(0, len(positions), 2):
        pos = [positions[p], positions[p+1]]
       # positions3.append((positions[p], positions[p+1]))
        positions3[p//2][0] = positions[p]
        positions3[p//2][1] = positions[p+1]
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(positions3)
    
    positions2.append(kmeans.cluster_centers_[0][0])
    positions2.append(kmeans.cluster_centers_[0][1])
    positions2.append(kmeans.cluster_centers_[1][0])
    positions2.append(kmeans.cluster_centers_[1][1])
    positions2.append(kmeans.cluster_centers_[2][0])
    positions2.append(kmeans.cluster_centers_[2][1])
    return positions2

# Filter positions
def Filter(positions):
    # Filter through found positions
    positions2 = []
    m = np.zeros(len(positions))
    for p in range (0, len(positions), 2):
        # Already processed
        if (m[p] == 1):
            continue

        # New position - average
        pos = [0, 0]
        count = 0
        # Compare with other positions
        for p2 in range (0, len(positions), 2):
            if (m[p2] == 1):
                continue

            # Add positions that are close and mark them as processed
            if ( math.sqrt((positions[p] - positions[p2]) * (positions[p] - positions[p2]) + (positions[p+1] - positions[p2+1]) * (positions[p+1] - positions[p2+1])) < 30 ):
                m[p2] = 1
                m[p2+1] = 1
                count = count + 1
                pos[0] += positions[p2]
                pos[1] += positions[p2+1]

        pos[0] /= count
        pos[1] /= count

        positions2.append((int)(pos[0]))
        positions2.append((int)(pos[1]))
        
    return positions2