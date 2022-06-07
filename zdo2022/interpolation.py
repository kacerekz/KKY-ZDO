import math

# Interpolates positions using linear interpolation
# Returns list of edited positions
def InterpolatePositions(positions):
    newPositions = []

    # Adds empty frames at the start if neccessary
    for i in range(0, positions[0][0]):
        newPos = [i, positions[0][1]]
        newPositions.append(newPos)

    newPositions.append(positions[0])

    for i in range(1, len(positions)):
        closest = []
        
        # frame index - get 2 consecutive saved frames
        index2 = positions[i][0]
        index1 = positions[i-1][0]

        # distance in between the two frames time 
        diff = index2 - index1
        if (diff == 1):
            newPositions.append(positions[i])
            continue

        # find the closest position to each position from the second frame
        usedStarts = []
        copies = []
        minDist = float('inf')
        closestInd = -1
        for j in range(0, len(positions[i][1]), 2):
            minDist = float('inf')
            closestInd = -1
            for k in range(0, len(positions[i-1][1]), 2):
                if (usedStarts.count(k) > 0):
                    continue
                
                x1 = positions[i-1][1][k]
                y1 = positions[i-1][1][k+1]
                
                x2 = positions[i][1][j]
                y2 = positions[i][1][j+1]
                
                dist = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
                if(dist < minDist):
                    minDist = dist
                    closestInd = k
            
            if(minDist < 100):
                closestPair = [j,closestInd]
                closest.append(closestPair)
                usedStarts.append(closestInd)

                # this test should not be neessary but im not sure:
                if (copies.count(j) > 0):
                    copies.remove(j)
            else:
                copies.append(j)

            
        # Create new positions for missing frames
        for j in range(1, diff):
            newPos2 = []
            for k in range(0, len(closest)):
                # Compute step in x and y
                stepX = (positions[i][1][closest[k][0]] - positions[i-1][1][closest[k][1]]) / diff
                stepY = (positions[i][1][closest[k][0]+1] - positions[i-1][1][closest[k][1]+1]) / diff
            
                # Linear interpolation time
                position = [0, 0]
                position[0] = positions[i-1][1][closest[k][1]] + j * stepX
                position[1] = positions[i-1][1][closest[k][1]+1] + j * stepY
                newPos2.append(position[0])
                newPos2.append(position[1])
            
            # uncomment this if you want to see the unused positions copied into the frames:
            #for k in range(0, len(copies)):
            #    newPos2.append(positions[i][1][copies[k]])
            #    newPos2.append(positions[i][1][copies[k] + 1])
                
            newPos = [index1 + j, newPos2]
            newPositions.append(newPos)

        newPositions.append(positions[i])

    return newPositions

def InterpolatePositionsK(positions):
    newPositions = []

    # Adds empty frames at the start if neccessary
    for i in range(0, positions[0][0]):
        newPos = [i, positions[0][1]]
        newPositions.append(newPos)

    newPositions.append(positions[0])

    for i in range(1, len(positions)):
        closest = []
        
        # frame index - get 2 consecutive saved frames
        index2 = positions[i][0]
        index1 = positions[i-1][0]

        # distance in between the two frames time 
        diff = index2 - index1
        if (diff == 1):
            newPositions.append(positions[i])
            continue

        # find the closest position to each position from the second frame
        usedStarts = []
        minDist = float('inf')
        closestInd = -1
        for j in range(0, len(positions[i][1]), 2):
            minDist = float('inf')
            closestInd = -1
            for k in range(0, len(positions[i-1][1]), 2):
                if (usedStarts.count(k) > 0):
                    continue
                
                x1 = positions[i-1][1][k]
                y1 = positions[i-1][1][k+1]
                
                x2 = positions[i][1][j]
                y2 = positions[i][1][j+1]
                
                dist = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
                if(dist < minDist):
                    minDist = dist
                    closestInd = k
            
            
            closestPair = [j,closestInd]
            closest.append(closestPair)
            usedStarts.append(closestInd)
            
        # Create new positions for missing frames
        for j in range(1, diff):
            newPos2 = []
            for k in range(0, len(closest)):
                # Compute step in x and y
                stepX = (positions[i][1][closest[k][0]] - positions[i-1][1][closest[k][1]]) / diff
                stepY = (positions[i][1][closest[k][0]+1] - positions[i-1][1][closest[k][1]+1]) / diff
            
                # Linear interpolation time
                position = [0, 0]
                position[0] = positions[i-1][1][closest[k][1]] + j * stepX
                position[1] = positions[i-1][1][closest[k][1]+1] + j * stepY
                newPos2.append(position[0])
                newPos2.append(position[1])
                
            newPos = [index1 + j, newPos2]
            newPositions.append(newPos)

        newPositions.append(positions[i])

    return newPositions

# Pads out missing frames
def InterpolationPadding(positions):
    newPositions = []

    # Adds empty frames at the start if neccessary
    for i in range(0, positions[0][0]):
        newPos = [i, positions[0][1]]
        newPositions.append(newPos)

    newPositions.append(positions[0])

    for i in range(1, len(positions)):
        # frame index - get 2 consecutive saved frames
        index2 = positions[i][0]
        index1 = positions[i-1][0]

        # distance in between the two frames time 
        diff = index2 - index1
        for c in range(0, diff-1):
            newPositions.append([index1 + (c+1), positions[i-1][1]])
        
        newPositions.append(positions[i])

    return newPositions