import pandas as pd
import numpy as np
from scipy import spatial
import operator
r_cols = ['user_id','song_id','rating']
ratings = pd.read_csv('ratings.csv', sep = '\t', names = r_cols, usecols = range(3) )
songProperties = ratings.groupby('song_id').agg({'rating':[np.size,np.mean]})
songProperties.head()
songDict=[]
with open(r'genre_indexed.csv') as f:
    for line in f:
        fields = line.rstrip('\n').split(',')
        songID = int(fields[0])
        actual_id = fields[1]
        genres = fields[3:832]
        songDict.append(songID)
        songDict[songID] = (actual_id,genres)
def ComputeDistance(a,b):
    genresA = np.array(a)
    genresB = np.array(b)
    genresA = np.array(genresA,dtype=float)
    genresB = np.array(genresB,dtype=float)
    genres = [genresA,genresB]
    genreDistance = spatial.distance.pdist(genres,metric='cityblock')
    return genreDistance

def getNeighbors(song_id,K):
    distances=[]
    for j in range(len(songDict)-1):
        if songDict[j][0] == song_id:
            songID = j
    for i in range(10000):
        if(i != songID):
            dist = int(ComputeDistance(songDict[songID][1],songDict[i][1])[0])
            distances.append((songDict[i][0],dist))
    distances.sort(key=operator.itemgetter(1),reverse = True)
    neighbors = []
    for x in range(K):
    	neighbors.append(distances[x])
    return neighbors

neighbors = getNeighbors('SOWBWRV12A6D4FB3D0',10)
neighbors
