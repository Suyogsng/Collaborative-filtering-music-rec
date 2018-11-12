import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from music.models import PlayCount
from scipy import sparse
import numpy as np

def cos_sim(SongID):
    songModel = PlayCount.objects.all()
    q = songModel.values(  'playcount','song_id', 'user_id')
    songRatings = pd.DataFrame.from_records(q)
    songRatings = songRatings.pivot_table(index = ['song_id'],columns = ['user_id'],values='playcount')
    songRatings.reset_index(inplace = True)
    songRatingsNormalized = (songRatings- songRatings.mean())/(songRatings.max()-songRatings.min())
    songRatingsNormalized = songRatingsNormalized.fillna(0)
    songRatingsNormalized =sparse.csr_matrix(songRatingsNormalized)
    cosSim = cosine_similarity(songRatingsNormalized)
    testSong = songRatings[songRatings['song_id'] == int(SongID)].index.values.astype(int)[0]
    recommendations = []
    for index in np.argsort(cosSim[testSong])[:-5:-1]:
        if index != testSong:
            recommendations.append(songRatings.iloc[index]['song_id'])

    return recommendations

def pearsonCorr(userID):
    songModel = PlayCount.objects.all()
    q = songModel.values('playcount','song_id', 'user_id')
    songRatings = pd.DataFrame.from_records(q)
    songRatings = songRatings.pivot_table(index = ['user_id'],columns = ['song_id'],values='playcount')
    corrMatrix = songRatings.corr(method='pearson',min_periods=2)

    myRatings = songRatings.iloc[songRatings.index.get_loc(userID) - 1].dropna()
    simCandidates = pd.Series()
    for i in range(0,len(myRatings.index)):
       sims = corrMatrix[myRatings.index[i]].dropna()
    #    sims = sims.map(lambda x:x*myRatings[i])
       simCandidates = simCandidates.append(sims)
    simCandidates = simCandidates.groupby(simCandidates.index).sum()
    simCandidates.sort_values(inplace = True, ascending= False)
    #filteredSims = simCandidates.drop(myRatings.index)
    filteredSims = simCandidates[~simCandidates.isin(myRatings)]
    return filteredSims
