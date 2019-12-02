from app.algorithms.random_clustering import RandomClustering
from app.algorithms.spectral_clustering import SpectralClustering
from app.algorithms.fifty_kmeans import BalancedSpectralClustering
from app.algorithms.spectral_clustering_bis import SpectralClusteringBis
from app.data.reader import Reader

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# files = ["ca-GrQc", "Oregon-1"]  # , "roadNet-CA", "soc-Epinions1", "web-NotreDame"]+
files = ["ca-GrQc", "Oregon-1", "roadNet-CA", "soc-Epinions1", "web-NotreDame"]
path = "graphs_processed/%s.txt"
paths = [path % file for file in files]

for filepath in paths:
    for offset in range(20):
        task = Reader.read(filepath, offset)
        task.solve(SpectralClusteringBis(normalised=True))
        task.solve(SpectralClusteringBis(normalised=False))
