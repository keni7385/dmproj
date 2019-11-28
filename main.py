from app.algorithms.random_clustering import RandomClustering
from app.data.reader import Reader

path = "graphs_processed/ca-GrQc.txt"
task = Reader.read(path)
task.solve(RandomClustering(1137))
