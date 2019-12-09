import networkx as nx
from app.data.reader import Reader

########################################################################################################
# Supplementary code
# No score is computed, only the number of self-edges is computed for each graph.
########################################################################################################

files = ["ca-GrQc", "Oregon-1", "soc-Epinions1", "web-NotreDame", "roadNet-CA"]
path = "graphs_processed/%s.txt"
output_directory = "results"
paths = [path % file for file in files]

for file in paths:
    task = Reader.read(file)
    print("Selfloops in %s: [%d]" % (task["name"], sum(1 for i in nx.selfloop_edges(task["graph"]))))
exit(0)

# Results:
# Selfloops in ca-GrQc: [6]
# Selfloops in Oregon-1: [0]
# Selfloops in soc-Epinions1: [0]
# Selfloops in web-NotreDame: [27455]
# Selfloops in roadNet-CA: [0]
