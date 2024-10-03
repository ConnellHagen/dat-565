import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import umap

SEED = 98532

df_read = pd.read_table("assignment_5/seeds.tsv", header=None)

# normalize to z-scores
scalar = StandardScaler()
df = pd.DataFrame(scalar.fit_transform(df_read.iloc[:, :7]))

# add labels column back in
df = df.assign(**{"label": df_read.iloc[:, 7]})
df.columns = df.columns.astype(str)


inertia = []
for i in range(1,8):
    kmeans = KMeans(n_clusters=i, random_state=SEED).fit(df)
    inertia += [kmeans.inertia_]

# elbow plot
# plt.scatter(range(1, len(inertia) + 1), inertia)
# plt.plot(range(1, len(inertia) + 1), inertia)
# plt.show()

color_map = {
    1: "red",
    2: "green",
    3: "blue"
}

for i in range(1, 7 + 1):
    for j in range(1, 7 + 1):
        if (j <= i):
            continue
        # plt.subplot(7, 7, (i - 1) * 7 + j)
        # plt.scatter(df.iloc[:, i - 1], df.iloc[:, j - 1], c=df["label"].map(color_map), s=1)
# plt.show()

# i = 1, j = 7 was the best
# plt.scatter(df.iloc[:, 0], df.iloc[:, 6], c=df["label"].map(color_map), s=5)
# plt.show()

# transformer = GaussianRandomProjection(n_components=2, random_state=SEED)
# df_random = transformer.fit_transform(df.iloc[:, :7])

# the green is pretty clustered, but the red and blue are mixing significantly
# when not seeding, there was an occasional good one, but most were pretty mixed up
# plt.scatter(df_random[:, 0], df_random[:, 1], c=df["label"].map(color_map))
# plt.show()

# reducer = umap.UMAP(random_state=SEED)
# df_umap = reducer.fit_transform(df.iloc[:, :7])
# # plt.scatter(df_umap[:, 0], df_umap[:, 1], c=df["label"].map(color_map))
# plt.show()

kmeans = KMeans(n_clusters=3, random_state=SEED).fit(df)
k_labels = kmeans.labels_

same_cluster = np.zeros((df.shape[0], df.shape[0]))

for i in range(0, same_cluster.shape[0]):
    for j in range(0, same_cluster.shape[1]):
        if (i >= j):
            continue

        if (k_labels[i] == k_labels[j]):
            same_cluster[i][j] += 1
        if (df.iloc[i, :]["label"] == df.iloc[j, :]["label"]):
            same_cluster[i][j] += 1

same = 0
for i in range(0, same_cluster.shape[0]):
    for j in range(0, same_cluster.shape[1]):
        if (i >= j):
            continue

        if (same_cluster[i][j] != 1):
            same += 1

# finding the correct clustering by taking the clusters that are most commonly correct -- assumes that the model doesnt completely suck
correct_cluster = np.zeros((3, 3))
for i in range(0, same_cluster.shape[0]):
    correct_cluster[k_labels[i]][(df.iloc[i, :]["label"] - 1).astype(np.int32)] += 1
num_correct_cluster = 0
for i in range(0, correct_cluster.shape[0]):
    largest = correct_cluster[i][0]
    for j in range(1, correct_cluster[i].shape[0]):
        if correct_cluster[i][j] > largest:
            largest = correct_cluster[i][j]
    num_correct_cluster += largest

num_items = same_cluster.shape[0]

rand_index = same / ((pow(num_items, 2) - num_items) / 2)
accuracy = num_correct_cluster / num_items


linkage_matrix = linkage(df, method="ward", metric="euclidean")
dendrogram(linkage_matrix)
plt.show()