# finalCapstone
# Project – PCA and various clustering techniques
In this project, we explore the differences between various states in the US using unsupervised learning methods. We use 2 clustering methods, hierarchical clustering and K-means Clustering to cluster unlabelled data points. Ultimately, there are two regions: safer region and less safer region.


**Table of Contents**
- Installing 
- Dataset
- PCA - Standardised Data
- Hierarchical clustering
- K-means clustering

# Installing the codes
Python codes are written in Jupyter Notebook goto https://github.com/Kai-hung-tang/finalCapstone.git
# Dataset
This dataset is from the US Arrests Kaggle challengelink [here](https://www.kaggle.com/datasets/kurohana/usarrets). This dataset contains statistics, in arrests per 100,000 residents for assault, murder, and rape in each of the 50 US states in 1973. Also given is the percentage of the population living in urban areas.
# PCA – Standardised Data
####Code 

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
	X_std = StandardScaler().fit_transform(X)
	std_pca = PCA()
    X_std_trans = std_pca.fit_transform(X_std)

# Hierarchical clustering
#### Code 
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(pca_df)

# K-means clustering
#### Code
    from sklearn.cluster import KMeans

    k=2
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(pca_df)
    cent = kmeans.cluster_centers_

# Contact
This project is produced by me. If you have any comments, please email me at dtangkh5@gmail.com.

