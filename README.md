# Iris Dataset Clustering Project

## Overview
This project focuses on clustering the Iris dataset using two different clustering algorithms:
1. **K-Means Clustering**
2. **Agglomerative Hierarchical Clustering**

The Iris dataset is a popular dataset in machine learning and statistics, consisting of 150 samples of iris flowers with three species (Setosa, Versicolor, and Virginica). Each sample has four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The goal of this project is to group the data into clusters that correspond to the three species of Iris flowers.


---

## Dataset
The Iris dataset is available in the scikit-learn library. It can also be downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

### Features:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

### Target:
- Three species of iris flowers: Setosa, Versicolor, Virginica.

---

## Algorithms Used

### 1. K-Means Clustering
K-Means is a centroid-based clustering algorithm. It partitions the data into `k` clusters, minimizing the variance within each cluster. 

**Steps:**
- Choose the number of clusters (`k`).
- Randomly initialize cluster centroids.
- Assign data points to the nearest centroid.
- Recalculate centroids.
- Repeat until convergence.

**Parameters:**
- `n_clusters`: Number of clusters.
- `init`: Initialization method for centroids.
- `max_iter`: Maximum iterations allowed.

### 2. Agglomerative Hierarchical Clustering
This is a bottom-up clustering approach where each data point starts as its own cluster, and pairs of clusters are merged iteratively based on a linkage criterion.

**Linkage Methods:**
- Single Linkage
- Complete Linkage
- Average Linkage
- Ward’s Method

**Steps:**
1. Compute the pairwise distance matrix.
2. Merge the closest clusters based on linkage.
3. Repeat until a single cluster is formed.

**Parameters:**
- `n_clusters`: Number of clusters.
- `linkage`: Linkage criterion (e.g., 'ward', 'complete').

---

## Project Workflow

1. **Data Preprocessing:**
   - Load the Iris dataset.
   - Normalize features for better clustering performance.

2. **K-Means Clustering:**
   - Implement the algorithm with `n_clusters=3`.
   - Visualize the clusters.
   - Analyze the clustering performance using metrics like silhouette score.

3. **Agglomerative Hierarchical Clustering:**
   - Apply the algorithm with `n_clusters=3` and different linkage criteria.
   - Visualize dendrograms and clusters.

4. **Comparison:**
   - Compare the performance of K-Means and Agglomerative Clustering.
   - Discuss strengths and limitations of each method.

---

## Results

- Visualizations include scatter plots of clustered data and dendrograms.
- Metrics like silhouette score and Davies-Bouldin index are used for evaluation.

---



---

## Dependencies

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Install dependencies using the command:
```bash
pip install -r requirements.txt
```




## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

---
