// Module: kmeans.rs
// Purpose: Implement the K-Means clustering algorithm for grouping data points based on features.

use nalgebra::DMatrix;
use rand::Rng;

/// KMeans struct to hold clustering parameters.
/// Fields:
/// - k: Number of clusters to form
/// - max_iter: Maximum number of iterations for convergence
pub struct KMeans {
    pub k: usize,
    pub max_iter: usize,
}

impl KMeans {
    /// Create a new KMeans instance with specified parameters.
    /// Inputs:
    /// - k: Number of clusters
    /// - max_iter: Maximum number of iterations
    /// Outputs:
    /// - A new KMeans instance
    pub fn new(k: usize, max_iter: usize) -> Self {
        KMeans { k, max_iter }
    }

    /// Fit the KMeans model to the data and return cluster assignments.
    /// What it does: Groups data points into k clusters by iteratively updating centroids.
    /// Inputs:
    /// - data: Feature matrix where each row represents a data point
    /// Outputs:
    /// - Vector of cluster labels for each data point
    /// High-level logic:
    /// 1. Randomly initialize centroids from the data points.
    /// 2. Assign each data point to the nearest centroid.
    /// 3. Recalculate centroids as the mean of assigned points.
    /// 4. Repeat steps 2-3 until max_iter is reached.
    pub fn fit(&self, data: &DMatrix<f64>) -> Vec<usize> {
        let mut rng = rand::rng(); // Initialize random number generator
        let num_rows = data.nrows();
        let num_cols = data.ncols();
        let mut centroids: DMatrix<f64> = DMatrix::zeros(self.k, num_cols);

        // Step 1: Randomly initialize centroids
        for i in 0..self.k {
            let idx = rng.random_range(0..num_rows); // Select random row index
            centroids.set_row(i, &data.row(idx));
        }
        let mut labels = vec![0; num_rows];

        // Main loop for max_iter iterations
        for _ in 0..self.max_iter {
            // Step 2: Assign each point to the nearest centroid
            for i in 0..num_rows {
                let point = data.row(i);
                let mut min_dist = f64::MAX;
                let mut best_cluster = 0;
                for j in 0..self.k {
                    let centroid = centroids.row(j);
                    let dist = (point - centroid).norm_squared(); // Compute squared Euclidean distance
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                labels[i] = best_cluster;
            }

            // Step 3: Update centroids
            let mut counts = vec![0; self.k];
            let mut new_centroids = DMatrix::zeros(self.k, num_cols);
            for (i, &label) in labels.iter().enumerate() {
                // Accumulate sum of points for each cluster
                for col in 0..num_cols {
                    new_centroids[(label, col)] += data[(i, col)];
                }
                counts[label] += 1;
            }
            for j in 0..self.k {
                if counts[j] > 0 {
                    // Compute mean by dividing sum by number of points
                    for col in 0..num_cols {
                        new_centroids[(j, col)] /= counts[j] as f64;
                    }
                }
            }
            centroids = new_centroids; // Update centroids for next iteration
        }
        labels // Return final cluster assignments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_kmeans_new() {
        let kmeans = KMeans::new(3, 50);
        assert_eq!(kmeans.k, 3);
        assert_eq!(kmeans.max_iter, 50);
    }

    #[test]
    fn test_kmeans_fit() {
        let data = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 1.1, 1.1]);
        let kmeans = KMeans::new(2, 10);
        let labels = kmeans.fit(&data);

        // Check that we have 4 labels (one for each row)
        assert_eq!(labels.len(), 4);

        // Check that labels are either 0 or 1 (since k=2)
        let unique_labels: std::collections::HashSet<_> = labels.into_iter().collect();
        assert!(unique_labels.len() <= 2);
        assert!(unique_labels.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    fn test_kmeans_clustering() {
        // Two distinct clusters: (0,0), (0.1,0.1) and (10,10), (10.1,10.1)
        let data = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1]);
        let kmeans = KMeans::new(2, 100);
        let labels = kmeans.fit(&data);

        // Check that points close to each other are in the same cluster
        assert_eq!(labels[0], labels[1]); // (0,0) and (0.1,0.1) should be in same cluster
        assert_eq!(labels[2], labels[3]); // (10,10) and (10.1,10.1) should be in same cluster
        assert_ne!(labels[0], labels[2]); // Different clusters for distant points
    }
}
