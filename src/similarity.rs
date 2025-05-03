// Module: similarity.rs
// Purpose: Compute the cosine similarity matrix for the crime data.

use nalgebra::DMatrix;

/// Compute cosine similarity between rows of the feature matrix.
/// Inputs:
/// - data: Feature matrix
/// Outputs:
/// - Similarity matrix
pub fn compute_cosine_similarity(data: &DMatrix<f64>) -> DMatrix<f64> {
    let num_rows = data.nrows();

    // Compute norms of each row
    let norms: Vec<f64> = (0..num_rows).map(|i| data.row(i).norm()).collect();

    // Handle zero norms to avoid division by zero
    let norms: Vec<f64> = norms
        .into_iter()
        .map(|x| if x == 0.0 { 1.0 } else { x })
        .collect();

    // Normalize each row by its norm
    let mut normalized = data.clone();
    for i in 0..num_rows {
        normalized.row_mut(i).scale_mut(1.0 / norms[i]);
    }

    // Compute similarity as dot product of normalized vectors
    let similarity = normalized.clone() * normalized.transpose();

    similarity
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_compute_cosine_similarity() {
        // Create a 3x2 dynamic matrix from a row-major slice
        let data = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let sim = compute_cosine_similarity(&data);

        // Verify the results
        assert!((sim[(0, 0)] - 1.0).abs() < 1e-6); // Similarity of row 0 with itself
        assert!((sim[(0, 1)] - 0.0).abs() < 1e-6); // Row 0 vs Row 1
        assert!((sim[(0, 2)] - (1.0 / 2.0f64.sqrt())).abs() < 1e-6); // Row 0 vs Row 2
        assert!((sim[(1, 2)] - (1.0 / 2.0f64.sqrt())).abs() < 1e-6); // Row 1 vs Row 2
        assert!((sim[(2, 2)] - 1.0).abs() < 1e-6); // Similarity of row 2 with itself
    }
}
