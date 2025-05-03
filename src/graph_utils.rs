// Module: graph_utils.rs
// Purpose: Create a similarity graph from the similarity matrix.

use nalgebra::DMatrix;
use petgraph::graph::UnGraph;
use std::collections::HashMap;

/// Create a graph with edges where similarity exceeds the threshold.
/// Inputs:
/// - similarity: Similarity matrix
/// - keys: List of keys (e.g., state names)
/// - threshold: Similarity threshold
/// Outputs:
/// - Undirected graph
pub fn create_similarity_graph(
    similarity: &DMatrix<f64>,
    keys: &[String],
    threshold: f64,
) -> UnGraph<String, f64> {
    let mut G = UnGraph::<String, f64>::default();
    let mut node_map = HashMap::new();

    // Add nodes
    for (i, loc) in keys.iter().enumerate() {
        let node = G.add_node(loc.clone());
        node_map.insert(i, node);
    }

    // Add edges where similarity exceeds threshold
    let num_rows = similarity.nrows();
    for i in 0..num_rows {
        for j in (i + 1)..num_rows {
            if similarity[(i, j)] > threshold {
                let node_i = node_map[&i];
                let node_j = node_map[&j];
                G.add_edge(node_i, node_j, similarity[(i, j)]);
            }
        }
    }

    G
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_create_similarity_graph() {
        let similarity =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.9, 0.1, 0.9, 1.0, 0.2, 0.1, 0.2, 1.0]);
        let keys = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let G = create_similarity_graph(&similarity, &keys, 0.5);

        assert_eq!(G.node_count(), 3);
        assert_eq!(G.edge_count(), 1); // Only A-B with similarity 0.9 > 0.5
    }
}
