// Module: analysis.rs
// Purpose: Perform centrality and cluster analysis on the graph.

use petgraph::algo::{connected_components, dijkstra};
use petgraph::graph::UnGraph;
use std::collections::HashMap;

/// Compute degree centrality for each node.
/// Inputs:
/// - G: Graph
/// Outputs:
/// - HashMap mapping locations to centrality scores
pub fn compute_degree_centrality(G: &UnGraph<String, f64>) -> HashMap<String, f64> {
    let mut centrality = HashMap::new();

    // Calculate degree for each node
    for node in G.node_indices() {
        let degree = G.neighbors(node).count() as f64;
        let loc = &G[node];
        centrality.insert(loc.clone(), degree);
    }

    // Normalize by (n-1)
    let n = G.node_count() as f64;
    if n > 1.0 {
        for val in centrality.values_mut() {
            *val /= n - 1.0;
        }
    }

    centrality
}

/// Compute closeness centrality for each node in the graph.
/// What it does: Measures how close a node is to all other nodes based on shortest paths.
/// Inputs:
/// - G: Graph with nodes as crime types and edge weights as similarities
/// Outputs:
/// - HashMap mapping crime types to their closeness centrality scores
/// High-level logic:
/// 1. For each node, compute shortest paths to all other nodes using Dijkstra's algorithm.
/// 2. sum the distances
/// 3. calculate closeness as (n-1)/total_distance.
pub fn compute_closeness_centrality(G: &UnGraph<String, f64>) -> HashMap<String, f64> {
    let mut centrality = HashMap::new();

    for node in G.node_indices() {
        // Compute shortest paths from the current node to all others using Dijkstra's algorithm
        // Edge weights are similarities, so higher weights mean "closer" nodes; we invert them for distance
        let distances = dijkstra(G, node, None, |e| {
            let weight = *e.weight();
            if weight > 0.0 {
                1.0 / weight
            } else {
                f64::INFINITY
            }
        });

        // Sum the distances to all other nodes
        let total_distance: f64 = distances.values().sum();

        let n = G.node_count() as f64;
        // Calculate closeness: (n-1)/total_distance, handle case where total_distance is 0
        let closeness = if total_distance > 0.0 {
            (n - 1.0) / total_distance
        } else {
            0.0
        };

        let crime = &G[node];
        centrality.insert(crime.clone(), closeness);
    }

    centrality
}

/// Detect the number of connected components as clusters.
/// Inputs:
/// - G: Graph
/// Outputs:
/// - Number of clusters
pub fn detect_communities(G: &UnGraph<String, f64>) -> usize {
    connected_components(G)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_compute_degree_centrality() {
        let mut G = UnGraph::<String, f64>::default();
        let a = G.add_node("A".to_string());
        let b = G.add_node("B".to_string());
        let c = G.add_node("C".to_string());
        G.add_edge(a, b, 1.0);
        G.add_edge(b, c, 1.0);

        let centrality = compute_degree_centrality(&G);

        assert_eq!(centrality["A"], 0.5); // degree 1 / 2
        assert_eq!(centrality["B"], 1.0); // degree 2 / 2
        assert_eq!(centrality["C"], 0.5); // degree 1 / 2
    }

    #[test]
    fn test_compute_closeness_centrality() {
        let mut G = UnGraph::<String, f64>::default();
        let a = G.add_node("A".to_string());
        let b = G.add_node("B".to_string());
        let c = G.add_node("C".to_string());
        G.add_edge(a, b, 0.9); // Similarity 0.9 -> Distance 1/0.9
        G.add_edge(b, c, 0.9); // Similarity 0.9 -> Distance 1/0.9

        let centrality = compute_closeness_centrality(&G);

        let dist_ab = 1.0 / 0.9;
        let dist_bc = 1.0 / 0.9;
        let dist_ac = dist_ab + dist_bc;

        // For A: distance to B is 1/0.9, to C is (1/0.9 + 1/0.9)
        let closeness_a = 2.0 / (dist_ab + dist_ac);
        // For B: distance to A is 1/0.9, to C is 1/0.9
        let closeness_b = 2.0 / (dist_ab + dist_bc);
        // For C: distance to B is 1/0.9, to A is (1/0.9 + 1/0.9)
        let closeness_c = 2.0 / (dist_bc + dist_ac);

        assert!((centrality["A"] - closeness_a).abs() < 1e-6);
        assert!((centrality["B"] - closeness_b).abs() < 1e-6);
        assert!((centrality["C"] - closeness_c).abs() < 1e-6);
    }

    #[test]
    fn test_detect_communities() {
        let mut G = UnGraph::<String, f64>::default();
        G.add_node("A".to_string());
        G.add_node("B".to_string());
        G.add_node("C".to_string());

        assert_eq!(detect_communities(&G), 3); // No edges, 3 communities

        let a = G.node_indices().next().unwrap();
        let b = G.node_indices().nth(1).unwrap();
        G.add_edge(a, b, 1.0);

        assert_eq!(detect_communities(&G), 2); // A-B connected, C separate
    }
}
