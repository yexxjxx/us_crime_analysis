// Main module: us_crime_analysis

#![allow(non_snake_case)]

mod analysis;
mod correlation;
mod data_loader;
mod graph_utils;
mod kmeans;
mod similarity;

use crate::data_loader::CrimeRecord;
use std::env;
use std::error::Error;

/// Main function to run the crime data analysis pipeline.
/// Inputs: None
/// Outputs: Result indicating success or failure
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    // Default arguments
    let mut path = "US_Crime_DataSet.csv".to_string();
    let mut frac = 0.003; // 0.3% of the data
    let mut seed = 42;

    // Parse command-line arguments
    if args.len() > 1 {
        path = args[1].clone();
    }
    if args.len() > 2 {
        frac = args[2].parse::<f64>().unwrap_or(0.003);
    }
    if args.len() > 3 {
        seed = args[3].parse::<u64>().unwrap_or(42);
    }

    // Load data with provided path, frac, and seed
    let records = data_loader::load_records(&path, frac, seed)?;

    // Analysis 1: State-based Analysis
    println!("\n=== Analysis 1: State-based Analysis ===");
    let (state_sex_corr, state_rel_corr, state_weapon_corr) =
        correlation::analyze_state_correlations(&records);
    println!(
        "State vs Victim Sex Correlation (Chi-Square p-value): {:.6}",
        state_sex_corr
    );
    println!(
        "State vs Relationship Correlation (Chi-Square p-value): {:.6}",
        state_rel_corr
    );
    println!(
        "State vs Weapon Correlation (Chi-Square p-value): {:.6}",
        state_weapon_corr
    );

    let (states, state_data) = data_loader::load_state_data(&records)?;
    let state_similarity = similarity::compute_cosine_similarity(&state_data);
    let state_G = graph_utils::create_similarity_graph(&state_similarity, &states, 0.95);

    let state_degree_centrality = analysis::compute_degree_centrality(&state_G);
    let mut sorted_state_degree: Vec<_> = state_degree_centrality.into_iter().collect();
    sorted_state_degree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 states by degree centrality:");
    for (state, cent) in sorted_state_degree.iter().take(5) {
        println!("{}: {:.4}", state, cent);
    }

    let state_closeness_centrality = analysis::compute_closeness_centrality(&state_G);
    let mut sorted_state_closeness: Vec<_> = state_closeness_centrality.into_iter().collect();
    sorted_state_closeness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 states by closeness centrality:");
    for (state, cent) in sorted_state_closeness.iter().take(5) {
        println!("{}: {:.4}", state, cent);
    }

    let state_num_communities = analysis::detect_communities(&state_G);
    println!(
        "\nNumber of communities detected in the state graph: {}",
        state_num_communities
    );

    let state_kmeans = kmeans::KMeans::new(5, 100);
    let state_labels = state_kmeans.fit(&state_data);
    let mut state_clusters: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for (i, &label) in state_labels.iter().enumerate() {
        state_clusters
            .entry(label)
            .or_insert(Vec::new())
            .push(states[i].clone());
    }
    println!("\nClusters of states based on crime features:");
    for (cluster_id, states) in &state_clusters {
        println!("Cluster {}: {:?}", cluster_id, states);
    }
    for (cluster_id, cluster_states) in &state_clusters {
        let cluster_records: Vec<CrimeRecord> = records
            .iter()
            .filter(|r| cluster_states.contains(&r.state))
            .cloned()
            .collect();
        let (sex_p, rel_p, weapon_p) = correlation::analyze_state_correlations(&cluster_records);
        println!(
            "\nCluster {} - State vs Victim Sex Correlation: {:.6}",
            cluster_id, sex_p
        );
        println!(
            "Cluster {} - State vs Relationship Correlation: {:.6}",
            cluster_id, rel_p
        );
        println!(
            "Cluster {} - State vs Weapon Correlation: {:.6}",
            cluster_id, weapon_p
        );
    }

    // Analysis 2: Age Group-based Analysis (10-year bins)
    println!("\n=== Analysis 2: Age Group-based Analysis ===");
    let (age_sex_corr, age_rel_corr, age_weapon_corr) =
        correlation::analyze_age_group_correlations(&records);
    println!(
        "Age Group vs Victim Sex Correlation (Chi-Square p-value): {:.6}",
        age_sex_corr
    );
    println!(
        "Age Group vs Relationship Correlation (Chi-Square p-value): {:.6}",
        age_rel_corr
    );
    println!(
        "Age Group vs Weapon Correlation (Chi-Square p-value): {:.6}",
        age_weapon_corr
    );

    let (age_groups, age_data) = data_loader::load_age_group_data(&records)?;
    let age_similarity = similarity::compute_cosine_similarity(&age_data);
    let age_G = graph_utils::create_similarity_graph(&age_similarity, &age_groups, 0.95);

    let age_degree_centrality = analysis::compute_degree_centrality(&age_G);
    let mut sorted_age_degree: Vec<_> = age_degree_centrality.into_iter().collect();
    sorted_age_degree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 age groups by degree centrality:");
    for (age_group, cent) in sorted_age_degree.iter().take(5) {
        println!("{}: {:.4}", age_group, cent);
    }

    let age_closeness_centrality = analysis::compute_closeness_centrality(&age_G);
    let mut sorted_age_closeness: Vec<_> = age_closeness_centrality.into_iter().collect();
    sorted_age_closeness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 age groups by closeness centrality:");
    for (age_group, cent) in sorted_age_closeness.iter().take(5) {
        println!("{}: {:.4}", age_group, cent);
    }

    let age_num_communities = analysis::detect_communities(&age_G);
    println!(
        "\nNumber of communities detected in the age group graph: {}",
        age_num_communities
    );

    let age_kmeans = kmeans::KMeans::new(5, 100);
    let age_labels = age_kmeans.fit(&age_data);
    let mut age_clusters: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for (i, &label) in age_labels.iter().enumerate() {
        age_clusters
            .entry(label)
            .or_insert(Vec::new())
            .push(age_groups[i].clone());
    }
    println!("\nClusters of age groups based on crime features:");
    for (cluster_id, ages) in &age_clusters {
        println!("Cluster {}: {:?}", cluster_id, ages);
    }
    for (cluster_id, cluster_ages) in &age_clusters {
        let cluster_records: Vec<CrimeRecord> = records
            .iter()
            .filter(|r| {
                let age_group = if r.victim_age < 0.0 || r.victim_age >= 100.0 {
                    "Unknown".to_string()
                } else {
                    format!(
                        "{}-{}",
                        (r.victim_age / 10.0).floor() * 10.0,
                        (r.victim_age / 10.0).floor() * 10.0 + 9.0
                    )
                };
                cluster_ages.contains(&age_group)
            })
            .cloned()
            .collect();
        let (sex_p, rel_p, weapon_p) =
            correlation::analyze_age_group_correlations(&cluster_records);
        println!(
            "\nCluster {} - Age Group vs Victim Sex Correlation: {:.6}",
            cluster_id, sex_p
        );
        println!(
            "Cluster {} - Age Group vs Relationship Correlation: {:.6}",
            cluster_id, rel_p
        );
        println!(
            "Cluster {} - Age Group vs Weapon Correlation: {:.6}",
            cluster_id, weapon_p
        );
    }

    // Analysis 3: Decade-based Analysis
    println!("\n=== Analysis 3: Decade-based Analysis ===");
    let (decade_sex_corr, decade_rel_corr, decade_state_corr) =
        correlation::analyze_decade_correlations(&records);
    println!(
        "Decade vs Victim Sex Correlation (Chi-Square p-value): {:.6}",
        decade_sex_corr
    );
    println!(
        "Decade vs Relationship Correlation (Chi-Square p-value): {:.6}",
        decade_rel_corr
    );
    println!(
        "Decade vs State Correlation (Chi-Square p-value): {:.6}",
        decade_state_corr
    );

    let (decades, decade_data) = data_loader::load_decade_data(&records)?;
    let decade_similarity = similarity::compute_cosine_similarity(&decade_data);
    let decade_G = graph_utils::create_similarity_graph(&decade_similarity, &decades, 0.95);

    let decade_degree_centrality = analysis::compute_degree_centrality(&decade_G);
    let mut sorted_decade_degree: Vec<_> = decade_degree_centrality.into_iter().collect();
    sorted_decade_degree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 decades by degree centrality:");
    for (decade, cent) in sorted_decade_degree.iter().take(5) {
        println!("{}: {:.4}", decade, cent);
    }

    let decade_closeness_centrality = analysis::compute_closeness_centrality(&decade_G);
    let mut sorted_decade_closeness: Vec<_> = decade_closeness_centrality.into_iter().collect();
    sorted_decade_closeness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 decades by closeness centrality:");
    for (decade, cent) in sorted_decade_closeness.iter().take(5) {
        println!("{}: {:.4}", decade, cent);
    }

    let decade_num_communities = analysis::detect_communities(&decade_G);
    println!(
        "\nNumber of communities detected in the decade graph: {}",
        decade_num_communities
    );

    let decade_kmeans = kmeans::KMeans::new(4, 100);
    let decade_labels = decade_kmeans.fit(&decade_data);
    let mut decade_clusters: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for (i, &label) in decade_labels.iter().enumerate() {
        decade_clusters
            .entry(label)
            .or_insert(Vec::new())
            .push(decades[i].clone());
    }
    println!("\nClusters of decades based on crime features:");
    for (cluster_id, decades) in &decade_clusters {
        println!("Cluster {}: {:?}", cluster_id, decades);
    }
    for (cluster_id, cluster_decades) in &decade_clusters {
        let cluster_records: Vec<CrimeRecord> = records
            .iter()
            .filter(|r| cluster_decades.contains(&format!("{}s", (r.year / 10) * 10)))
            .cloned()
            .collect();
        let (sex_p, rel_p, state_p) = correlation::analyze_decade_correlations(&cluster_records);
        println!(
            "\nCluster {} - Decade vs Victim Sex Correlation: {:.6}",
            cluster_id, sex_p
        );
        println!(
            "Cluster {} - Decade vs Relationship Correlation: {:.6}",
            cluster_id, rel_p
        );
        println!(
            "Cluster {} - Decade vs State Correlation: {:.6}",
            cluster_id, state_p
        );
    }

    // Analysis 4: Victim-Perpetrator Sex and Relationship Correlation
    println!("\n=== Analysis 4: Victim-Perpetrator Sex and Relationship Correlation ===");
    let victim_perp_sex_corr = correlation::analyze_victim_perpetrator_sex(&records);
    let victim_perp_rel_corr = correlation::analyze_victim_perpetrator_relationship(&records);
    println!(
        "Victim Sex vs Perpetrator Sex Correlation (Chi-Square p-value): {:.6}",
        victim_perp_sex_corr
    );
    println!(
        "Victim Sex vs Relationship Correlation (Chi-Square p-value): {:.6}",
        victim_perp_rel_corr
    );

    // Analysis 5: Relationship and Weapon (Brutality) Correlation
    println!("\n=== Analysis 5: Relationship and Weapon Correlation ===");
    let rel_weapon_corr = correlation::analyze_relationship_weapon(&records);
    println!(
        "Relationship vs Weapon Correlation (Chi-Square p-value): {:.6}",
        rel_weapon_corr
    );

    Ok(())
}
