# US Crime Analysis using Rust

This project analyzes US crime data from 1980 onwards to identify patterns and relationships across various dimensions, including state-based, age group-based, and decade-based analyses. Implemented in Rust, it leverages statistical and machine learning techniques such as correlation analysis, similarity analysis, centrality measures, and K-Means clustering. The dataset is sourced from the [US Crime DataSet on Kaggle](https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset).

## A. Project Overview

The project conducts the following analyses to uncover crime patterns:

- **State-based Analysis**:
  - Examines correlations between state and victim sex, relationship, and weapon.
  - Performs similarity, centrality, and clustering analyses on state-level crime data.
- **Age Group-based Analysis**:
  - Analyzes correlations between age groups (10-year bins) and victim sex, relationship, and weapon.
  - Includes similarity, centrality, and clustering analyses for age groups.
- **Decade-based Analysis**:
  - Investigates correlations between decades and victim sex, relationship, and state.
  - Conducts similarity, centrality, and clustering analyses for decades.
- **Victim-Perpetrator Analysis**:
  - Studies correlations between victim sex and perpetrator sex, and victim sex and relationship.
- **Relationship and Weapon Analysis**:
  - Analyzes the correlation between relationship and weapon used in crimes.

The dataset contains crime records from 1980 onwards, enabling a comprehensive exploration of crime trends.

## B. Data Processing

- **Data Loading**:
  - Loads data from `US_Crime_DataSet.csv` using the `csv` crate.
  - Supports sampling (e.g., 0.3% of data) with a seed for reproducibility to handle large datasets.
- **Data Cleaning and Transformation**:
  - Bins victim ages into 10-year age groups (e.g., 10-19, 20-29).
  - Extracts decades from crime years (e.g., 1990s, 2000s).
- **Feature Extraction**:
  - Aggregates crime type counts per state, age group, and decade for similarity and clustering analyses.

## C. Code Structure

The project is organized into modular components:

| Module           | Purpose                                                                       |
| ---------------- | ----------------------------------------------------------------------------- |
| `data_loader.rs` | Loads and preprocesses crime data, including sampling and feature extraction. |
| `similarity.rs`  | Computes cosine similarity matrices for extracted features.                   |
| `graph_utils.rs` | Creates similarity graphs based on similarity matrices and thresholds.        |
| `analysis.rs`    | Performs centrality (degree and closeness) and community detection analyses.  |
| `correlation.rs` | Conducts Chi-Square tests for correlation between categorical variables.      |
| `kmeans.rs`      | Implements K-Means clustering for grouping similar entities.                  |
| `main.rs`        | Orchestrates the analysis pipeline and outputs results.                       |

### Key Functions

- `load_records` (data_loader.rs): Loads and samples crime data.
- `load_state_data`, `load_age_group_data`, `load_decade_data` (data_loader.rs): Extract features for states, age groups, and decades.
- `compute_cosine_similarity` (similarity.rs): Calculates similarity matrices.
- `create_similarity_graph` (graph_utils.rs): Builds graphs from similarity matrices.
- `compute_degree_centrality`, `compute_closeness_centrality`, `detect_communities` (analysis.rs): Analyze graph properties.
- `analyze_state_correlations`, `analyze_age_group_correlations`, etc. (correlation.rs): Perform correlation analyses.
- `fit` (kmeans.rs): Executes K-Means clustering.

## D. Testing

Unit tests validate each module's functionality, with 19 tests covering all key components:

- `test_load_records`: Ensures correct data loading and sampling.
- `test_load_state_data`, `test_load_age_group_data`, `test_load_decade_data`: Verify feature extraction.
- `test_compute_cosine_similarity`: Confirms accurate similarity calculations.
- `test_create_similarity_graph`: Validates graph construction.
- `test_compute_degree_centrality`, `test_compute_closeness_centrality`, `test_detect_communities`: Test centrality and community detection.
- `test_analyze_state_correlations`, etc.: Validate correlation analyses.
- `test_kmeans_clustering`: Ensures correct clustering.

Run tests with:

```bash
cargo test
```

### Test Output Example

```bash
running 19 tests
test analysis::tests::test_compute_degree_centrality ... ok
test analysis::tests::test_detect_communities ... ok
test correlation::tests::test_analyze_age_group_correlations ... ok
test correlation::tests::test_analyze_decade_correlations ... ok
test correlation::tests::test_analyze_victim_perpetrator_relationship ... ok
test correlation::tests::test_analyze_relationship_weapon ... ok
test correlation::tests::test_analyze_victim_perpetrator_sex ... ok
test correlation::tests::test_analyze_state_correlations ... ok
test analysis::tests::test_compute_closeness_centrality ... ok
test data_loader::tests::test_load_age_group_data ... ok
test data_loader::tests::test_load_decade_data ... ok
test data_loader::tests::test_load_state_data ... ok
test graph_utils::tests::test_create_similarity_graph ... ok
test kmeans::tests::test_kmeans_fit ... ok
test kmeans::tests::test_kmeans_new ... ok
test similarity::tests::test_compute_cosine_similarity ... ok
test data_loader::tests::test_load_records ... ok
test kmeans::tests::test_kmeans_clustering ... ok

test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

## E. Results

The program generates detailed outputs for each analysis type:

- **Correlation Analyses**: Chi-Square p-values indicating significance.
- **Centrality Analyses**: Top entities (states, age groups, decades) by degree and closeness centrality.
- **Clustering Analyses**: Cluster assignments and community counts, grouping entities by crime profiles.

### Interpretations

- **Correlation Analyses**: Low p-values (< 0.05) suggest significant correlations.
- **Centrality Analyses**: High centrality scores indicate influential entities in the similarity network.
- **Clustering Analyses**: Clusters reveal groups with similar crime characteristics.

### Example Results

Below are outputs from running the project:

#### State-based Analysis

```bash
=== Analysis 1: State-based Analysis ===
State vs Victim Sex Correlation (Chi-Square p-value): 0.025962
State vs Relationship Correlation (Chi-Square p-value): 0.000000
State vs Weapon Correlation (Chi-Square p-value): 0.000000

Top 5 states by degree centrality:
Alaska: 0.9796
Georgia: 0.9796
New York: 0.9796
Iowa: 0.9796
Alabama: 0.9796

Top 5 states by closeness centrality:
Florida: 1.0205
Louisiana: 1.0205
New York: 1.0205
California: 1.0205
North Carolina: 1.0205

Number of communities detected in the state graph: 2

Clusters of states based on crime features:
Cluster 2: ["Missouri", "Virginia", "New Jersey", "Maryland", "Ohio", "Tennessee", "South Carolina", "North Carolina", "Indiana", "Georgia", "Louisiana", "Arizona", "Pennsylvania"]
Cluster 4: ["Rhodes Island", "Kansas", "Montana", "Oregon", "South Dakota", "Utah", "West Virginia", "Wyoming", "North Dakota", "New Hampshire", "Nebraska", "Vermont", "Minnesota", "Delaware", "Iowa", "Hawaii", "Alaska", "Idaho"]
Cluster 1: ["Texas", "California"]
Cluster 3: ["Michigan", "New York", "Illinois", "Florida"]
Cluster 0: ["Kentucky", "Nevada", "New Mexico", "District of Columbia", "Massachusetts", "Colorado", "Washington", "Wisconsin", "Mississippi", "Arkansas", "Oklahoma", "Connecticut", "Alabama"]

Cluster 2 - State vs Victim Sex Correlation: 0.096996
Cluster 2 - State vs Relationship Correlation: 0.923930
Cluster 2 - State vs Weapon Correlation: 0.003699

Cluster 4 - State vs Victim Sex Correlation: 0.762773
Cluster 4 - State vs Relationship Correlation: 0.999999
Cluster 4 - State vs Weapon Correlation: 0.409502

Cluster 1 - State vs Victim Sex Correlation: 0.265652
Cluster 1 - State vs Relationship Correlation: 0.007296
Cluster 1 - State vs Weapon Correlation: 0.270169

Cluster 3 - State vs Victim Sex Correlation: 0.178691
Cluster 3 - State vs Relationship Correlation: 0.167988
Cluster 3 - State vs Weapon Correlation: 0.000087

Cluster 0 - State vs Victim Sex Correlation: 0.159423
Cluster 0 - State vs Relationship Correlation: 0.430229
Cluster 0 - State vs Weapon Correlation: 0.933552
```

#### Age Group-based Analysis

```bash
=== Analysis 2: Age Group-based Analysis ===
Age Group vs Victim Sex Correlation (Chi-Square p-value): 0.000000
Age Group vs Relationship Correlation (Chi-Square p-value): 0.000000
Age Group vs Weapon Correlation (Chi-Square p-value): 0.000000

Top 5 age groups by degree centrality:
80-89: 1.0000
30-39: 1.0000
0-9: 1.0000
40-49: 1.0000
60-69: 1.0000

Top 5 age groups by closeness centrality:
20-29: 0.9996
50-59: 0.9996
30-39: 0.9996
60-69: 0.9995
40-49: 0.9995

Number of communities detected in the age group graph: 1

Clusters of age groups based on crime features:
Cluster 3: ["50-59"]
Cluster 1: ["20-29"]
Cluster 4: ["Unknown", "70-79", "90-99", "80-89"]
Cluster 2: ["30-39", "40-49", "10-19"]
Cluster 0: ["60-69", "0-9"]

Cluster 3 - Age Group vs Victim Sex Correlation: 1.000000
Cluster 3 - Age Group vs Relationship Correlation: 1.000000
Cluster 3 - Age Group vs Weapon Correlation: 1.000000

Cluster 1 - Age Group vs Victim Sex Correlation: 1.000000
Cluster 1 - Age Group vs Relationship Correlation: 1.000000
Cluster 1 - Age Group vs Weapon Correlation: 1.000000

Cluster 4 - Age Group vs Victim Sex Correlation: 0.000000
Cluster 4 - Age Group vs Relationship Correlation: 0.992786
Cluster 4 - Age Group vs Weapon Correlation: 0.406399

Cluster 2 - Age Group vs Victim Sex Correlation: 0.619696
Cluster 2 - Age Group vs Relationship Correlation: 0.113444
Cluster 2 - Age Group vs Weapon Correlation: 0.046482

Cluster 0 - Age Group vs Victim Sex Correlation: 0.005784
Cluster 0 - Age Group vs Relationship Correlation: 0.000006
Cluster 0 - Age Group vs Weapon Correlation: 0.015649
```

#### Decade-based Analysis

```bash
=== Analysis 3: Decade-based Analysis ===
Decade vs Victim Sex Correlation (Chi-Square p-value): 0.648423
Decade vs Relationship Correlation (Chi-Square p-value): 0.053295
Decade vs State Correlation (Chi-Square p-value): 0.021006

Top 5 decades by degree centrality:
1990s: 1.0000
2010s: 1.0000
2000s: 1.0000
1980s: 1.0000

Top 5 decades by closeness centrality:
1990s: 1.0000
1980s: 1.0000
2000s: 0.9999
2010s: 0.9999

Number of communities detected in the decade graph: 1

Clusters of decades based on crime features:
Cluster 2: ["2010s"]
Cluster 0: ["1980s", "1990s"]
Cluster 3: ["2000s"]

Cluster 2 - Decade vs Victim Sex Correlation: 1.000000
Cluster 2 - Decade vs Relationship Correlation: 1.000000
Cluster 2 - Decade vs State Correlation: 1.000000

Cluster 0 - Decade vs Victim Sex Correlation: 0.154735
Cluster 0 - Decade vs Relationship Correlation: 0.353007
Cluster 0 - Decade vs State Correlation: 0.628805

Cluster 3 - Decade vs Victim Sex Correlation: 1.000000
Cluster 3 - Decade vs Relationship Correlation: 1.000000
Cluster 3 - Decade vs State Correlation: 1.000000
```

#### Victim-Perpetrator and Relationship-Weapon Analyses

```bash
=== Analysis 4: Victim-Perpetrator Sex and Relationship Correlation ===
Victim Sex vs Perpetrator Sex Correlation (Chi-Square p-value): 0.000531
Victim Sex vs Relationship Correlation (Chi-Square p-value): 0.000000

=== Analysis 5: Relationship and Weapon Correlation ===
Relationship vs Weapon Correlation (Chi-Square p-value): 0.000000
```

These results provide insights into crime patterns across different dimensions, identifying significant correlations, influential entities, and groups with similar crime profiles.

## F. Usage Instructions

1. **Dependencies**: Install Rust and dependencies listed in `Cargo.toml`.
2. **Data**: Place `US_Crime_DataSet.csv` in the project root.
3. **Build and Run**:

   ```bash
   cargo build --release
   cargo run --release [path] [frac] [seed]
   ```

   - `[path]`: Path to the CSV file (default: `US_Crime_DataSet.csv`).
   - `[frac]`: Fraction of data to sample (default: 0.003, i.e., 0.3%).
   - `[seed]`: Random seed for sampling reproducibility (default: 42).

   Example:

   ```bash
   cargo run --release US_Crime_DataSet.csv 0.01 123
   ```

   This command runs the analysis on 1% of the data with a seed of 123.

4. **Output**: Results (p-values, centrality scores, clusters) are printed to the console.

## G. AI Assistance and Citations

- **AI Assistance**: AI language models assisted in designing analysis logic (e.g., cosine similarity, Chi-Square tests). All suggestions were verified and implemented independently.
- **Citations**:
  - Rust documentation and crate resources guided implementation.
  - Dataset: [US Crime DataSet from Kaggle](https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset).

## H. Appendix

### Directory Structures

```
us_crime_analysis/
├── src/
│   ├── analysis.rs
│   ├── correlation.rs
│   ├── data_loader.rs
│   ├── graph_utils.rs
│   ├── kmeans.rs
│   ├── main.rs
│   ├── similarity.rs
├── ...
├── Cargo.toml
└── US_Crime_DataSet.csv
```

### Cargo.toml

```toml
[package]
name = "us_crime_analysis"
version = "0.1.0"
edition = "2024"

[dependencies]
statrs = "0.18"
nalgebra = "0.33"
petgraph = "0.8"
rand = "0.9"
csv = "1.3"
```
