// Module: data_loader.rs
// Purpose: Load and preprocess the US crime dataset for various analyses.

use csv;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

// Struct: CrimeRecord
// Represents a single crime incident with relevant fields.
#[derive(Debug, Clone)]
pub struct CrimeRecord {
    pub state: String,
    pub year: i32,
    pub crime_type: String,
    pub victim_sex: String,
    pub victim_age: f64,
    pub perpetrator_sex: String,
    pub relationship: String,
    pub weapon: String,
}

/// Load all records from the CSV file into a vector.
/// Inputs:
/// - path: Path to the CSV file
/// Outputs:
/// - Result containing a vector of CrimeRecord
pub fn load_records<P: AsRef<Path>>(
    path: P,
    frac: f64,
    seed: u64,
) -> Result<Vec<CrimeRecord>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut records = Vec::new();

    // Skip headers
    rdr.headers()?;

    // Process each record
    for result in rdr.records() {
        let record = result?;
        let year: i32 = record[6].parse().unwrap_or(0); // Year (index 6)
        let victim_age: f64 = record[12].parse().unwrap_or(0.0); // Victim Age (index 12)
        let record = CrimeRecord {
            state: record[5].to_string(), // State (index 5)
            year,
            crime_type: record[9].to_string(), // Crime Type (index 9)
            victim_sex: record[11].to_string(), // Victim Sex (index 11)
            victim_age,
            perpetrator_sex: record[15].to_string(), // Perpetrator Sex (index 15)
            relationship: record[19].to_string(),    // Relationship (index 19)
            weapon: record[20].to_string(),          // Weapon (index 20)
        };
        records.push(record);
    }

    // Sample {frac * 100}% of the records
    let sample_size = (records.len() as f64 * frac).ceil() as usize;
    let mut rng = StdRng::seed_from_u64(seed);
    records.shuffle(&mut rng);
    records.truncate(sample_size);

    Ok(records)
}

/// Load data for state-based similarity analysis by aggregating crime features per state.
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - Result containing states and feature matrix (crime type counts)
pub fn load_state_data(
    records: &[CrimeRecord],
) -> Result<(Vec<String>, DMatrix<f64>), Box<dyn Error>> {
    let mut state_map: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let crime_types: Vec<String> = records
        .iter()
        .map(|r| r.crime_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Aggregate by state and crime type
    for record in records {
        let state_entry = state_map.entry(record.state.clone()).or_insert_with(|| {
            let mut map = HashMap::new();
            for ct in &crime_types {
                map.insert(ct.clone(), 0.0);
            }
            map
        });
        *state_entry.entry(record.crime_type.clone()).or_insert(0.0) += 1.0;
    }

    // Create states and feature vectors
    let mut states = Vec::new();
    let mut feature_vectors = Vec::new();
    for (state, crime_counts) in state_map {
        states.push(state);
        let mut features = Vec::new();
        for ct in &crime_types {
            features.push(*crime_counts.get(ct).unwrap_or(&0.0));
        }
        feature_vectors.push(features);
    }

    // Convert to DMatrix
    let num_rows = states.len();
    let num_cols = crime_types.len();
    let flat_data: Vec<f64> = feature_vectors.into_iter().flatten().collect();
    let data = DMatrix::from_row_slice(num_rows, num_cols, &flat_data);

    Ok((states, data))
}

/// Load data for age group-based similarity analysis by aggregating crime features per age group.
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - Result containing age groups and feature matrix (crime type counts)
pub fn load_age_group_data(
    records: &[CrimeRecord],
) -> Result<(Vec<String>, DMatrix<f64>), Box<dyn Error>> {
    let mut age_group_map: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let crime_types: Vec<String> = records
        .iter()
        .map(|r| r.crime_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Aggregate by age group and crime type
    for record in records {
        // Determine age group (10-year bins)
        let age_group = if record.victim_age < 0.0 || record.victim_age >= 100.0 {
            "Unknown".to_string()
        } else {
            format!(
                "{}-{}",
                (record.victim_age / 10.0).floor() * 10.0,
                (record.victim_age / 10.0).floor() * 10.0 + 9.0
            )
        };
        let age_entry = age_group_map.entry(age_group).or_insert_with(|| {
            let mut map = HashMap::new();
            for ct in &crime_types {
                map.insert(ct.clone(), 0.0);
            }
            map
        });
        *age_entry.entry(record.crime_type.clone()).or_insert(0.0) += 1.0;
    }

    // Create age groups and feature vectors
    let mut age_groups = Vec::new();
    let mut feature_vectors = Vec::new();
    for (age_group, crime_counts) in age_group_map {
        age_groups.push(age_group);
        let mut features = Vec::new();
        for ct in &crime_types {
            features.push(*crime_counts.get(ct).unwrap_or(&0.0));
        }
        feature_vectors.push(features);
    }

    // Convert to DMatrix
    let num_rows = age_groups.len();
    let num_cols = crime_types.len();
    let flat_data: Vec<f64> = feature_vectors.into_iter().flatten().collect();
    let data = DMatrix::from_row_slice(num_rows, num_cols, &flat_data);

    Ok((age_groups, data))
}

/// Load data for decade-based similarity analysis by aggregating crime features per decade.
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - Result containing decades and feature matrix (crime type counts)
pub fn load_decade_data(
    records: &[CrimeRecord],
) -> Result<(Vec<String>, DMatrix<f64>), Box<dyn Error>> {
    let mut decade_map: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let crime_types: Vec<String> = records
        .iter()
        .map(|r| r.crime_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Aggregate by decade and crime type
    for record in records {
        // Determine decade (e.g., 1990s, 2000s)
        let decade = format!("{}s", (record.year / 10) * 10);
        let decade_entry = decade_map.entry(decade).or_insert_with(|| {
            let mut map = HashMap::new();
            for ct in &crime_types {
                map.insert(ct.clone(), 0.0);
            }
            map
        });
        *decade_entry.entry(record.crime_type.clone()).or_insert(0.0) += 1.0;
    }

    // Create decades and feature vectors
    let mut decades = Vec::new();
    let mut feature_vectors = Vec::new();
    for (decade, crime_counts) in decade_map {
        decades.push(decade);
        let mut features = Vec::new();
        for ct in &crime_types {
            features.push(*crime_counts.get(ct).unwrap_or(&0.0));
        }
        feature_vectors.push(features);
    }

    // Convert to DMatrix
    let num_rows = decades.len();
    let num_cols = crime_types.len();
    let flat_data: Vec<f64> = feature_vectors.into_iter().flatten().collect();
    let data = DMatrix::from_row_slice(num_rows, num_cols, &flat_data);

    Ok((decades, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_load_records() {
        let sample_csv = "Record ID,Agency Code,Agency Name,Agency Type,City,State,Year,Month,Incident,Crime Type,Crime Solved,Victim Sex,Victim Age,Victim Race,Victim Ethnicity,Perpetrator Sex,Perpetrator Age,Perpetrator Race,Perpetrator Ethnicity,Relationship,Weapon,Victim Count,Perpetrator Count,Record Source\n1,AK00101,Anchorage,Municipal Police,Anchorage,Alaska,1980,January,1,Murder or Manslaughter,Yes,Male,14,Native American/Alaska Native,Unknown,Male,15,Native American/Alaska Native,Unknown,Acquaintance,Blunt Object,0,0,FBI\n2,AK00101,Anchorage,Municipal Police,Anchorage,Alaska,1980,March,1,Murder or Manslaughter,Yes,Male,43,White,Unknown,Male,42,White,Unknown,Acquaintance,Strangulation,0,0,FBI";
        let path = "test.csv";
        File::create(path)
            .unwrap()
            .write_all(sample_csv.as_bytes())
            .unwrap();

        let records = load_records(path, 1.0, 42).unwrap();
        assert_eq!(records.len(), 2); // 100% sampling

        let records = load_records(path, 0.5, 42).unwrap();
        assert_eq!(records.len(), 1); // 50% sampling

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_load_state_data() {
        let records = vec![
            CrimeRecord {
                state: "Alaska".to_string(),
                year: 1980,
                crime_type: "Murder or Manslaughter".to_string(),
                victim_sex: "Male".to_string(),
                victim_age: 14.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Acquaintance".to_string(),
                weapon: "Blunt Object".to_string(),
            },
            CrimeRecord {
                state: "Alaska".to_string(),
                year: 1980,
                crime_type: "Murder or Manslaughter".to_string(),
                victim_sex: "Male".to_string(),
                victim_age: 43.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Acquaintance".to_string(),
                weapon: "Strangulation".to_string(),
            },
        ];
        let (states, data) = load_state_data(&records).unwrap();
        assert_eq!(states, vec!["Alaska"]);
        assert_eq!(data.nrows(), 1);
        assert_eq!(data.ncols(), 1); // Only one crime type
        assert_eq!(data[(0, 0)], 2.0); // Two incidents of "Murder or Manslaughter"
    }

    #[test]
    fn test_load_age_group_data() {
        let records = vec![
            CrimeRecord {
                state: "Alaska".to_string(),
                year: 1980,
                crime_type: "Murder or Manslaughter".to_string(),
                victim_sex: "Male".to_string(),
                victim_age: 14.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Acquaintance".to_string(),
                weapon: "Blunt Object".to_string(),
            },
            CrimeRecord {
                state: "Alaska".to_string(),
                year: 1980,
                crime_type: "Murder or Manslaughter".to_string(),
                victim_sex: "Male".to_string(),
                victim_age: 43.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Acquaintance".to_string(),
                weapon: "Strangulation".to_string(),
            },
        ];
        let (age_groups, data) = load_age_group_data(&records).unwrap();
        assert_eq!(age_groups, vec!["10-19", "40-49"]);
        assert_eq!(data.nrows(), 2);
        assert_eq!(data.ncols(), 1); // One crime type
        assert_eq!(data[(0, 0)], 1.0); // One incident in "10-19"
        assert_eq!(data[(1, 0)], 1.0); // One incident in "40-49"
    }

    #[test]
    fn test_load_decade_data() {
        let records = vec![
            CrimeRecord {
                state: "Alaska".to_string(),
                year: 1980,
                crime_type: "Murder or Manslaughter".to_string(),
                victim_sex: "Male".to_string(),
                victim_age: 14.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Acquaintance".to_string(),
                weapon: "Blunt Object".to_string(),
            },
            CrimeRecord {
                state: "Alaska".to_string(),
                year: 1985,
                crime_type: "Murder or Manslaughter".to_string(),
                victim_sex: "Male".to_string(),
                victim_age: 43.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Acquaintance".to_string(),
                weapon: "Strangulation".to_string(),
            },
        ];
        let (decades, data) = load_decade_data(&records).unwrap();
        assert_eq!(decades, vec!["1980s"]);
        assert_eq!(data.nrows(), 1);
        assert_eq!(data.ncols(), 1); // One crime type
        assert_eq!(data[(0, 0)], 2.0); // Two incidents in "1980s"
    }
}
