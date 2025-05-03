// Module: correlation.rs
// Purpose: Perform correlation and statistical analysis on crime data.

use crate::data_loader::CrimeRecord;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::HashMap;

// Helper function to perform Chi-Square test for two categorical variables
fn chi_square_test(data: &HashMap<(String, String), usize>) -> f64 {
    let mut row_totals: HashMap<String, usize> = HashMap::new();
    let mut col_totals: HashMap<String, usize> = HashMap::new();
    let mut total = 0;

    // Calculate row and column totals
    for ((row, col), &count) in data {
        *row_totals.entry(row.clone()).or_insert(0) += count;
        *col_totals.entry(col.clone()).or_insert(0) += count;
        total += count;
    }

    // Check if there are enough categories for the Chi-Square test
    if row_totals.len() < 2 || col_totals.len() < 2 {
        return 1.0; // Return p-value of 1.0 indicating no significant association
    }

    // Calculate expected frequencies and Chi-Square statistic
    let mut chi_square = 0.0;
    for ((row, col), &observed) in data {
        let row_total = row_totals[row] as f64;
        let col_total = col_totals[col] as f64;
        let expected = (row_total * col_total) / total as f64;
        if expected > 0.0 {
            chi_square += (observed as f64 - expected).powi(2) / expected;
        }
    }

    // Degrees of freedom: (rows - 1) * (cols - 1)
    let df = (row_totals.len() - 1) * (col_totals.len() - 1);
    let chi_dist = ChiSquared::new(df as f64).unwrap();
    1.0 - chi_dist.cdf(chi_square) // p-value
}

/// Analyze correlations between State and other variables (Sex, Relationship, Weapon).
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - Tuple of p-values for each correlation
pub fn analyze_state_correlations(records: &[CrimeRecord]) -> (f64, f64, f64) {
    // State vs Victim Sex
    let mut state_sex: HashMap<(String, String), usize> = HashMap::new();
    // State vs Relationship
    let mut state_rel: HashMap<(String, String), usize> = HashMap::new();
    // State vs Weapon
    let mut state_weapon: HashMap<(String, String), usize> = HashMap::new();

    for record in records {
        *state_sex
            .entry((record.state.clone(), record.victim_sex.clone()))
            .or_insert(0) += 1;
        *state_rel
            .entry((record.state.clone(), record.relationship.clone()))
            .or_insert(0) += 1;
        *state_weapon
            .entry((record.state.clone(), record.weapon.clone()))
            .or_insert(0) += 1;
    }

    let state_sex_p = chi_square_test(&state_sex);
    let state_rel_p = chi_square_test(&state_rel);
    let state_weapon_p = chi_square_test(&state_weapon);

    (state_sex_p, state_rel_p, state_weapon_p)
}

/// Analyze correlations between Age Groups and other variables (Sex, Relationship, Weapon).
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - Tuple of p-values for each correlation
pub fn analyze_age_group_correlations(records: &[CrimeRecord]) -> (f64, f64, f64) {
    let mut age_sex: HashMap<(String, String), usize> = HashMap::new();
    let mut age_rel: HashMap<(String, String), usize> = HashMap::new();
    let mut age_weapon: HashMap<(String, String), usize> = HashMap::new();

    for record in records {
        let age_group = if record.victim_age < 0.0 || record.victim_age >= 100.0 {
            "Unknown".to_string()
        } else {
            format!(
                "{}-{}",
                (record.victim_age / 10.0).floor() * 10.0,
                (record.victim_age / 10.0).floor() * 10.0 + 9.0
            )
        };
        *age_sex
            .entry((age_group.clone(), record.victim_sex.clone()))
            .or_insert(0) += 1;
        *age_rel
            .entry((age_group.clone(), record.relationship.clone()))
            .or_insert(0) += 1;
        *age_weapon
            .entry((age_group, record.weapon.clone()))
            .or_insert(0) += 1;
    }

    let age_sex_p = chi_square_test(&age_sex);
    let age_rel_p = chi_square_test(&age_rel);
    let age_weapon_p = chi_square_test(&age_weapon);

    (age_sex_p, age_rel_p, age_weapon_p)
}

/// Analyze correlations between Decades and other variables (Sex, Relationship, State).
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - Tuple of p-values for each correlation
pub fn analyze_decade_correlations(records: &[CrimeRecord]) -> (f64, f64, f64) {
    let mut decade_sex: HashMap<(String, String), usize> = HashMap::new();
    let mut decade_rel: HashMap<(String, String), usize> = HashMap::new();
    let mut decade_state: HashMap<(String, String), usize> = HashMap::new();

    for record in records {
        let decade = format!("{}s", (record.year / 10) * 10);
        *decade_sex
            .entry((decade.clone(), record.victim_sex.clone()))
            .or_insert(0) += 1;
        *decade_rel
            .entry((decade.clone(), record.relationship.clone()))
            .or_insert(0) += 1;
        *decade_state
            .entry((decade, record.state.clone()))
            .or_insert(0) += 1;
    }

    let decade_sex_p = chi_square_test(&decade_sex);
    let decade_rel_p = chi_square_test(&decade_rel);
    let decade_state_p = chi_square_test(&decade_state);

    (decade_sex_p, decade_rel_p, decade_state_p)
}

/// Analyze correlation between Victim Sex and Perpetrator Sex.
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - p-value for the correlation
pub fn analyze_victim_perpetrator_sex(records: &[CrimeRecord]) -> f64 {
    let mut sex_corr: HashMap<(String, String), usize> = HashMap::new();
    for record in records {
        *sex_corr
            .entry((record.victim_sex.clone(), record.perpetrator_sex.clone()))
            .or_insert(0) += 1;
    }
    chi_square_test(&sex_corr)
}

/// Analyze correlation between Victim Sex and Relationship.
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - p-value for the correlation
pub fn analyze_victim_perpetrator_relationship(records: &[CrimeRecord]) -> f64 {
    let mut sex_rel_corr: HashMap<(String, String), usize> = HashMap::new();
    for record in records {
        *sex_rel_corr
            .entry((record.victim_sex.clone(), record.relationship.clone()))
            .or_insert(0) += 1;
    }
    chi_square_test(&sex_rel_corr)
}

/// Analyze correlation between Relationship and Weapon.
/// Inputs:
/// - records: Vector of CrimeRecord
/// Outputs:
/// - p-value for the correlation
pub fn analyze_relationship_weapon(records: &[CrimeRecord]) -> f64 {
    let mut rel_weapon_corr: HashMap<(String, String), usize> = HashMap::new();
    for record in records {
        *rel_weapon_corr
            .entry((record.relationship.clone(), record.weapon.clone()))
            .or_insert(0) += 1;
    }
    chi_square_test(&rel_weapon_corr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::CrimeRecord;

    #[test]
    fn test_chi_square_test() {
        let mut data: HashMap<(String, String), usize> = HashMap::new();
        data.insert(("A".to_string(), "X".to_string()), 10);
        data.insert(("A".to_string(), "Y".to_string()), 10);
        data.insert(("B".to_string(), "X".to_string()), 30);
        data.insert(("B".to_string(), "Y".to_string()), 50);
        let p_value = chi_square_test(&data);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_analyze_state_correlations() {
        let records = vec![
            CrimeRecord {
                state: "New Mexico".to_string(),
                year: 1988,
                victim_sex: "Male".to_string(),
                victim_age: 26.0,
                perpetrator_sex: "Unknown".to_string(),
                relationship: "Unknown".to_string(),
                weapon: "Rifle".to_string(),
            },
            CrimeRecord {
                state: "Nevada".to_string(),
                year: 1988,
                victim_sex: "Female".to_string(),
                victim_age: 29.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Ex-Wife".to_string(),
                weapon: "Handgun".to_string(),
            },
        ];
        let (sex_p, rel_p, weapon_p) = analyze_state_correlations(&records);
        assert!(sex_p >= 0.0 && sex_p <= 1.0);
        assert!(rel_p >= 0.0 && rel_p <= 1.0);
        assert!(weapon_p >= 0.0 && weapon_p <= 1.0);
    }

    #[test]
    fn test_analyze_age_group_correlations() {
        let records = vec![
            CrimeRecord {
                state: "New Mexico".to_string(),
                year: 1988,
                victim_sex: "Male".to_string(),
                victim_age: 26.0,
                perpetrator_sex: "Unknown".to_string(),
                relationship: "Unknown".to_string(),
                weapon: "Rifle".to_string(),
            },
            CrimeRecord {
                state: "Nevada".to_string(),
                year: 1988,
                victim_sex: "Female".to_string(),
                victim_age: 29.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Ex-Wife".to_string(),
                weapon: "Handgun".to_string(),
            },
        ];
        let (sex_p, rel_p, weapon_p) = analyze_age_group_correlations(&records);
        assert!(sex_p >= 0.0 && sex_p <= 1.0);
        assert!(rel_p >= 0.0 && rel_p <= 1.0);
        assert!(weapon_p >= 0.0 && weapon_p <= 1.0);
    }

    #[test]
    fn test_analyze_decade_correlations() {
        let records = vec![
            CrimeRecord {
                state: "New Mexico".to_string(),
                year: 1988,
                victim_sex: "Male".to_string(),
                victim_age: 26.0,
                perpetrator_sex: "Unknown".to_string(),
                relationship: "Unknown".to_string(),
                weapon: "Rifle".to_string(),
            },
            CrimeRecord {
                state: "Nevada".to_string(),
                year: 1988,
                victim_sex: "Female".to_string(),
                victim_age: 29.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Ex-Wife".to_string(),
                weapon: "Handgun".to_string(),
            },
        ];
        let (sex_p, rel_p, state_p) = analyze_decade_correlations(&records);
        assert!(sex_p >= 0.0 && sex_p <= 1.0);
        assert!(rel_p >= 0.0 && rel_p <= 1.0);
        assert!(state_p >= 0.0 && state_p <= 1.0);
    }

    #[test]
    fn test_analyze_victim_perpetrator_sex() {
        let records = vec![
            CrimeRecord {
                state: "New Mexico".to_string(),
                year: 1988,
                victim_sex: "Male".to_string(),
                victim_age: 26.0,
                perpetrator_sex: "Unknown".to_string(),
                relationship: "Unknown".to_string(),
                weapon: "Rifle".to_string(),
            },
            CrimeRecord {
                state: "Nevada".to_string(),
                year: 1988,
                victim_sex: "Female".to_string(),
                victim_age: 29.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Ex-Wife".to_string(),
                weapon: "Handgun".to_string(),
            },
        ];
        let p_value = analyze_victim_perpetrator_sex(&records);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_analyze_victim_perpetrator_relationship() {
        let records = vec![
            CrimeRecord {
                state: "New Mexico".to_string(),
                year: 1988,
                victim_sex: "Male".to_string(),
                victim_age: 26.0,
                perpetrator_sex: "Unknown".to_string(),
                relationship: "Unknown".to_string(),
                weapon: "Rifle".to_string(),
            },
            CrimeRecord {
                state: "Nevada".to_string(),
                year: 1988,
                victim_sex: "Female".to_string(),
                victim_age: 29.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Ex-Wife".to_string(),
                weapon: "Handgun".to_string(),
            },
        ];
        let p_value = analyze_victim_perpetrator_relationship(&records);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_analyze_relationship_weapon() {
        let records = vec![
            CrimeRecord {
                state: "New Mexico".to_string(),
                year: 1988,
                victim_sex: "Male".to_string(),
                victim_age: 26.0,
                perpetrator_sex: "Unknown".to_string(),
                relationship: "Unknown".to_string(),
                weapon: "Rifle".to_string(),
            },
            CrimeRecord {
                state: "Nevada".to_string(),
                year: 1988,
                victim_sex: "Female".to_string(),
                victim_age: 29.0,
                perpetrator_sex: "Male".to_string(),
                relationship: "Ex-Wife".to_string(),
                weapon: "Handgun".to_string(),
            },
        ];
        let p_value = analyze_relationship_weapon(&records);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }
}
