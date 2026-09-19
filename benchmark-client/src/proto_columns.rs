// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! The feature payload a request carries: named columns, as protobuf.
//!
//! A request's `features` field is `map<string, Feature>`, one entry per feature, each
//! holding that feature's values for the whole batch. The generated `Feature` is a width
//! and a `oneof` over five typed arrays, which is already exactly what a caller has to
//! produce -- so it is used as it is. There is no client-side mirror of it and nothing to
//! convert between: a generator builds the wire type, [`encode`] checks the set of them
//! agrees, and the map is the request's field.
//!
//! What is left here is the arithmetic that a `Feature` does not do for itself: how many
//! values it holds, how wide a row is, and whether a batch's features agree about the row
//! count.

use std::collections::HashMap;

use crate::hushar_proto::{
    DoubleArray, Feature, FloatArray, IntegerArray, LongArray, StringArray, feature::Data,
};

/// One feature of `width` float values per row.
pub fn floats(width: usize, values: Vec<f32>) -> Feature {
    feature(width, Data::FloatValues(FloatArray { values }))
}

/// One feature of `width` double values per row.
pub fn doubles(width: usize, values: Vec<f64>) -> Feature {
    feature(width, Data::DoubleValues(DoubleArray { values }))
}

/// One feature of `width` 32-bit integer values per row.
pub fn ints(width: usize, values: Vec<i32>) -> Feature {
    feature(width, Data::IntegerValues(IntegerArray { values }))
}

/// One feature of `width` 64-bit integer values per row.
pub fn longs(width: usize, values: Vec<i64>) -> Feature {
    feature(width, Data::LongValues(LongArray { values }))
}

/// One feature of `width` text values per row.
pub fn strings(width: usize, values: Vec<String>) -> Feature {
    feature(width, Data::StringValues(StringArray { values }))
}

/// The wire type, with its width stated.
fn feature(width: usize, data: Data) -> Feature {
    Feature {
        width: u32::try_from(width).unwrap_or(0),
        data: Some(data),
    }
}

/// Values a feature holds, across every row. An entry with no data holds none.
pub fn len(feature: &Feature) -> usize {
    match feature.data.as_ref() {
        Some(Data::FloatValues(a)) => a.values.len(),
        Some(Data::DoubleValues(a)) => a.values.len(),
        Some(Data::IntegerValues(a)) => a.values.len(),
        Some(Data::LongValues(a)) => a.values.len(),
        Some(Data::StringValues(a)) => a.values.len(),
        None => 0,
    }
}

/// Values per row: the width the entry states, or what its values imply over `rows`.
///
/// A sender states its width, so this reads it back rather than recomputing it. An entry
/// that states none -- a minimal client's, or one written before the field existed -- falls
/// back to the same arithmetic the service uses for scores.
pub fn width(feature: &Feature, rows: usize) -> usize {
    let stated = feature.width as usize;
    if stated > 0 {
        return stated;
    }
    len(feature).checked_div(rows).unwrap_or(0)
}

/// Builds the feature map a request's `features` field carries.
///
/// Takes the features by value, so each buffer is moved into the map rather than copied.
///
/// No features is an empty map, which is what a request that sends none looks like.
///
/// # Errors
///
/// Fails if a feature does not carry a whole number of its stated width, or if the features
/// disagree about how many rows they cover -- the service would refuse either, and saying so
/// here names the feature that broke it.
pub fn encode(features: Vec<(String, Feature)>) -> Result<HashMap<String, Feature>, String> {
    let mut covering: Option<(usize, String)> = None;
    for (name, feature) in &features {
        let stated = feature.width as usize;
        if stated == 0 {
            return Err(format!("feature {name:?} states no width"));
        }
        let held = len(feature);
        if !held.is_multiple_of(stated) {
            return Err(format!(
                "feature {name:?} carries {held} value(s), which is not a whole number of \
                 rows at {stated} per row"
            ));
        }
        let rows = held / stated;
        match &covering {
            Some((first, by)) if *first != rows => {
                return Err(format!(
                    "feature {name:?} covers {rows} row(s) but {by:?} covers {first}; one \
                     request is one batch, so every feature covers the same rows"
                ));
            }
            Some(_) => {}
            None => covering = Some((rows, name.clone())),
        }
    }
    Ok(features.into_iter().collect())
}

/// Every feature name, in order.
///
/// A map has none of its own, so a report that lists features does not reorder itself
/// between runs.
pub fn names_in_order(features: &HashMap<String, Feature>) -> Vec<&str> {
    let mut names: Vec<&str> = features.keys().map(String::as_str).collect();
    names.sort_unstable();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scalar_feature_states_a_width_of_one() {
        let wire = encode(vec![
            ("age".to_owned(), floats(1, vec![30.0, 41.0])),
            (
                "city".to_owned(),
                strings(1, vec!["london".into(), "leeds".into()]),
            ),
        ])
        .expect("encodable");
        assert_eq!(wire.len(), 2, "two features, keyed by name");
        assert_eq!(wire["age"].width, 1);
        assert_eq!(len(&wire["age"]), 2, "one value per row over two rows");
        assert_eq!(width(&wire["age"], 2), 1);
    }

    /// The width travels, so it is read back rather than re-derived.
    #[test]
    fn a_wide_feature_states_its_width_on_the_wire() {
        let wire = encode(vec![(
            "embedding".to_owned(),
            floats(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )])
        .expect("encodable");
        assert_eq!(wire["embedding"].width, 3);
        assert_eq!(width(&wire["embedding"], 2), 3, "as stated");
    }

    /// An entry that states no width falls back to the row count.
    #[test]
    fn an_unstated_width_is_recovered_from_the_row_count() {
        let mut f = floats(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        f.width = 0;
        assert_eq!(width(&f, 2), 3, "6 values over 2 rows is 3 per row");
        assert_eq!(width(&f, 0), 0, "no rows leaves nothing to divide by");
    }

    #[test]
    fn every_served_type_reports_its_length() {
        assert_eq!(len(&floats(1, vec![1.0])), 1);
        assert_eq!(len(&doubles(1, vec![0.1, 0.2])), 2);
        assert_eq!(len(&ints(1, vec![1, 2, 3])), 3);
        assert_eq!(len(&longs(1, vec![1, 2])), 2);
        assert_eq!(len(&strings(1, vec!["a".into()])), 1);
        assert_eq!(
            len(&Feature {
                width: 1,
                data: None
            }),
            0,
            "an entry with no data holds no values"
        );
    }

    #[test]
    fn no_features_is_an_empty_map() {
        assert!(encode(Vec::new()).expect("encodable").is_empty());
    }

    /// Columns that cover different numbers of rows are not one batch.
    #[test]
    fn features_that_disagree_about_the_row_count_are_refused_naming_both() {
        let err = encode(vec![
            ("a".to_owned(), floats(1, vec![1.0, 2.0])),
            ("b".to_owned(), floats(1, vec![1.0])),
        ])
        .expect_err("2 rows and 1 row cannot be one batch");
        assert!(err.contains('a') && err.contains('b'), "got {err}");
    }

    #[test]
    fn a_feature_that_is_not_a_whole_number_of_rows_is_refused() {
        let err = encode(vec![("v".to_owned(), floats(3, vec![1.0, 2.0]))])
            .expect_err("2 values is not a whole number of 3s");
        assert!(err.contains('v') && err.contains('2'), "got {err}");
    }

    #[test]
    fn a_feature_that_states_no_width_is_refused() {
        let mut f = floats(1, vec![1.0]);
        f.width = 0;
        let err = encode(vec![("v".to_owned(), f)]).expect_err("a sender states its width");
        assert!(err.contains('v') && err.contains("width"), "got {err}");
    }

    /// A report should list features in the same order every run, since a map has none.
    #[test]
    fn names_come_back_in_order() {
        let wire = encode(vec![
            ("zeta".to_owned(), floats(1, vec![1.0])),
            ("alpha".to_owned(), floats(1, vec![2.0])),
        ])
        .expect("encodable");
        assert_eq!(names_in_order(&wire), ["alpha", "zeta"]);
    }
}
