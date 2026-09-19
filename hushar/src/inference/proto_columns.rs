// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Reading a request's feature columns out of the wire's feature map.
//!
//! A request carries `map<string, Feature>`: one entry per feature, each holding that
//! feature's values for the whole batch. The service is columnar throughout, so a column
//! arrives as a column and is taken whole -- and taken by *move*, since the decoder
//! already allocated the `Vec` this service wants. Decoding a request therefore copies no
//! feature values at all: [`decode`] rewraps buffers and [`encode`] rewraps them back.
//!
//! Two things the map settles, and one it shares with the model configuration:
//!
//! * **A name travels once.** It is the map key, so it cannot be repeated and cannot
//!   disagree with a name carried inside the value.
//! * **A type is per feature.** An entry is one array, so a column cannot be mixed.
//! * **Width is stated by the caller, and required by the model.** An entry carries the
//!   width the caller believes it is sending, which [`decode`] checks against the values
//!   that arrived with it -- so a caller that contradicts itself is told so, naming the
//!   feature. What the model *requires* is its own configuration's business, checked later
//!   against the input the column feeds. A width of 0 is "not stated", and then only the
//!   weaker check applies: the length must divide by the row count.

use std::collections::HashMap;

use crate::inference::InferenceError;
use crate::inference::batch::FeatureData;
use hushar::hushar_proto::{
    DoubleArray, Feature, FloatArray, IntegerArray, LongArray, StringArray, feature::Data,
};

/// Decodes a request's feature map into the service's own columns, checked against `rows`.
///
/// An empty map is no features, which is how a request that sends none and a log entry
/// that recorded none both read.
///
/// Every value is moved out of the message it arrived in, so this costs one rehash of the
/// keys and no copying of feature data.
///
/// # Errors
///
/// Fails if an entry carries no data, if its stated width disagrees with the values it
/// sent, or -- when no width is stated -- if its length is not a whole number of rows.
pub(crate) fn decode(
    features: HashMap<String, Feature>,
    rows: usize,
) -> Result<HashMap<String, FeatureData>, InferenceError> {
    features
        .into_iter()
        .map(|(name, feature)| {
            let width = feature.width as usize;
            let data = column(&name, feature)?;
            check_shape(&name, &data, rows, width)?;
            Ok((name, data))
        })
        .collect()
}

/// One entry's values, in the service's own types. A move, not a copy.
fn column(name: &str, feature: Feature) -> Result<FeatureData, InferenceError> {
    let data = feature.data.ok_or_else(|| -> InferenceError {
        format!(
            "feature {name:?} carries no values. An entry holds one of float, double, \
             integer, long or string values; a feature a batch has no value for is left \
             out of the map, and its default covers it"
        )
        .into()
    })?;
    Ok(match data {
        Data::FloatValues(a) => FeatureData::F32(a.values),
        Data::DoubleValues(a) => FeatureData::F64(a.values),
        Data::IntegerValues(a) => FeatureData::I32(a.values),
        Data::LongValues(a) => FeatureData::I64(a.values),
        Data::StringValues(a) => FeatureData::Str(a.values),
    })
}

/// Checks a column against the row count, and against the width the caller stated.
///
/// A stated width makes this exact: `rows * width` values, or the caller contradicted
/// itself. Nothing here consults the model, so a failure is a property of the request
/// alone and can be blamed on it without ambiguity -- which is the point of carrying the
/// width at all.
///
/// Unstated, the most that can be required is that the length divides by the row count. A
/// column that fails even that is ragged whichever input it feeds, so saying so here is
/// both earlier and clearer than leaving it to the per-input width check.
fn check_shape(
    name: &str,
    data: &FeatureData,
    rows: usize,
    width: usize,
) -> Result<(), InferenceError> {
    let len = data.len();
    if rows == 0 {
        if len > 0 {
            return Err(format!(
                "feature {name:?} carries {len} value(s), but row_ids states no rows"
            )
            .into());
        }
        return Ok(());
    }
    if width > 0 {
        let wanted = rows * width;
        if len != wanted {
            let implied = if len.is_multiple_of(rows) {
                format!("{}", len / rows)
            } else {
                format!("{len}/{rows}")
            };
            return Err(format!(
                "feature {name:?} states a width of {width}, so it must carry {wanted} \
                 value(s) for {rows} row(s), but it carries {len}, which is {implied} per \
                 row"
            )
            .into());
        }
        return Ok(());
    }
    if !len.is_multiple_of(rows) {
        return Err(format!(
            "feature {name:?} carries {len} value(s), which is not a whole number of \
             values per row for the {rows} row(s) row_ids states"
        )
        .into());
    }
    Ok(())
}

/// Writes feature columns back into the wire's feature map, for the inference log.
///
/// The reverse of [`decode`] and equally free: each buffer is moved back into the message
/// it came from, so recording a request re-wraps its columns rather than rebuilding them.
/// What this yields is what the caller sent -- the named values, before any
/// transformation.
///
/// Each entry's width is restated as `values / rows`. For a caller that stated one, that
/// is the width it stated, since [`decode`] refused the request otherwise; for one that
/// did not, it is the width its values imply. Either way the log carries a width a reader
/// can use without knowing which happened.
pub(crate) fn encode(
    columns: HashMap<String, FeatureData>,
    rows: usize,
) -> HashMap<String, Feature> {
    columns
        .into_iter()
        .map(|(name, data)| {
            let width = data.len().checked_div(rows).unwrap_or(0);
            (name, wrap(data, width))
        })
        .collect()
}

/// One column as the wire carries it. A move, not a copy.
fn wrap(data: FeatureData, width: usize) -> Feature {
    let data = match data {
        FeatureData::F32(values) => Data::FloatValues(FloatArray { values }),
        FeatureData::F64(values) => Data::DoubleValues(DoubleArray { values }),
        FeatureData::I32(values) => Data::IntegerValues(IntegerArray { values }),
        FeatureData::I64(values) => Data::LongValues(LongArray { values }),
        FeatureData::Str(values) => Data::StringValues(StringArray { values }),
    };
    Feature {
        width: u32::try_from(width).unwrap_or(0),
        data: Some(data),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{HashMap, wrap};
    use crate::inference::batch::FeatureData;
    use hushar::hushar_proto::Feature;

    /// One feature as it is sent: its name, its values for the whole batch, and the width
    /// it states.
    pub(crate) type Column = (String, FeatureData, usize);

    /// Builds a feature map from columns, the way a caller would.
    pub(crate) fn features(columns: Vec<Column>) -> HashMap<String, Feature> {
        columns
            .into_iter()
            .map(|(name, data, width)| (name, wrap(data, width)))
            .collect()
    }

    /// One named column of one value per row, the common case.
    pub(crate) fn one(name: &str, data: FeatureData) -> HashMap<String, Feature> {
        features(vec![(name.to_owned(), data, 1)])
    }

    /// One named column of `width` values per row.
    pub(crate) fn wide(name: &str, width: usize, data: FeatureData) -> HashMap<String, Feature> {
        features(vec![(name.to_owned(), data, width)])
    }

    /// One named column that states no width, as a minimal caller would send it.
    pub(crate) fn unstated(name: &str, data: FeatureData) -> HashMap<String, Feature> {
        features(vec![(name.to_owned(), data, 0)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::batch::FeatureData;

    /// Round-trips one column of `rows` rows, one value per row.
    fn round_trip(data: FeatureData, rows: usize) -> FeatureData {
        let sent = test_support::one("f", data);
        let mut decoded = decode(sent, rows).expect("a well-formed map");
        assert_eq!(decoded.len(), 1, "one column in, one column out");
        decoded.remove("f").expect("the column keeps its name")
    }

    #[test]
    fn every_served_element_type_survives_a_round_trip() {
        for data in [
            FeatureData::F32(vec![1.0, 2.0]),
            FeatureData::F64(vec![0.1, 0.2]),
            FeatureData::I32(vec![1, 2]),
            FeatureData::I64(vec![1, 2]),
            FeatureData::Str(vec!["a".to_owned(), "b".to_owned()]),
        ] {
            let element = data.element_type();
            assert_eq!(
                round_trip(data.clone(), 2),
                data,
                "{element} did not survive"
            );
        }
    }

    /// 0.1 has no exact `f32`, so a double column proves nothing narrowed on the way.
    #[test]
    fn a_double_column_keeps_its_precision() {
        let doubles = FeatureData::F64(vec![0.1, 0.2]);
        assert_eq!(round_trip(doubles.clone(), 2), doubles);
    }

    /// A wide feature is the flat row-major buffer; the width states how to read it.
    #[test]
    fn a_wide_feature_arrives_as_one_row_major_buffer() {
        let wide = FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let sent = test_support::wide("v", 3, wide.clone());
        let mut decoded = decode(sent, 2).expect("2 rows of 3 is 6 values");
        assert_eq!(
            decoded.remove("v").expect("v"),
            wide,
            "the values arrive in order, whatever the width says"
        );
    }

    #[test]
    fn a_batch_of_no_features_decodes_as_none() {
        assert!(
            decode(HashMap::new(), 0)
                .expect("empty is a valid absence")
                .is_empty(),
            "no entries is no features, not an error"
        );
    }

    /// The map key is the name and the width is restated, so a log entry reads back as the
    /// request that produced it.
    #[test]
    fn encode_returns_what_decode_was_given() {
        let sent = test_support::features(vec![
            ("age".to_owned(), FeatureData::F32(vec![30.0, 41.0]), 1),
            (
                "embedding".to_owned(),
                FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0]),
                2,
            ),
        ]);
        let logged = encode(decode(sent.clone(), 2).expect("well-formed"), 2);
        assert_eq!(logged, sent, "what arrived is what the log records");
    }

    /// The width the caller stated is what a log reader sees, so a wide feature does not
    /// come back looking scalar.
    #[test]
    fn the_log_restates_each_width() {
        let sent = test_support::wide("v", 4, FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0]));
        let logged = encode(decode(sent, 1).expect("1 row of 4"), 1);
        assert_eq!(
            logged.get("v").expect("v").width,
            4,
            "4 values over 1 row is a width of 4"
        );
    }

    /// A stated width that disagrees with the values sent is the caller contradicting
    /// itself, so it is refused naming the feature, the width and both counts.
    #[test]
    fn a_stated_width_that_disagrees_with_the_values_is_refused() {
        let sent = test_support::wide("v", 3, FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0]));
        let err = decode(sent, 2).expect_err("2 rows of 3 is 6 values, not 4");
        let m = err.to_string();
        assert!(
            m.contains('v') && m.contains('3') && m.contains('6') && m.contains('4'),
            "the error should name the feature, the stated width, the wanted and the \
             actual count: {m}"
        );
    }

    /// Nothing here consults the model, so the check holds for a width the model would
    /// never ask for.
    #[test]
    fn a_stated_width_is_checked_without_consulting_the_model() {
        let ok = test_support::wide("v", 7, FeatureData::F32(vec![0.0; 14]));
        assert!(
            decode(ok, 2).is_ok(),
            "2 rows of 7 is self-consistent, whatever the model wants"
        );
    }

    /// A caller that states no width is held to the weaker check.
    #[test]
    fn an_unstated_width_falls_back_to_the_row_multiple_check() {
        let ok = test_support::unstated("f", FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0]));
        assert!(decode(ok, 2).is_ok(), "4 values over 2 rows is 2 per row");

        let ragged = test_support::unstated("f", FeatureData::F32(vec![1.0, 2.0, 3.0]));
        let err = decode(ragged, 2).expect_err("3 values is not a whole number of 2 rows");
        let m = err.to_string();
        assert!(
            m.contains('f') && m.contains('3') && m.contains('2'),
            "the error should name the feature, its length and the row count: {m}"
        );
    }

    #[test]
    fn values_sent_for_a_batch_of_no_rows_are_refused() {
        let sent = test_support::one("f", FeatureData::F32(vec![1.0]));
        let err = decode(sent, 0).expect_err("no rows takes no values");
        assert!(err.to_string().contains("no rows"), "got {err}");
    }

    /// An entry whose `oneof` was never set names the types it could have carried.
    #[test]
    fn an_entry_with_no_values_is_refused_naming_the_served_types() {
        let sent = HashMap::from([(
            "age".to_owned(),
            Feature {
                width: 1,
                data: None,
            },
        )]);
        let err = decode(sent, 1).expect_err("an empty oneof is not a column");
        let m = err.to_string();
        assert!(
            m.contains("age") && m.contains("float") && m.contains("string"),
            "the error should name the feature and the served types: {m}"
        );
    }
}
