// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Reads an inference log back and prints it **row by row**.
//!
//! The log is written columnar, one array per feature, so this transposes it on the way
//! out.
//!
//! Takes the log file path, then optionally `--row ROW_ID` or `--feature NAME` to print
//! only that row or that feature. A file holds one `InferenceLogs`; a bare
//! `InferenceLogBatch` is accepted too.
//!
//! ```bash
//! cargo run -p benchmark-client --example inspect_log -- ./logs/<file>
//! cargo run -p benchmark-client --example inspect_log -- ./logs/<file> --row row-37
//! cargo run -p benchmark-client --example inspect_log -- ./logs/<file> --feature age
//! ```

use benchmark_client::hushar_proto::{Feature, feature::Data};
use benchmark_client::hushar_proto::{
    InferenceLogBatch, InferenceLogs, score_type::ScoreType as Scores,
};
use benchmark_client::proto_columns;
use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(
        "usage: inspect_log <file> [--row ROW_ID] [--feature NAME]\n\
         reads a hushar inference log and prints each row's features and scores",
    )?;

    let (mut want_row, mut want_feature) = (None, None);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--row" => want_row = args.next(),
            "--feature" => want_feature = args.next(),
            other => return Err(format!("unknown flag {other:?}").into()),
        }
    }

    let bytes = std::fs::read(&path)?;
    let batches = match InferenceLogs::decode(&bytes[..]) {
        Ok(logs) if !logs.inference_log_batches.is_empty() => logs.inference_log_batches,
        _ => vec![InferenceLogBatch::decode(&bytes[..])?],
    };

    println!("{}: {} batch(es)\n", path, batches.len());
    for batch in &batches {
        report(batch, want_row.as_deref(), want_feature.as_deref());
    }
    Ok(())
}

/// Prints one batch, a row at a time, restricted to `want_row` and `want_feature` when
/// they are given.
///
/// The features are the request's own feature map, each entry stating its own width.
/// Score width is `values / rows`, the same arithmetic the service uses, and values print
/// at whichever precision the log carries.
fn report(batch: &InferenceLogBatch, want_row: Option<&str>, want_feature: Option<&str>) {
    let rows = batch.row_ids.len();
    let scores: Vec<String> = match batch.scores.as_ref().and_then(|s| s.score_type.as_ref()) {
        Some(Scores::FloatScores(a)) => a.values.iter().map(|v| format!("{v:?}")).collect(),
        Some(Scores::DoubleScores(a)) => a.values.iter().map(|v| format!("{v:?}")).collect(),
        None => Vec::new(),
    };
    let score_width = scores.len().checked_div(rows).unwrap_or(0);

    let names = proto_columns::names_in_order(&batch.features);

    println!(
        "request {:?}  model {:?}  {rows} row(s), {} feature(s)",
        batch.request_id,
        batch.model_id,
        names.len()
    );
    if names.is_empty() {
        println!("  (features not logged for this request -- sampling skipped them)");
    }

    for (r, row_id) in batch.row_ids.iter().enumerate() {
        if want_row.is_some_and(|wanted| wanted != row_id) {
            continue;
        }
        println!("  row {row_id:?}");
        for name in &names {
            if want_feature.is_some_and(|wanted| wanted != *name) {
                continue;
            }
            let feature = &batch.features[*name];
            match slice_of(feature, r, rows) {
                Some(values) => println!("    {name:<28} {values}"),
                None => println!("    {name:<28} <no values for this row>"),
            }
        }
        if score_width > 0 {
            let at = r * score_width;
            println!(
                "    {:<28} [{}]",
                "scores",
                scores[at..at + score_width].join(", ")
            );
        }
    }
    println!();
}

/// One row's values of one feature, rendered.
///
/// The width is the one the entry states, so a row's values are exactly the ones it
/// declares. The values are borrowed straight out of the message -- nothing is copied to
/// print them.
///
/// # Returns
///
/// `None` for a row past the end of the column, or for an entry carrying no values.
fn slice_of(feature: &Feature, row: usize, rows: usize) -> Option<String> {
    fn render<T: std::fmt::Debug>(values: &[T]) -> String {
        if values.len() == 1 {
            format!("{:?}", values[0])
        } else {
            format!("{values:?}")
        }
    }
    let width = proto_columns::width(feature, rows);
    let at = row.checked_mul(width)?;
    let end = at.checked_add(width)?;
    Some(match feature.data.as_ref()? {
        Data::FloatValues(a) => render(a.values.get(at..end)?),
        Data::DoubleValues(a) => render(a.values.get(at..end)?),
        Data::IntegerValues(a) => render(a.values.get(at..end)?),
        Data::LongValues(a) => render(a.values.get(at..end)?),
        Data::StringValues(a) => render(a.values.get(at..end)?),
    })
}
