// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Everything that talks to something outside the process.
//!
//! Three traits are the extension points: [`BlobStore`] for objects, [`MetricsSink`] for
//! timings and [`DataSink`] for the data a request produced. Local and Amazon
//! implementations are provided.
//!
//! Nothing here runs on a thread of its own: both sinks accumulate on the request path
//! and spawn their sends, which is what [`accumulator`] is for.

pub(crate) type FileReaderResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub(crate) mod accumulator;
pub(crate) mod blob_store;
pub(crate) mod data_sink;
pub(crate) mod metrics;
pub(crate) mod model_loader;

pub(crate) use blob_store::{BlobStore, BlobStores, Location};
pub(crate) use data_sink::{DataDestination, DataSink, Sampler};
pub(crate) use metrics::MetricsSink;
pub(crate) use model_loader::*;

/// Renders an AWS SDK error together with its whole source chain.
///
/// The SDK's own `Display` is a bare summary such as "service error"; the reason -- no
/// such key, access denied, no credentials -- sits one or two levels down in
/// [`std::error::Error::source`]. A layer whose message repeats the layer above it is
/// skipped.
pub(crate) fn describe_aws_error(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut cause = error.source();
    while let Some(current) = cause {
        let message = current.to_string();
        if !text.ends_with(&message) {
            text.push_str(": ");
            text.push_str(&message);
        }
        cause = current.source();
    }
    text
}
