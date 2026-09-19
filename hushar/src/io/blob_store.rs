// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Reading and writing objects, wherever they live.
//!
//! Today that means the local filesystem and Amazon S3. Adding a cloud is one
//! implementation of [`BlobStore`] and one arm of [`BlobStores::open`]; nothing else in
//! the service knows where objects live.
//!
//! # Adding a cloud
//!
//! 1. Implement [`BlobStore`] for it.
//! 2. Add a scheme to [`Location::parse`].
//! 3. Add an arm to [`BlobStores::open`].

use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::io::FileReaderResult;

/// Where an object or a prefix of objects lives.
///
/// # Fields
///
/// * `Local` - a path on the local filesystem
/// * `S3` - a bucket and key; the key may be empty, addressing the bucket root
///
/// Parsed once at startup, so an unusable URI fails before the server binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Location {
    Local(PathBuf),
    S3 { bucket: String, key: String },
}

impl Location {
    /// Parses a URI, treating anything without a recognised scheme as a local path.
    ///
    /// The recognised schemes are `s3://` and `file://`. Bare filesystem paths are
    /// accepted as they are. Only the empty-authority `file:///path` form is a local
    /// file URL.
    ///
    /// # Errors
    ///
    /// Fails on an `s3://` URI with no bucket name, on a `file://host/path` URI, and on
    /// any other scheme — the last naming the schemes that are supported.
    pub(crate) fn parse(uri: &str) -> FileReaderResult<Self> {
        if let Some(rest) = uri.strip_prefix("s3://") {
            let (bucket, key) = rest.split_once('/').unwrap_or((rest, ""));
            if bucket.is_empty() {
                return Err(format!("{uri:?} has no bucket name").into());
            }
            return Ok(Self::S3 {
                bucket: bucket.to_owned(),
                key: key.trim_start_matches('/').to_owned(),
            });
        }

        if let Some(rest) = uri.strip_prefix("file://") {
            let path = rest.strip_prefix('/').map(|p| format!("/{p}"));
            return match path {
                Some(p) => Ok(Self::Local(PathBuf::from(p))),
                None => Err(format!(
                    "{uri:?} is not a local file URL; expected file:///absolute/path"
                )
                .into()),
            };
        }

        if let Some((scheme, _)) = uri.split_once("://") {
            return Err(format!(
                "{uri:?} uses scheme {scheme:?}, which this build does not support. \
                 Supported: s3://, file://, and bare filesystem paths"
            )
            .into());
        }

        Ok(Self::Local(PathBuf::from(uri)))
    }

    /// This location with `suffix` appended, collapsing slashes on either side.
    ///
    /// Used for inference logs, where the configured URI names a prefix and each
    /// batch becomes one object beneath it.
    pub(crate) fn join(&self, suffix: &str) -> Self {
        let suffix = suffix.trim_matches('/');
        match self {
            Self::Local(path) => Self::Local(path.join(suffix)),
            Self::S3 { bucket, key } => Self::S3 {
                bucket: bucket.clone(),
                key: if key.is_empty() {
                    suffix.to_owned()
                } else {
                    format!("{}/{suffix}", key.trim_end_matches('/'))
                },
            },
        }
    }

    /// The location as a URI, for banners and error messages.
    pub(crate) fn uri(&self) -> String {
        match self {
            Self::Local(path) => path.display().to_string(),
            Self::S3 { bucket, key } if key.is_empty() => format!("s3://{bucket}"),
            Self::S3 { bucket, key } => format!("s3://{bucket}/{key}"),
        }
    }
}

/// Somewhere objects can be read and written.
///
/// One implementation per storage system. Implementations are shared across
/// sidecar workers behind an [`Arc`], so every method takes `&self`.
#[tonic::async_trait]
pub(crate) trait BlobStore: Debug + Send + Sync {
    /// Which storage system this is, for the startup banner.
    fn backend(&self) -> &'static str;

    /// Reads the object at `location`.
    async fn get(&self, location: &Location) -> FileReaderResult<Vec<u8>>;

    /// Writes `bytes` to `location`, replacing whatever was there.
    async fn put(&self, location: &Location, bytes: Vec<u8>) -> FileReaderResult<()>;

    /// Reads the object at `location` as UTF-8 text.
    ///
    /// # Errors
    ///
    /// As [`BlobStore::get`], or if the bytes are not valid UTF-8 — invalid bytes are
    /// reported rather than replaced, so a corrupt object is not read as JSON.
    async fn get_string(&self, location: &Location) -> FileReaderResult<String> {
        let bytes = self.get(location).await?;
        String::from_utf8(bytes)
            .map_err(|e| format!("{} is not valid UTF-8: {e}", location.uri()).into())
    }
}

/// The local filesystem.
#[derive(Debug, Default)]
pub(crate) struct LocalBlobStore;

#[tonic::async_trait]
/// Reads and writes on `spawn_blocking`, since filesystem calls are blocking syscalls
/// and these run on async workers.
impl BlobStore for LocalBlobStore {
    fn backend(&self) -> &'static str {
        "local filesystem"
    }

    async fn get(&self, location: &Location) -> FileReaderResult<Vec<u8>> {
        let path = local_path(location)?;
        let owned = path.to_path_buf();
        tokio::task::spawn_blocking(move || std::fs::read(&owned))
            .await
            .map_err(|e| format!("read task for {} did not complete: {e}", location.uri()))?
            .map_err(|e| format!("cannot read {}: {e}", location.uri()).into())
    }

    /// Writes `bytes`, creating any parent directories first, as a time-partitioned log
    /// key needs.
    async fn put(&self, location: &Location, bytes: Vec<u8>) -> FileReaderResult<()> {
        let path = local_path(location)?.to_path_buf();
        let uri = location.uri();
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)
        })
        .await
        .map_err(|e| format!("write task for {uri} did not complete: {e}"))?
        .map_err(|e| format!("cannot write {uri}: {e}").into())
    }
}

/// The location as a filesystem path, or an error naming the mismatch.
fn local_path(location: &Location) -> FileReaderResult<&Path> {
    match location {
        Location::Local(path) => Ok(path),
        other => Err(format!(
            "{} is not a local path; it was routed to the wrong store",
            other.uri()
        )
        .into()),
    }
}

/// Amazon S3.
#[derive(Debug)]
pub(crate) struct S3BlobStore {
    client: Arc<aws_sdk_s3::Client>,
}

impl S3BlobStore {
    pub(crate) fn new(client: Arc<aws_sdk_s3::Client>) -> Self {
        Self { client }
    }
}

#[tonic::async_trait]
impl BlobStore for S3BlobStore {
    fn backend(&self) -> &'static str {
        "Amazon S3"
    }

    async fn get(&self, location: &Location) -> FileReaderResult<Vec<u8>> {
        let (bucket, key) = s3_parts(location)?;
        let response = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                format!(
                    "cannot read {}: {}",
                    location.uri(),
                    crate::io::describe_aws_error(&e)
                )
            })?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| format!("cannot read the body of {}: {e}", location.uri()))?;
        Ok(bytes.into_bytes().to_vec())
    }

    /// Writes `bytes` to the location's key. S3 keys are flat, so there are no parents
    /// to create.
    async fn put(&self, location: &Location, bytes: Vec<u8>) -> FileReaderResult<()> {
        let (bucket, key) = s3_parts(location)?;
        self.client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(aws_sdk_s3::primitives::ByteStream::from(bytes))
            .send()
            .await
            .map_err(|e| {
                format!(
                    "cannot write {}: {}",
                    location.uri(),
                    crate::io::describe_aws_error(&e)
                )
            })?;
        Ok(())
    }
}

/// The location's bucket and key, or an error naming the mismatch.
fn s3_parts(location: &Location) -> FileReaderResult<(&str, &str)> {
    match location {
        Location::S3 { bucket, key } => Ok((bucket, key)),
        other => Err(format!(
            "{} is not an S3 location; it was routed to the wrong store",
            other.uri()
        )
        .into()),
    }
}

/// Resolves locations to the store that serves them.
///
/// The S3 client is built on first use rather than at startup:
/// `aws_config::load_from_env` walks the credential chain, which on a machine with no
/// credentials waits on an instance-metadata probe that times out. A local-only run
/// never builds it.
#[derive(Debug, Default)]
pub(crate) struct BlobStores {
    local: Arc<LocalBlobStore>,
    s3: tokio::sync::OnceCell<Arc<S3BlobStore>>,
}

impl BlobStores {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The store that serves `location`.
    pub(crate) async fn open(&self, location: &Location) -> FileReaderResult<Arc<dyn BlobStore>> {
        match location {
            Location::Local(_) => Ok(Arc::clone(&self.local) as Arc<dyn BlobStore>),
            Location::S3 { .. } => {
                let store = self
                    .s3
                    .get_or_init(|| async {
                        let config = aws_config::load_from_env().await;
                        Arc::new(S3BlobStore::new(Arc::new(aws_sdk_s3::Client::new(&config))))
                    })
                    .await;
                Ok(Arc::clone(store) as Arc<dyn BlobStore>)
            }
        }
    }

    /// Parses `uri` and returns both the location and the store that serves it.
    pub(crate) async fn resolve(
        &self,
        uri: &str,
    ) -> FileReaderResult<(Location, Arc<dyn BlobStore>)> {
        let location = Location::parse(uri)?;
        let store = self.open(&location).await?;
        Ok((location, store))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// An `s3://` URI splits into bucket and key; a bucket with no key is its root.
    #[test]
    fn s3_uris_split_into_bucket_and_key() {
        assert_eq!(
            Location::parse("s3://my-bucket/path/to/file.json").expect("valid"),
            Location::S3 {
                bucket: "my-bucket".into(),
                key: "path/to/file.json".into()
            }
        );
        assert_eq!(
            Location::parse("s3://my-bucket").expect("valid"),
            Location::S3 {
                bucket: "my-bucket".into(),
                key: String::new()
            }
        );
        assert_eq!(
            Location::parse("s3://my-bucket/").expect("valid"),
            Location::S3 {
                bucket: "my-bucket".into(),
                key: String::new()
            }
        );
    }

    #[test]
    fn an_s3_uri_without_a_bucket_is_rejected() {
        assert!(Location::parse("s3://").is_err());
        assert!(Location::parse("s3:///key-with-no-bucket").is_err());
    }
    /// Bare paths and `file:///` URLs both parse to a local location.
    #[test]
    fn bare_paths_and_file_urls_both_mean_the_filesystem() {
        assert_eq!(
            Location::parse("/tmp/model.onnx").expect("valid"),
            Location::Local(PathBuf::from("/tmp/model.onnx"))
        );
        assert_eq!(
            Location::parse("test-data/model.onnx").expect("valid"),
            Location::Local(PathBuf::from("test-data/model.onnx"))
        );
        assert_eq!(
            Location::parse("file:///tmp/model.onnx").expect("valid"),
            Location::Local(PathBuf::from("/tmp/model.onnx"))
        );
    }
    /// An unsupported scheme is refused, naming the schemes that are supported.
    #[test]
    fn an_unsupported_scheme_says_what_is_supported() {
        for uri in ["az://container/key", "gs://bucket/key", "https://host/key"] {
            let err = Location::parse(uri).unwrap_err().to_string();
            assert!(
                err.contains("s3://") && err.contains("file://"),
                "{uri} should list the supported schemes, got: {err}"
            );
        }
    }
    /// Joining a suffix keeps the backend and collapses slashes on either side; a bucket
    /// root gains no leading slash.
    #[test]
    fn joining_a_suffix_keeps_the_backend() {
        let s3 = Location::parse("s3://bucket/logs").expect("valid");
        assert_eq!(
            s3.join("year=2026/x.pb").uri(),
            "s3://bucket/logs/year=2026/x.pb"
        );

        let root = Location::parse("s3://bucket").expect("valid");
        assert_eq!(root.join("a/b.pb").uri(), "s3://bucket/a/b.pb");

        let trailing = Location::parse("s3://bucket/logs/").expect("valid");
        assert_eq!(trailing.join("/a.pb").uri(), "s3://bucket/logs/a.pb");

        let local = Location::parse("/tmp/logs").expect("valid");
        assert_eq!(
            local.join("year=2026/x.pb").uri(),
            "/tmp/logs/year=2026/x.pb"
        );
    }
    /// A location sent to the wrong store is reported rather than panicking.
    #[test]
    fn a_location_sent_to_the_wrong_store_is_reported() {
        let s3 = Location::parse("s3://bucket/key").expect("valid");
        let err = local_path(&s3).unwrap_err().to_string();
        assert!(err.contains("not a local path"), "got {err}");

        let local = Location::parse("/tmp/x").expect("valid");
        let err = s3_parts(&local).unwrap_err().to_string();
        assert!(err.contains("not an S3 location"), "got {err}");
    }
    /// A local object round-trips through a prefix, creating intermediate directories.
    #[tokio::test]
    async fn a_local_object_round_trips() {
        let dir = std::env::temp_dir().join(format!("hushar-blob-{}", uuid::Uuid::new_v4()));
        let stores = BlobStores::new();

        let prefix = Location::parse(dir.to_str().expect("utf-8")).expect("valid");
        let target = prefix.join("year=2026/month=09/batch.pb");
        let store = stores.open(&target).await.expect("local store");
        assert_eq!(store.backend(), "local filesystem");
        store
            .put(&target, b"payload".to_vec())
            .await
            .expect("write");

        assert_eq!(store.get(&target).await.expect("read"), b"payload");
        assert_eq!(store.get_string(&target).await.expect("read"), "payload");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_missing_local_object_names_what_was_looked_for() {
        let stores = BlobStores::new();
        let (location, store) = stores
            .resolve("/tmp/hushar-definitely-not-here.json")
            .await
            .expect("resolves");
        let err = store.get(&location).await.expect_err("no such file");
        assert!(
            err.to_string().contains("hushar-definitely-not-here"),
            "the error should name the object: {err}"
        );
    }

    #[tokio::test]
    async fn invalid_utf8_is_reported_rather_than_replaced() {
        let path = std::env::temp_dir().join(format!("hushar-bad-{}.bin", uuid::Uuid::new_v4()));
        std::fs::write(&path, [0xff, 0xfe, 0xfd]).expect("write");

        let stores = BlobStores::new();
        let (location, store) = stores
            .resolve(path.to_str().expect("utf-8"))
            .await
            .expect("resolves");

        assert_eq!(store.get(&location).await.expect("bytes").len(), 3);
        let err = store
            .get_string(&location)
            .await
            .expect_err("not valid UTF-8");
        assert!(err.to_string().contains("UTF-8"), "got {err}");

        std::fs::remove_file(&path).ok();
    }
    /// Resolving a local path must not initialise the S3 client.
    #[tokio::test]
    async fn a_local_only_run_never_builds_an_aws_client() {
        let stores = BlobStores::new();
        let local = Location::parse("/tmp/whatever").expect("valid");
        let _ = stores.open(&local).await.expect("local store");
        assert!(
            stores.s3.get().is_none(),
            "resolving a local path must not initialise the S3 client"
        );
    }
}
