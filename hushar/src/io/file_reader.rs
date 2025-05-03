// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! File Reader module that abstracts reading files from different storage types.
//!
//! This module provides functionality to read files from either local filesystem
//! or an S3 bucket through a unified trait interface.

use std::fs::File;
use std::future::Future;
use std::io::Read;
use std::pin::Pin;
use std::sync::Arc;

use crate::io::FileReaderResult;

/// Trait defining the interface for reading files from different storage types
pub trait FileReader {
    /// Reads the content of a file as a String
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file (format depends on the implementation)
    ///
    /// # Returns
    ///
    /// The content of the file as a String if successful
    fn read_string<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<String>> + Send + 'a>>;

    /// Reads the content of a file as bytes
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file (format depends on the implementation)
    ///
    /// # Returns
    ///
    /// The content of the file as a Vec<u8> if successful
    fn read_bytes<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<Vec<u8>>> + Send + 'a>>;
}

/// Reader for local filesystem files
#[derive(Debug)]
pub struct LocalReader;

#[allow(dead_code)]
impl LocalReader {
    /// Creates a new LocalReader
    pub fn new() -> Self {
        LocalReader
    }
}

impl FileReader for LocalReader {
    fn read_string<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let mut file = File::open(path)?;
            let mut content = String::new();
            file.read_to_string(&mut content)?;
            Ok(content)
        })
    }

    fn read_bytes<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let mut file = File::open(path)?;
            let mut content = Vec::new();
            file.read_to_end(&mut content)?;
            Ok(content)
        })
    }
}

/// Reader for files stored in S3
#[derive(Debug)]
pub struct S3Reader {
    client: Arc<aws_sdk_s3::Client>,
}

#[allow(dead_code)]
impl S3Reader {
    /// Creates a new S3Reader
    ///
    /// # Arguments
    ///
    /// * `client` - An initialized AWS S3 client
    pub fn new(client: aws_sdk_s3::Client) -> Self {
        S3Reader {
            client: Arc::new(client),
        }
    }

    /// Downloads a file from S3 to the local filesystem
    ///
    /// # Arguments
    ///
    /// * `s3_path` - Path in the format "s3://bucket-name/key/path"
    /// * `local_path` - Path where the file should be saved locally
    ///
    /// # Returns
    ///
    /// Ok(()) if the download was successful
    pub fn download_to_file<'a>(
        &'a self,
        s3_path: &'a str,
        local_path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let bytes = self.read_bytes(s3_path).await?;
            let mut file = File::create(local_path)?;
            use std::io::Write;
            file.write_all(&bytes)?;
            Ok(())
        })
    }

    /// Parses an S3 path in the format "s3://bucket-name/key/path"
    ///
    /// # Arguments
    ///
    /// * `s3_path` - Path in the format "s3://bucket-name/key/path"
    ///
    /// # Returns
    ///
    /// A tuple (bucket, key) if the path is valid
    fn parse_s3_path(s3_path: &str) -> FileReaderResult<(String, String)> {
        if !s3_path.starts_with("s3://") {
            return Err("S3 path must start with s3://".into());
        }

        let path = &s3_path[5..]; // Remove "s3://"
        let mut parts = path.splitn(2, '/');

        let bucket = parts
            .next()
            .ok_or_else(|| "Invalid S3 path format, missing bucket name")?;
        let key = parts
            .next()
            .ok_or_else(|| "Invalid S3 path format, missing key")?;

        Ok((bucket.to_string(), key.to_string()))
    }
}

impl FileReader for S3Reader {
    fn read_string<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<String>> + Send + 'a>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            let (bucket, key) = Self::parse_s3_path(path)?;
            let s3_response = client.get_object().bucket(bucket).key(key).send().await?;
            let body = s3_response.body;
            let bytes = body.collect().await?;
            let content = String::from_utf8(bytes.into_bytes().to_vec())?;
            Ok(content)
        })
    }

    fn read_bytes<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = FileReaderResult<Vec<u8>>> + Send + 'a>> {
        let client = Arc::clone(&self.client);

        Box::pin(async move {
            let (bucket, key) = Self::parse_s3_path(path)?;
            let s3_response = client.get_object().bucket(bucket).key(key).send().await?;
            let body = s3_response.body;
            let bytes = body.collect().await?;
            Ok(bytes.into_bytes().to_vec())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;

    // Helper function to create a temporary file with content
    fn create_temp_file(content: &str, file_number: u16) -> (PathBuf, String) {
        let temp_dir = env::temp_dir();
        let file_name = format!("test_file_{}.txt", file_number);
        let file_path = temp_dir.join(file_name);

        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        (file_path, content.to_string())
    }

    #[tokio::test]
    async fn test_local_reader_read_string() {
        let (file_path, expected_content) = create_temp_file("Hello, world!", 1);

        let reader = LocalReader::new();
        let content = reader
            .read_string(file_path.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(content, expected_content);

        // Clean up
        fs::remove_file(file_path).unwrap();
    }

    #[tokio::test]
    async fn test_local_reader_read_bytes() {
        let (file_path, expected_content) = create_temp_file("Bytes test", 2);

        let reader = LocalReader::new();
        let bytes = reader
            .read_bytes(file_path.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(bytes, expected_content.as_bytes());

        // Clean up
        fs::remove_file(file_path).unwrap();
    }

    #[tokio::test]
    async fn test_local_reader_file_not_found() {
        let reader = LocalReader::new();
        let result = reader.read_string("non_existent_file.txt").await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_s3_parse_path() {
        let valid_path = "s3://my-bucket/path/to/file.txt";
        let (bucket, key) = S3Reader::parse_s3_path(valid_path).unwrap();

        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.txt");

        let invalid_path = "invalid-path";
        let result = S3Reader::parse_s3_path(invalid_path);
        assert!(result.is_err());
    }
}
