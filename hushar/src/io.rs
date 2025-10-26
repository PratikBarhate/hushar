// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

pub(crate) type FileReaderResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub(crate) mod file_reader;
pub(crate) mod model_loader;
pub(crate) mod side_car;

pub(crate) use file_reader::*;
pub(crate) use model_loader::*;
