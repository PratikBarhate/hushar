// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use tract_onnx::prelude::*;

pub(crate) mod scoring;
pub(crate) mod transformations;
pub(crate) mod vectorization;

pub(crate) struct InferenceMicros {
    pub vec_time: u128,
    pub tensor_time: u128,
    pub inference_time: u128,
}
pub(crate) type TractRunnableModel =
    RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
