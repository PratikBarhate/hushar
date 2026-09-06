/*
 * Copyright (c) 2025 Pratik Barhate
 * Licensed under the MIT License. See the LICENSE file in the project root.
 *
 * bindgen entry point for the AWS Neuron runtime C API.
 *
 * This file only includes headers; it deliberately contains no copy of them.
 * The Neuron SDK headers are distributed by AWS under "All Rights Reserved"
 * terms, so they are read from the SDK installed on the build host rather than
 * vendored into this repository. Point `NEURON_INCLUDE_DIR` at the include
 * directory (default /opt/aws/neuron/include) and build with
 * `--features bindgen`.
 *
 * Verified against aws-neuronx-runtime-lib 2.34.10.0.
 */

/* nrt_version.h uses uint64_t without including <stdint.h> itself. */
#include <stddef.h>
#include <stdint.h>

#include "nrt/nrt_status.h"
#include "nrt/nrt.h"
#include "nrt/nrt_version.h"

/*
 * The tensor description API (nrt_get_model_tensor_info, nrt_tensor_info_t,
 * nrt_tensor_info_array_t) lives in nrt_experimental.h, not nrt.h. It is the
 * only way to learn a loaded NEFF's input and output names, shapes, and dtypes,
 * so this crate depends on it -- and inherits its experimental status.
 */
#include "nrt/nrt_experimental.h"
