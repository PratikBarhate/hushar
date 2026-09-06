/*
 * Copyright (c) 2025 Pratik Barhate
 * Licensed under the MIT License. See the LICENSE file in the project root.
 *
 * Minimal stand-in for the Linux kernel's <linux/types.h>.
 *
 * The Neuron SDK's ndl/neuron_driver_shared.h includes <linux/types.h> because
 * it shares structs with the neuron kernel driver. That header only exists on
 * Linux, which would otherwise make it impossible to regenerate the bindings
 * anywhere else.
 *
 * build.rs adds this directory with `-idirafter`, so it is searched *after*
 * every real include path: on a Linux host the genuine kernel header always
 * wins, and this file is only ever used when none is installed.
 *
 * The Neuron headers use exactly the six typedefs below (verified against
 * aws-neuronx-runtime-lib 2.34.10.0), and each is a fixed-width alias whose
 * definition is identical to the kernel's.
 */

#pragma once

#include <stdint.h>

typedef uint8_t __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef int32_t __s32;
typedef int64_t __s64;
