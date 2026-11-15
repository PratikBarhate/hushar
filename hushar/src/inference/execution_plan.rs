// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! How a request's rows reach the engine.
//!
//! One decision, taken once at startup from what the backend says it can overlap and what
//! the configuration asks for, so the request path carries no policy of its own. The
//! alternative was for the scoring path to split whenever a mini batch was configured,
//! which is right on a CPU session and queues on a device one.
//!
//! The three plans are mutually exclusive by construction. Cutting a request into mini
//! batches and merging rows across requests are opposite moves: merging fills a call to
//! spread its fixed cost over more rows, and splitting divides a call and pays that cost
//! again per mini batch. A model does one or the other.

use crate::inference::backend::ExecutionProfile;
use crate::inference::scoring::MiniBatch;

/// How a request's rows are cut for the engine.
///
/// Chosen by [`Self::resolve`] at startup and fixed for the process. See the module
/// documentation for why the variants cannot be combined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExecutionPlan {
    /// One mini batch per request: the whole request, one engine call.
    ///
    /// The default, and the right plan for a device already filled by one call.
    WholeRequest,

    /// Mini batches of the configured size, all on the request's leased session.
    ///
    /// Correct where concurrent calls on a session overlap, which is a CPU provider.
    SharedSession { mini_batch: MiniBatch },

    /// One mini batch assembled from the rows of several requests.
    ///
    /// Padded to exactly `max_rows` so the engine is always shown one shape, which is
    /// what keeps a shape-compiling provider from building an engine per observed width.
    MergedRequests {
        max_rows: usize,
        max_delay_micros: u64,
    },
}

impl ExecutionPlan {
    /// Chooses the plan, or refuses a configuration that cannot do what it asks.
    ///
    /// Precedence, highest first:
    ///
    /// 1. `merge` is explicit, so it wins. The configuration has already refused it
    ///    alongside a mini batch.
    /// 2. No mini batch is [`Self::WholeRequest`].
    /// 3. A unit whose concurrent calls overlap is [`Self::SharedSession`].
    /// 4. A unit whose calls serialize, where the model demands a fixed row count, is
    ///    also [`Self::SharedSession`]: the mini batch is the model's call shape and the
    ///    calls simply follow one another.
    /// 5. Anything left asks to split for parallelism on a unit that has none.
    ///
    /// # Arguments
    ///
    /// * `profile` - what one session of this backend overlaps
    /// * `required_rows` - the row count the model demands per call, if any
    /// * `mini_batch` - the model configuration's `mini_batch_size` and `is_fixed`
    /// * `merge` - the model configuration's `merge` block
    /// * `engine` - the backend's name, for a refusal that names the hardware
    ///
    /// # Errors
    ///
    /// Refuses a mini batch on a serializing unit when the model does not demand one: the
    /// mini batches would queue on one stream and each would pay the call's fixed cost
    /// again, which is slower than not splitting at all.
    pub(crate) fn resolve(
        profile: ExecutionProfile,
        required_rows: Option<usize>,
        mini_batch: Option<MiniBatch>,
        merge: Option<crate::config::MergeConfig>,
        engine: &str,
    ) -> Result<Self, String> {
        if let Some(merge) = merge {
            return Ok(Self::MergedRequests {
                max_rows: merge.max_rows,
                max_delay_micros: merge.max_delay_micros,
            });
        }
        let Some(mini_batch) = mini_batch else {
            return Ok(Self::WholeRequest);
        };
        if profile.overlaps() || required_rows.is_some() {
            return Ok(Self::SharedSession { mini_batch });
        }
        Err(format!(
            "model declares \"mini_batch_size\": {} on {engine}, where concurrent calls on one \
             session queue on the device rather than overlapping. The mini batches would run one \
             after another and each would pay the call's fixed cost again, so the split is \
             slower than not splitting. Remove \"mini_batch_size\" to score each request in one \
             call, or configure \"merge\" to fill a call from several requests instead",
            mini_batch.size,
        ))
    }

    /// The mini batch this plan cuts requests into, if it cuts them.
    pub(crate) fn mini_batch(self) -> Option<MiniBatch> {
        match self {
            Self::SharedSession { mini_batch } => Some(mini_batch),
            Self::WholeRequest | Self::MergedRequests { .. } => None,
        }
    }

    /// The plan's name, for the startup banner.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::WholeRequest => "whole request",
            Self::SharedSession { .. } => "mini batches, shared session",
            Self::MergedRequests { .. } => "merged requests",
        }
    }

    /// One line describing what this plan does to a request, for the banner.
    pub(crate) fn describe(self) -> String {
        match self {
            Self::WholeRequest => {
                "a request of any size is one engine call of that many rows".to_owned()
            }
            Self::SharedSession { mini_batch } => format!(
                "rows cut into mini batches of {}, scored at the same time on the leased \
                 session, {}",
                mini_batch.size,
                tail(mini_batch),
            ),
            Self::MergedRequests {
                max_rows,
                max_delay_micros,
            } => format!(
                "rows from concurrent requests merged into one call of {max_rows} rows, \
                 waiting at most {max_delay_micros} us to fill it, padded when it is short"
            ),
        }
    }
}

/// How the last mini batch of a request is handled.
fn tail(mini_batch: MiniBatch) -> &'static str {
    if mini_batch.is_fixed {
        "the last padded up to it"
    } else {
        "the last however many remain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(size: usize) -> Option<MiniBatch> {
        Some(MiniBatch {
            size,
            is_fixed: true,
        })
    }

    fn dynamic(size: usize) -> Option<MiniBatch> {
        Some(MiniBatch {
            size,
            is_fixed: false,
        })
    }

    /// The CPU shape: concurrent calls on one session overlap, so the mini batches share
    /// it.
    #[test]
    fn an_overlapping_unit_shares_its_session() {
        let plan = ExecutionPlan::resolve(
            ExecutionProfile::new(8),
            None,
            dynamic(4),
            None,
            "onnxruntime/CPU",
        )
        .expect("a CPU session overlaps its calls");
        assert_eq!(
            plan,
            ExecutionPlan::SharedSession {
                mini_batch: dynamic(4).expect("set")
            }
        );
        assert_eq!(plan.mini_batch().expect("cuts rows").size, 4);
    }

    /// A device session cannot overlap two calls, and the model does not demand the
    /// split, so it is refused.
    #[test]
    fn a_serializing_unit_refuses_an_optional_split() {
        let err = ExecutionPlan::resolve(
            ExecutionProfile::serialized(),
            None,
            dynamic(25),
            None,
            "onnxruntime/TensorRT",
        )
        .expect_err("a split cannot overlap on one device session");

        assert!(
            err.contains("TensorRT"),
            "the refusal should name the hardware: {err}"
        );
        assert!(
            err.contains("25"),
            "the refusal should quote the configured size: {err}"
        );
        for fix in ["merge", "mini_batch_size"] {
            assert!(
                err.contains(fix),
                "the refusal should name the fix {fix}: {err}"
            );
        }
    }

    /// A pinned graph or a NEFF demands its row count, so the split is the model's own
    /// call shape and must not be refused even where the calls serialize.
    #[test]
    fn a_model_that_demands_a_row_count_is_never_refused() {
        let plan = ExecutionPlan::resolve(
            ExecutionProfile::serialized(),
            Some(50),
            fixed(50),
            None,
            "onnxruntime/TensorRT",
        )
        .expect("a pinned graph has to be called at its own size");
        assert_eq!(
            plan,
            ExecutionPlan::SharedSession {
                mini_batch: fixed(50).expect("set")
            },
            "the calls follow one another, which is the only option the model leaves"
        );
    }

    /// No mini batch is one call per request, whatever the backend.
    #[test]
    fn no_mini_batch_is_one_call_per_request() {
        for profile in [ExecutionProfile::serialized(), ExecutionProfile::new(8)] {
            let plan = ExecutionPlan::resolve(profile, None, None, None, "any").expect("valid");
            assert_eq!(plan, ExecutionPlan::WholeRequest);
            assert!(plan.mini_batch().is_none());
        }
    }

    /// Merging is explicit, so it decides regardless of what the unit overlaps.
    #[test]
    fn merging_wins_wherever_it_is_configured() {
        let plan = ExecutionPlan::resolve(
            ExecutionProfile::serialized(),
            None,
            None,
            Some(crate::config::MergeConfig {
                max_rows: 256,
                max_delay_micros: 300,
            }),
            "onnxruntime/TensorRT",
        )
        .expect("merging needs nothing from the unit");
        assert_eq!(
            plan,
            ExecutionPlan::MergedRequests {
                max_rows: 256,
                max_delay_micros: 300,
            }
        );
        assert!(
            plan.mini_batch().is_none(),
            "merging is the opposite of cutting a request up"
        );
    }

    /// Every plan names itself and says what it does, for the banner.
    #[test]
    fn every_plan_describes_itself() {
        let plans = [
            ExecutionPlan::WholeRequest,
            ExecutionPlan::SharedSession {
                mini_batch: dynamic(4).expect("set"),
            },
            ExecutionPlan::MergedRequests {
                max_rows: 256,
                max_delay_micros: 300,
            },
        ];
        for plan in plans {
            assert!(!plan.name().is_empty());
            assert!(
                plan.describe().len() > 20,
                "{} should explain itself",
                plan.name()
            );
        }
        assert!(
            ExecutionPlan::SharedSession {
                mini_batch: fixed(4).expect("set")
            }
            .describe()
            .contains("padded"),
            "a fixed mini batch should say the last one is padded"
        );
    }
}
