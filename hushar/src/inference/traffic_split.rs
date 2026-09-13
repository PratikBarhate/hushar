// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Which of two models serves a request.
//!
//! A roll-out has a control and a candidate, and moves traffic from one to the other a
//! few percent at a time. This decides, per request, which one it is.
//!
//! Two ways in, and they are not equal. The caller may name a model, which wins
//! outright; otherwise the configured percentage decides. Naming it is for when the arm
//! was already chosen elsewhere — the experiment bucket a user belongs to — and the
//! service must honour that rather than re-roll it.

use std::sync::atomic::{AtomicU64, Ordering};

/// Which arm serves a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Arm {
    /// The model named by `model_config_path`.
    Control,
    /// The model named by `candidate_model.config_path`.
    Candidate,
}

/// Splits traffic between the control and the candidate.
///
/// # Counted, not random
///
/// The same reasoning as [`Sampler`](crate::io::data_sink::Sampler): a counter needs no
/// random-number generator on the request path, and it delivers the ratio exactly rather
/// than approaching it. A 10% candidate gets 10 of every 100 requests, not 10% in
/// expectation — which matters when the two arms' latencies are being compared and a
/// short run would otherwise carry a sampling error nobody accounted for.
///
/// Whole percent, so there is no rounding to explain. `Sampler` realises a rate as one
/// request in `n` and reports the nearest unit fraction it could manage; a split cannot
/// afford that, because 30% becoming 33% would quietly bias a comparison.
///
/// # Not sticky
///
/// The arm depends on arrival order, not on anything in the request, so the same caller
/// asking twice may be served by both. That is right for measuring the arms against each
/// other and wrong for an experiment that needs a subject held to one arm. An experiment
/// like that should decide on its own side and name the model in the request.
#[derive(Debug)]
pub(crate) struct TrafficSplit {
    /// Percent of requests the candidate serves, `0..=100`.
    candidate_percent: u8,
    seen: AtomicU64,
}

impl TrafficSplit {
    /// `candidate_percent` is the share the candidate serves, `0..=100`.
    ///
    /// Values above 100 are treated as 100. The configuration refuses them before this
    /// is reached, so this is a total function rather than a second line of defence.
    pub(crate) fn new(candidate_percent: u8) -> Self {
        Self {
            candidate_percent: candidate_percent.min(100),
            seen: AtomicU64::new(0),
        }
    }

    /// Everything to the control, for a service with no candidate.
    pub(crate) fn control_only() -> Self {
        Self::new(0)
    }

    /// The arm for the next request.
    ///
    /// One relaxed increment in the split case and none at either extreme, so a
    /// service that is not mid-roll-out pays nothing for this.
    ///
    /// # Spread, not clumped
    ///
    /// The obvious counter — `seen % 100 < percent` — is exact but front-loads: the
    /// candidate takes positions 0..percent of every hundred and the control the rest.
    /// Multiplying first spreads them instead, so at 30% the candidate takes roughly
    /// every third request rather than the first thirty of each hundred. Both deliver
    /// the same count; only this one keeps the arms from correlating with anything
    /// periodic in the arrival pattern, which would otherwise show up as a latency
    /// difference that is really a difference in when each arm was asked.
    pub(crate) fn choose(&self) -> Arm {
        match self.candidate_percent {
            0 => Arm::Control,
            100 => Arm::Candidate,
            percent => {
                let percent = u64::from(percent);
                let position = self.seen.fetch_add(1, Ordering::Relaxed);
                // Error diffusion: the product's residue crosses below `percent` exactly
                // `percent` times per hundred, whatever the percentage.
                if position.wrapping_mul(percent) % 100 < percent {
                    Arm::Candidate
                } else {
                    Arm::Control
                }
            }
        }
    }

    /// What was configured, for the startup banner.
    pub(crate) fn describe(&self) -> String {
        match self.candidate_percent {
            0 => "no traffic".to_owned(),
            100 => "all traffic".to_owned(),
            percent => format!("{percent}% of traffic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts the arms over `n` requests.
    fn tally(split: &TrafficSplit, n: usize) -> (usize, usize) {
        let mut control = 0;
        let mut candidate = 0;
        for _ in 0..n {
            match split.choose() {
                Arm::Control => control += 1,
                Arm::Candidate => candidate += 1,
            }
        }
        (control, candidate)
    }

    /// Exact over 100, which is the property a counter buys over a random choice.
    #[test]
    fn the_split_is_exact_over_a_hundred_requests() {
        for percent in [1, 10, 30, 50, 70, 99] {
            let split = TrafficSplit::new(percent);
            let (control, candidate) = tally(&split, 100);
            assert_eq!(
                candidate, percent as usize,
                "{percent}% should be exactly {percent} of 100, got {candidate}"
            );
            assert_eq!(control, 100 - percent as usize);
        }
    }

    /// And stays exact across windows, rather than drifting.
    #[test]
    fn the_split_stays_exact_over_many_windows() {
        let split = TrafficSplit::new(30);
        let (control, candidate) = tally(&split, 10_000);
        assert_eq!(candidate, 3_000);
        assert_eq!(control, 7_000);
    }

    /// The reason for multiplying rather than taking the remainder directly. A
    /// front-loaded split would put every candidate request in the first third of each
    /// hundred, so the arms would differ in *when* they were asked as well as in which
    /// model answered -- and a periodic arrival pattern would then read as a latency
    /// difference between the models.
    #[test]
    fn the_candidate_is_spread_through_the_window_not_clumped() {
        let split = TrafficSplit::new(25);
        let arms: Vec<Arm> = (0..100).map(|_| split.choose()).collect();

        // Exact overall.
        assert_eq!(arms.iter().filter(|a| **a == Arm::Candidate).count(), 25);

        // And present in every quarter, which a front-loaded split would fail: it
        // would put all 25 in the first quarter and none in the other three.
        for (quarter, chunk) in arms.chunks(25).enumerate() {
            let candidates = chunk.iter().filter(|a| **a == Arm::Candidate).count();
            assert!(
                candidates > 0,
                "quarter {quarter} had no candidate requests, so the split is clumped"
            );
        }

        // The longest run of consecutive control requests stays small; a clumped split
        // would show a run of 75.
        let longest = arms
            .split(|arm| *arm == Arm::Candidate)
            .map(<[Arm]>::len)
            .max()
            .unwrap_or(0);
        assert!(
            longest <= 4,
            "longest control run was {longest}, expected a few"
        );
    }

    /// The ends of a roll-out. Both take a branch that never touches the counter.
    #[test]
    fn the_extremes_send_everything_one_way() {
        let (control, candidate) = tally(&TrafficSplit::new(0), 50);
        assert_eq!((control, candidate), (50, 0), "0% is control only");

        let (control, candidate) = tally(&TrafficSplit::new(100), 50);
        assert_eq!((control, candidate), (0, 50), "100% is candidate only");
    }

    #[test]
    fn a_service_without_a_candidate_never_chooses_one() {
        let (control, candidate) = tally(&TrafficSplit::control_only(), 20);
        assert_eq!((control, candidate), (20, 0));
    }

    /// Clamped rather than wrapped, so a percentage that escaped validation cannot
    /// turn into a small one by arithmetic.
    #[test]
    fn a_percentage_above_a_hundred_is_treated_as_all() {
        assert_eq!(TrafficSplit::new(200).choose(), Arm::Candidate);
        assert_eq!(TrafficSplit::new(u8::MAX).describe(), "all traffic");
    }

    /// The banner prints this, so it has to read as what was configured.
    #[test]
    fn the_description_says_what_was_configured() {
        assert_eq!(TrafficSplit::new(0).describe(), "no traffic");
        assert_eq!(TrafficSplit::new(10).describe(), "10% of traffic");
        assert_eq!(TrafficSplit::new(100).describe(), "all traffic");
    }
}
