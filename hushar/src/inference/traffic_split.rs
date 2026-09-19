// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Which of two models serves a request.
//!
//! A roll-out has a control and a candidate, and moves traffic from one to the other a
//! few percent at a time. A request naming a model is served by that model; otherwise
//! the configured percentage decides.

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
/// The share is counted rather than sampled, so it is delivered exactly: a candidate set
/// to 10% serves 10 of every 100 requests. Granularity is a whole percent.
///
/// The arm depends on arrival order, not on anything in the request, so the same caller
/// asking twice may be served by both. A caller that needs one subject held to one arm
/// must choose the arm itself and name the model in the request.
#[derive(Debug)]
pub(crate) struct TrafficSplit {
    /// Percent of requests the candidate serves, `0..=100`.
    candidate_percent: u8,
    seen: AtomicU64,
}

impl TrafficSplit {
    /// A split in which the candidate serves `candidate_percent` of requests.
    ///
    /// # Arguments
    ///
    /// * `candidate_percent` - the candidate's share, `0..=100`; a larger value is
    ///   clamped to 100
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
    /// Exactly `candidate_percent` of every hundred requests go to the candidate, spread
    /// through the hundred rather than taken from its front.
    ///
    /// Safe to call concurrently.
    pub(crate) fn choose(&self) -> Arm {
        match self.candidate_percent {
            0 => Arm::Control,
            100 => Arm::Candidate,
            percent => {
                let percent = u64::from(percent);
                let position = self.seen.fetch_add(1, Ordering::Relaxed);
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

    /// Each configured percentage yields exactly that many candidates per 100 requests.
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

    /// The split stays exact over many windows, rather than drifting.
    #[test]
    fn the_split_stays_exact_over_many_windows() {
        let split = TrafficSplit::new(30);
        let (control, candidate) = tally(&split, 10_000);
        assert_eq!(candidate, 3_000);
        assert_eq!(control, 7_000);
    }

    /// Candidate requests appear in every quarter of a window, with only short control
    /// runs between them.
    #[test]
    fn the_candidate_is_spread_through_the_window_not_clumped() {
        let split = TrafficSplit::new(25);
        let arms: Vec<Arm> = (0..100).map(|_| split.choose()).collect();

        assert_eq!(arms.iter().filter(|a| **a == Arm::Candidate).count(), 25);

        for (quarter, chunk) in arms.chunks(25).enumerate() {
            let candidates = chunk.iter().filter(|a| **a == Arm::Candidate).count();
            assert!(
                candidates > 0,
                "quarter {quarter} had no candidate requests, so the split is clumped"
            );
        }

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

    /// A 0% split serves only the control and a 100% split only the candidate.
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

    /// A percentage above 100 is clamped to 100 rather than wrapped.
    #[test]
    fn a_percentage_above_a_hundred_is_treated_as_all() {
        assert_eq!(TrafficSplit::new(200).choose(), Arm::Candidate);
        assert_eq!(TrafficSplit::new(u8::MAX).describe(), "all traffic");
    }

    /// `describe` reads as the configured share, for the banner.
    #[test]
    fn the_description_says_what_was_configured() {
        assert_eq!(TrafficSplit::new(0).describe(), "no traffic");
        assert_eq!(TrafficSplit::new(10).describe(), "10% of traffic");
        assert_eq!(TrafficSplit::new(100).describe(), "all traffic");
    }
}
