// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Which logical CPUs a thread may run on.
//!
//! Thread *counts* decide how much work runs at once; affinity decides *where*. The two
//! are separate problems and this file only does the second. See
//! [`crate::config::service_config::ThreadingConfig`] for the counts.
//!
//! # Why pin at all
//!
//! Without affinity the kernel is free to move a thread between cores, and each move
//! leaves its warm cache behind. That is cheap for a thread that mostly waits and
//! expensive for one running a matmul over a large weight matrix. Pinning also lets two
//! pools be kept apart: the async pool on a couple of cores and the engine on the rest,
//! so a burst of connection handling cannot preempt an inference mid-operator.
//!
//! On a NUMA machine it matters more than that, because a thread moved to another socket
//! reaches its weights across the interconnect.
//!
//! # Linux only, and honest about it
//!
//! `sched_setaffinity` is a Linux interface. This service is deployed on Linux, so that
//! is what is implemented. macOS is a development environment: its thread affinity is
//! advisory, the kernel is free to ignore it, and on Apple Silicon it effectively always
//! does — so rather than call something that does nothing, the non-Linux build compiles
//! the pinning out and the banner says the setting was accepted and not applied.

use std::fmt::Write as _;

/// Parses a CPU list of the form used by `taskset` and cgroups.
///
/// ```text
///   "2-9"        cores 2 through 9 inclusive
///   "0,2,4"      cores 0, 2 and 4
///   "0-3,8-11"   both forms, combined
/// ```
///
/// Returns the cores in ascending order with duplicates removed, so a spec written as
/// two overlapping ranges pins to the union rather than to a list that repeats.
///
/// # Errors
///
/// Reports the offending fragment rather than the whole spec, because a long list is
/// usually wrong in one place.
pub(crate) fn parse_cores(spec: &str) -> Result<Vec<usize>, String> {
    let mut cores = Vec::new();
    for piece in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match piece.split_once('-') {
            None => cores.push(
                piece
                    .parse::<usize>()
                    .map_err(|_| format!("{piece:?} is not a core number"))?,
            ),
            Some((first, last)) => {
                let first: usize = first
                    .trim()
                    .parse()
                    .map_err(|_| format!("{piece:?} does not start with a core number"))?;
                let last: usize = last
                    .trim()
                    .parse()
                    .map_err(|_| format!("{piece:?} does not end with a core number"))?;
                if first > last {
                    return Err(format!("{piece:?} counts downwards"));
                }
                cores.extend(first..=last);
            }
        }
    }
    if cores.is_empty() {
        return Err(format!("{spec:?} names no cores"));
    }
    cores.sort_unstable();
    cores.dedup();
    Ok(cores)
}

/// Renders a core list back into the compact form, for the banner.
pub(crate) fn describe(cores: &[usize]) -> String {
    let mut out = String::new();
    let mut index = 0;
    while index < cores.len() {
        let start = index;
        while index + 1 < cores.len() && cores[index + 1] == cores[index] + 1 {
            index += 1;
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == index {
            let _ = write!(out, "{}", cores[start]);
        } else {
            let _ = write!(out, "{}-{}", cores[start], cores[index]);
        }
        index += 1;
    }
    out
}

/// Splits `cores` into `groups` contiguous slices, for one session pool each.
///
/// Contiguous rather than round-robin on purpose: neighbouring core ids are usually on
/// the same NUMA node and share a cache level, so a pool given `0..8` keeps its threads
/// close to each other and to the memory they touch. Interleaving would spread every pool
/// across the whole machine, which is the arrangement this exists to avoid.
///
/// The remainder is spread over the first slices rather than piled onto the last, so the
/// largest and smallest differ by at most one. Fewer cores than groups yields empty
/// slices for the tail, and an empty slice means "place this one anywhere" -- honest, and
/// better than refusing to start.
pub(crate) fn partition(cores: &[usize], groups: usize) -> Vec<Vec<usize>> {
    if groups == 0 {
        return Vec::new();
    }
    let base = cores.len() / groups;
    let extra = cores.len() % groups;
    let mut slices = Vec::with_capacity(groups);
    let mut at = 0;
    for group in 0..groups {
        let take = base + usize::from(group < extra);
        slices.push(cores[at..at + take].to_vec());
        at += take;
    }
    slices
}

/// ONNX Runtime's `session.intra_op_thread_affinities` value, or `None` when there is
/// nothing to say.
///
/// The format is one entry per intra-op thread **other than the calling thread**, so
/// `intra_op_threads` of `n` needs `n - 1` entries; ONNX Runtime rejects a count that
/// disagrees. Cores are handed out round-robin, which gives one core each when there are
/// at least as many cores as threads and shares them predictably when there are not.
///
/// **The ids are 1-based.** ONNX Runtime enforces `processor_id > 0` and then subtracts one
/// to get the logical processor, so core `0` is written `1`. Emitting the core number
/// directly both shifts every thread down by one core and fails outright the moment core 0
/// is in the list -- `Processor id must start from 1: 0` -- which is a startup panic rather
/// than a warning.
///
/// `None` when the calling thread is the only one (`n <= 1`), since there is no extra
/// thread to place, or when no cores were configured.
pub(crate) fn ort_affinity(cores: &[usize], intra_op_threads: i32) -> Option<String> {
    let extra = usize::try_from(intra_op_threads).ok()?.checked_sub(1)?;
    if extra == 0 || cores.is_empty() {
        return None;
    }
    // The calling thread is pinned separately, by us, so it takes the first core and the
    // pool threads take the rest -- wrapping if the list is shorter than the pool.
    let mut entries = Vec::with_capacity(extra);
    for slot in 0..extra {
        entries.push((cores[(slot + 1) % cores.len()] + 1).to_string());
    }
    Some(entries.join(";"))
}

/// Pins the calling thread to `cores`.
///
/// # Errors
///
/// The message names the syscall and the core list, because the usual cause is a core
/// that does not exist on this host -- a configuration copied from a larger instance.
#[cfg(target_os = "linux")]
pub(crate) fn pin_current_thread(cores: &[usize]) -> Result<(), String> {
    if cores.is_empty() {
        return Ok(());
    }
    // SAFETY: `set` is zeroed before use and only touched through the CPU_SET macros,
    // which bound-check against CPU_SETSIZE. `sched_setaffinity` with pid 0 addresses
    // the calling thread and reads `size` bytes from the pointer we own.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &core in cores {
            if core >= libc::CPU_SETSIZE as usize {
                return Err(format!(
                    "core {core} is beyond this kernel's CPU_SETSIZE of {}",
                    libc::CPU_SETSIZE
                ));
            }
            libc::CPU_SET(core, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(format!(
                "sched_setaffinity to [{}] failed: {}",
                describe(cores),
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

/// Accepts the request and does nothing, on a platform where it would mean nothing.
///
/// Not an error: a developer running the production configuration on a laptop should get
/// the same service, minus a placement the kernel would have ignored. [`SUPPORTED`] is
/// what the banner reports, so the difference is stated rather than silent.
#[cfg(not(target_os = "linux"))]
pub(crate) fn pin_current_thread(_cores: &[usize]) -> Result<(), String> {
    Ok(())
}

/// Whether [`pin_current_thread`] actually places threads on this target.
pub(crate) const SUPPORTED: bool = cfg!(target_os = "linux");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioning_gives_each_group_contiguous_cores() {
        let slices = partition(&(0..8).collect::<Vec<_>>(), 4);
        assert_eq!(slices, vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
    }

    /// Contiguous, not interleaved: neighbouring ids share a NUMA node and a cache level,
    /// and spreading a pool across the machine is the thing this avoids.
    #[test]
    fn a_slice_is_a_run_of_neighbours_not_a_stride() {
        let slices = partition(&(0..192).collect::<Vec<_>>(), 24);
        assert_eq!(slices[0].first(), Some(&0));
        assert_eq!(slices[0].last(), Some(&7));
        assert_eq!(slices[23].first(), Some(&184));
        assert_eq!(slices[23].last(), Some(&191));
    }

    /// The remainder is spread over the first slices, so the largest and the smallest
    /// differ by one rather than the last group taking everything left over.
    #[test]
    fn an_uneven_split_differs_by_at_most_one() {
        let slices = partition(&(0..10).collect::<Vec<_>>(), 4);
        let sizes: Vec<usize> = slices.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![3, 3, 2, 2]);
        assert_eq!(
            slices.concat(),
            (0..10).collect::<Vec<_>>(),
            "every core placed once"
        );
    }

    /// Every core appears exactly once across the slices, whatever the split -- two pools
    /// sharing a core is the contention this exists to remove.
    #[test]
    fn slices_are_disjoint_and_cover_everything() {
        for groups in 1..=17 {
            let slices = partition(&(0..96).collect::<Vec<_>>(), groups);
            let mut seen = slices.concat();
            seen.sort_unstable();
            assert_eq!(seen, (0..96).collect::<Vec<_>>(), "groups = {groups}");
        }
    }

    /// Fewer cores than groups leaves empty slices rather than failing: an empty slice
    /// means "place this one anywhere", which is better than refusing to start.
    #[test]
    fn more_groups_than_cores_leaves_empty_slices() {
        let slices = partition(&[0, 1], 4);
        assert_eq!(slices, vec![vec![0], vec![1], vec![], vec![]]);
    }

    #[test]
    fn no_groups_is_no_slices() {
        assert!(partition(&[0, 1, 2], 0).is_empty());
    }

    #[test]
    fn a_single_core_parses() {
        assert_eq!(parse_cores("3").unwrap(), vec![3]);
    }

    #[test]
    fn a_range_is_inclusive_at_both_ends() {
        assert_eq!(parse_cores("2-5").unwrap(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn a_list_may_mix_ranges_and_singles() {
        assert_eq!(parse_cores("0-2,5,8-9").unwrap(), vec![0, 1, 2, 5, 8, 9]);
    }

    /// Overlapping ranges are a union, not a list that repeats: pinning to the same core
    /// twice is meaningless, and the count is used to report how many cores were taken.
    #[test]
    fn overlapping_ranges_are_deduplicated_and_sorted() {
        assert_eq!(parse_cores("4-6,2-5").unwrap(), vec![2, 3, 4, 5, 6]);
        assert_eq!(parse_cores("9,1,5").unwrap(), vec![1, 5, 9]);
    }

    #[test]
    fn whitespace_is_tolerated() {
        assert_eq!(parse_cores(" 0 - 1 , 4 ").unwrap(), vec![0, 1, 4]);
    }

    #[test]
    fn a_bad_spec_names_the_offending_fragment() {
        for (spec, fragment) in [("0-2,x,5", "\"x\""), ("0-,3", "\"0-\""), ("7-3", "\"7-3\"")] {
            let err = parse_cores(spec).expect_err("should be refused");
            assert!(
                err.contains(fragment),
                "error for {spec:?} should name {fragment}: {err}"
            );
        }
    }

    #[test]
    fn an_empty_spec_is_refused() {
        assert!(parse_cores("").is_err());
        assert!(parse_cores("  ,  ").is_err());
    }

    /// Round-trips through the compact form, since the banner prints it.
    #[test]
    fn describing_a_core_list_collapses_runs() {
        assert_eq!(describe(&[0, 1, 2, 3]), "0-3");
        assert_eq!(describe(&[2, 4, 6]), "2,4,6");
        assert_eq!(describe(&[0, 1, 5, 8, 9, 10]), "0-1,5,8-10");
        assert_eq!(describe(&[7]), "7");
        assert_eq!(describe(&[]), "");
    }

    /// One entry per intra-op thread other than the caller, which is the format ONNX
    /// Runtime checks against the thread count when the session is built.
    #[test]
    fn the_affinity_string_has_one_entry_per_extra_thread() {
        let cores = parse_cores("2-9").unwrap();
        let value = ort_affinity(&cores, 8).expect("8 threads leave 7 extra");
        assert_eq!(value.split(';').count(), 7);
        assert_eq!(
            value, "4;5;6;7;8;9;10",
            "the caller keeps the first core, and the ids are 1-based"
        );
    }

    /// ONNX Runtime enforces `processor_id > 0` and then subtracts one, so the ids in this
    /// string are 1-based. Emitting the core number directly shifts every pool thread down
    /// by one core, and panics at startup as soon as core 0 is in the list.
    #[test]
    fn affinity_ids_are_one_based() {
        assert_eq!(
            ort_affinity(&[0, 1, 2, 3], 4).expect("3 extra threads"),
            "2;3;4",
            "cores 1,2,3 are written 2,3,4"
        );
        // The case that used to fail the build of the session outright: a slice small
        // enough that the round-robin wraps back onto core 0.
        let wrapped = ort_affinity(&[0, 1], 4).expect("3 extra threads");
        assert!(
            !wrapped.split(';').any(|id| id == "0"),
            "never emits 0, got {wrapped}"
        );
        assert_eq!(wrapped, "2;1;2");
    }

    /// Nothing to place when the calling thread is the whole pool, which is the default
    /// configuration -- so the common case sets no entry at all.
    #[test]
    fn one_thread_or_fewer_needs_no_affinity_string() {
        let cores = parse_cores("0-7").unwrap();
        assert_eq!(ort_affinity(&cores, 1), None);
        assert_eq!(ort_affinity(&cores, 0), None);
        assert_eq!(ort_affinity(&[], 8), None, "no cores, nothing to say");
    }

    /// More threads than cores is legal and shares them, rather than failing or
    /// silently dropping the surplus threads.
    #[test]
    fn more_threads_than_cores_wraps_round_robin() {
        let cores = parse_cores("4-5").unwrap();
        let value = ort_affinity(&cores, 5).expect("5 threads leave 4 extra");
        assert_eq!(value, "6;5;6;5", "cores 5,4,5,4 written 1-based");
    }
}
