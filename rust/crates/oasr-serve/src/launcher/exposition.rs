// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Merging the replicas' Prometheus expositions into one.
//!
//! One scrape of the launcher answers for every replica, each sample labelled
//! with the replica it came from.  Concatenating the texts would not do: the
//! format requires all of a family's samples to follow its one `# TYPE` line,
//! and a second `# TYPE` for a name is a parse error, so the merge regroups by
//! family -- first-seen order, `HELP`/`TYPE` once, then every part's samples.

use std::collections::HashMap;

/// One exposition to merge, and the labels its samples get.  An empty label
/// list passes the samples through unchanged (the launcher's own metrics).
pub struct Part<'a> {
    pub labels: Vec<(&'static str, String)>,
    pub text: &'a str,
}

#[derive(Default)]
struct Family {
    help: Option<String>,
    kind: Option<String>,
    samples: Vec<String>,
}

/// The suffixes a histogram or summary family's samples carry.
const SAMPLE_SUFFIXES: &[&str] = &["_bucket", "_sum", "_count"];

/// `name`'s family, created (and its place in the output taken) on first sight.
fn family<'m>(
    order: &mut Vec<String>,
    families: &'m mut HashMap<String, Family>,
    name: &str,
) -> &'m mut Family {
    if !families.contains_key(name) {
        order.push(name.to_string());
    }
    families.entry(name.to_string()).or_default()
}

pub fn merge(parts: &[Part<'_>]) -> String {
    let mut order: Vec<String> = Vec::new();
    let mut families: HashMap<String, Family> = HashMap::new();

    for part in parts {
        let injected = render_labels(&part.labels);
        // The family the last HELP/TYPE line opened, in this part.
        let mut current: Option<String> = None;
        for line in part.text.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("# HELP ") {
                let name = first_word(rest);
                let f = family(&mut order, &mut families, name);
                f.help.get_or_insert_with(|| line.to_string());
                current = Some(name.to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("# TYPE ") {
                let name = first_word(rest);
                let f = family(&mut order, &mut families, name);
                f.kind.get_or_insert_with(|| line.to_string());
                current = Some(name.to_string());
                continue;
            }
            if line.starts_with('#') {
                continue; // a comment
            }
            let name = sample_name(line);
            let key = match &current {
                Some(c) if belongs_to(name, c) => c.clone(),
                _ => name.to_string(),
            };
            family(&mut order, &mut families, &key)
                .samples
                .push(with_labels(line, name, &injected));
        }
    }

    let mut out = String::new();
    for name in &order {
        let f = &families[name];
        for line in [&f.help, &f.kind].into_iter().flatten() {
            out.push_str(line);
            out.push('\n');
        }
        for s in &f.samples {
            out.push_str(s);
            out.push('\n');
        }
    }
    out
}

fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

/// A sample line's metric name: up to its label set or its value.
fn sample_name(line: &str) -> &str {
    let end = line.find(['{', ' ', '\t']).unwrap_or(line.len());
    &line[..end]
}

fn belongs_to(sample: &str, family: &str) -> bool {
    sample == family
        || sample
            .strip_prefix(family)
            .is_some_and(|rest| SAMPLE_SUFFIXES.contains(&rest))
}

/// `replica="0",gpu="1"` -- escaped as the text format requires.
fn render_labels(labels: &[(&'static str, String)]) -> String {
    labels
        .iter()
        .map(|(k, v)| {
            let v = v
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            format!("{k}=\"{v}\"")
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// `line` with `injected` as the first labels of its label set.  Placed first,
/// the insert never has to find where a label set ends, which a `}` inside a
/// quoted value would make a parsing problem.
fn with_labels(line: &str, name: &str, injected: &str) -> String {
    if injected.is_empty() {
        return line.to_string();
    }
    let rest = &line[name.len()..];
    match rest.strip_prefix('{') {
        Some(after) if after.trim_start().starts_with('}') => format!("{name}{{{injected}{after}"),
        Some(after) => format!("{name}{{{injected},{after}"),
        None => format!("{name}{{{injected}}}{rest}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "\
# HELP oasr_engine_running Requests running
# TYPE oasr_engine_running gauge
oasr_engine_running{engine=\"engine\"} 3

# TYPE oasr_request_duration_seconds histogram
oasr_request_duration_seconds_bucket{api=\"grpc\",le=\"0.1\"} 4
oasr_request_duration_seconds_bucket{api=\"grpc\",le=\"+Inf\"} 5
oasr_request_duration_seconds_sum{api=\"grpc\"} 0.42
oasr_request_duration_seconds_count{api=\"grpc\"} 5
# TYPE oasr_build_info counter
oasr_build_info 1
";

    fn replica<'a>(i: usize, gpu: &str, text: &'a str) -> Part<'a> {
        Part {
            labels: vec![("replica", i.to_string()), ("gpu", gpu.to_string())],
            text,
        }
    }

    #[test]
    fn each_family_keeps_one_type_line_and_all_samples_follow_it() {
        let own =
            "# TYPE oasr_router_replica_ready gauge\noasr_router_replica_ready{replica=\"0\"} 1\n";
        let got = merge(&[
            Part {
                labels: vec![],
                text: own,
            },
            replica(0, "0", A),
            replica(1, "1", A),
        ]);
        let want = "\
# TYPE oasr_router_replica_ready gauge
oasr_router_replica_ready{replica=\"0\"} 1
# HELP oasr_engine_running Requests running
# TYPE oasr_engine_running gauge
oasr_engine_running{replica=\"0\",gpu=\"0\",engine=\"engine\"} 3
oasr_engine_running{replica=\"1\",gpu=\"1\",engine=\"engine\"} 3
# TYPE oasr_request_duration_seconds histogram
oasr_request_duration_seconds_bucket{replica=\"0\",gpu=\"0\",api=\"grpc\",le=\"0.1\"} 4
oasr_request_duration_seconds_bucket{replica=\"0\",gpu=\"0\",api=\"grpc\",le=\"+Inf\"} 5
oasr_request_duration_seconds_sum{replica=\"0\",gpu=\"0\",api=\"grpc\"} 0.42
oasr_request_duration_seconds_count{replica=\"0\",gpu=\"0\",api=\"grpc\"} 5
oasr_request_duration_seconds_bucket{replica=\"1\",gpu=\"1\",api=\"grpc\",le=\"0.1\"} 4
oasr_request_duration_seconds_bucket{replica=\"1\",gpu=\"1\",api=\"grpc\",le=\"+Inf\"} 5
oasr_request_duration_seconds_sum{replica=\"1\",gpu=\"1\",api=\"grpc\"} 0.42
oasr_request_duration_seconds_count{replica=\"1\",gpu=\"1\",api=\"grpc\"} 5
# TYPE oasr_build_info counter
oasr_build_info{replica=\"0\",gpu=\"0\"} 1
oasr_build_info{replica=\"1\",gpu=\"1\"} 1
";
        assert_eq!(got, want);
        assert_eq!(got.matches("# TYPE oasr_engine_running ").count(), 1);
    }

    #[test]
    fn a_family_only_one_replica_has_is_still_grouped() {
        let b = "# TYPE oasr_engine_kv_exhausted_total counter\noasr_engine_kv_exhausted_total{engine=\"e\"} 2\n";
        let got = merge(&[replica(0, "0", A), replica(1, "1", b)]);
        let kv: Vec<&str> = got
            .lines()
            .filter(|l| l.contains("oasr_engine_kv_exhausted_total"))
            .collect();
        assert_eq!(
            kv,
            [
                "# TYPE oasr_engine_kv_exhausted_total counter",
                "oasr_engine_kv_exhausted_total{replica=\"1\",gpu=\"1\",engine=\"e\"} 2",
            ]
        );
    }

    #[test]
    fn empty_label_sets_values_with_braces_and_odd_gpu_names_survive() {
        let text = "x{} 1\ny{path=\"/a}b\"} 2\nz 3 1700000000\n";
        let got = merge(&[replica(0, "GPU-\"q\"", text)]);
        assert_eq!(
            got,
            "x{replica=\"0\",gpu=\"GPU-\\\"q\\\"\"} 1\n\
             y{replica=\"0\",gpu=\"GPU-\\\"q\\\"\",path=\"/a}b\"} 2\n\
             z{replica=\"0\",gpu=\"GPU-\\\"q\\\"\"} 3 1700000000\n"
        );
    }

    #[test]
    fn a_counter_whose_name_extends_another_family_is_its_own_family() {
        // `_total` is not a histogram suffix: it must not be filed under `x`.
        let text = "# TYPE x gauge\nx 1\n# TYPE x_total counter\nx_total 2\n";
        let got = merge(&[replica(0, "0", text), replica(1, "0", text)]);
        let x_total_type = got.find("# TYPE x_total").unwrap();
        let last_x = got.rfind("\nx{").unwrap();
        assert!(last_x < x_total_type, "{got}");
    }
}
