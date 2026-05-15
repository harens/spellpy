# SPELL Parser Findings

## Problem observed

Running SPELL on the raw HDFS_V1 log could pin a CPU core for a long time while the raw cache file stopped growing. The process was spending its time in the parser's fallback matching path, not in GPU work.

The symptoms pointed to a pathological inner loop:

- repeated scans over the full cluster list for each line;
- repeated LCS computation against too many candidates;
- no lightweight progress visibility during long parses;
- no guardrail to stop a single line from triggering excessive LCS work.

## What was changed

The parser implementation in `spellpy/spell.py` was updated to keep the original SPELL flow but avoid the worst-case CPU behavior.

Key changes:

- pre-tokenize each incoming content field once;
- index clusters by constant tokens so candidate matching is no longer a full scan;
- filter LCS candidates with a cheap overlap gate before running LCS;
- keep match indexes in sync when a template is updated;
- add configurable progress logging and periodic cache flushes;
- add an optional `max_lcs_comparisons_per_line` guardrail;
- collect parse metrics for observability.

## Metrics now reported

The parser now records:

- input lines processed;
- templates created;
- total LCS comparisons;
- mean and max candidate templates per line;
- elapsed parse time;
- guardrail skips.

## Paper comparison

The implementation still follows the paper's overall streaming structure:

1. prefix-tree match first;
2. simple-loop match second;
3. LCS fallback only when needed.

The new candidate indexing and guardrail are engineering additions for robustness on large logs. They are not explicit parts of the paper, but they preserve the intended behavior of the parser while preventing pathological CPU blowups.

One existing deviation from the paper remains in `SimpleLoopMatch()`: it uses set inclusion as a cheap subsequence proxy instead of a full pointer-based subsequence walk.

## Verification

Changes were covered with regression and stress-style tests in `tests/test_spellpy.py`.

