# Feedback and Suggestions

## Core Framing

The strongest workshop framing is not "we tried many event representations." It is:

1. FRED UAV forecasting has a surprisingly strong optimized Kalman baseline.
2. Extending Kalman from center-only to full center-size boxes already gives strong box metrics.
3. Event-conditioned residual learning improves standard-split box forecasting, but motion-prior shortcuts are a real confound.
4. Correlation analysis, linear residual diagnostics, and decorrelated subsets make the evaluation more honest.

This is a better story than claiming broad SOTA across all metrics. It lets you be precise: likely strong on box mIoU/ADE/FDE versus original FRED, but not necessarily better than recent work on center ADE/FDE.

## Paper Claims To Keep Safe

- Claim "strong box forecasting results" only after checking metric definitions against the original FRED paper.
- Claim "preliminary SOTA on box metrics" only if the table directly supports it.
- Avoid claiming OOD generalization. Say the cutout and decorrelation setup is "OOD-motivated" or "a diagnostic for shortcut reliance."
- Treat CSTR2-vs-event-frame findings as empirical observations, not explained facts.
- Keep the decorrelated split framed as a stricter in-dataset diagnostic, not a replacement for a new collection protocol.

## Highest-Priority Experiments

1. Populate a minimal final baseline table:
   - Last-four CV
   - Optimized Kalman
   - Linear residual center_full
   - Linear residual velocities
   - Best event residual on raw split
   - Best event residual on decorrelated split

2. Run the correlation/decorrelation summary for the exact splits used in the final tables:
   - before/after mean absolute correlation
   - before/after ridge R2
   - before/after mean acceleration norm
   - sample counts

3. Test the smallest image set that can support the main claim:
   - event_frames
   - CSTR2
   - XT+YT
   - CSTR2+XT/YT
   - optional RGB only if time allows

4. Verify comparison to prior work:
   - center ADE/FDE metric definition
   - box ADE/FDE metric definition
   - mIoU horizon and averaging convention
   - whether prior numbers use the same history and forecast horizon

## Useful But Deferrable Ablations

- Cutout sizes other than 64 at resized resolution.
- Temporal bin count and temporal crop length.
- Smaller CNN backbone.
- Kalman covariance features.
- RGB contribution.
- Full factorial representation combinations.
- Multiple random seeds and confidence intervals.

## Writing Notes

- Put the optimized Kalman result early. It motivates the whole residual design.
- Put the linear residual results before image results. They make the shortcut argument concrete.
- Use the acceleration-field plots as explanatory figures, not just diagnostics.
- A good figure sequence would be:
  1. Model diagram: Kalman rollout plus residual acceleration head.
  2. Acceleration field before decorrelation.
  3. Acceleration field after decorrelation.
  4. Forecast examples comparing Kalman, residual, and ground truth.

## Table Plan

Keep tables compact:

- Table 1: Non-image baselines and linear residual diagnostics.
- Table 2: Correlation/decorrelation statistics.
- Table 3: Standard-split image results.
- Table 4: Decorrelated-subset image results.
- Optional Table 5: Prior FRED comparison.

If space is tight, merge Tables 3 and 4 by adding a `Split` or `Decorrelated` column.

## Open Risks

- If decorrelated results are weak, the paper still works as a cautionary analysis plus baseline improvement paper.
- If event images only help on the raw split, avoid saying they learn maneuvers; say they improve under the standard protocol and analyze why decorrelation changes the picture.
- If linear residuals remain strong after decorrelation, emphasize that the decorrelation is imperfect and that complete OOD validation needs a new dataset split or collection protocol.
- If the prior FRED comparison is not apples-to-apples, move it to a short discussion paragraph instead of a headline result.
