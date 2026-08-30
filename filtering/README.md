# Uncertainty-gated filtering

This stage gates only the previously computed uncertainty records. A cluster is
accepted only if every enabled condition passes: normalized semantic entropy
is at most its maximum, raw localization variance is at most its maximum, and
persistence is at least its minimum. Rejected clusters are retained with all
applicable reasons for later analysis.

The values in `config/config.yaml` are provisional placeholders (`0.1`,
`500.0`, and `0.8`). They are deliberately not optimized from the test image
and must be selected using validation data in a later research phase.
