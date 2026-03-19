# Ablation Results

> This file is auto-populated by `make ablation` → `scripts/run_ablation.py`.
> Run the ablation suite to see real results here.

See `experiments/results/ablation_table.md` for the full generated table.

## Summary Interpretation

| Score | Interpretation |
|-------|---------------|
| 1.0   | Identical to baseline — ablation had no effect |
| < 1.0 | Degraded relative to baseline |
| > 1.0 | Better than baseline (unlikely except by stochasticity) |

## Key Circuits

- **AVB ablation**: Expected to reduce forward locomotion score by >30%.
  AVB is the primary command interneuron for forward movement.

- **AWC ablation**: Expected to reduce chemotaxis score by >40%.
  AWC is the primary olfactory neuron for attractive odours.

- **ASE ablation**: Expected to reduce chemotaxis to water-soluble attractants.

- **AIY ablation**: Disrupts integration of chemosensory signals; expected
  to reduce both locomotion directionality and chemotaxis.

- **AIZ ablation**: Modulates turning behaviour; ablation typically increases
  straight locomotion but reduces chemotaxis efficiency.
