# Example output

Figures produced by the automated Python pipeline on the built-in synthetic
generator (`mohawk.datasets.synthetic_raw`), which injects two
hemisphere-clustered, phase-coupled 10 Hz sources on a 1/f background.

Reproduce with:

```bash
uv run mohawk --demo -o out_demo
```

| File | MATLAB analogue | What it shows |
|---|---|---|
| `demo_spec.png` | `plotftspec.m` | log power spectrum; sharp 10 Hz alpha peak, band markers |
| `demo_alpha_mohawk.png` | `plothead.m` | alpha brain-network topograph; 2 modules = the injected clusters |
| `demo_delta_mohawk.png` | `plothead.m` | delta network; fragmented (no coupling injected) |
| `demo_alpha_conn.png` | (connectivity matrix) | alpha dwPLI matrix; uniformly high, zero diagonal |
