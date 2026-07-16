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
| `demo_alpha_mohawk.png` | `plothead.m` | 3D "mohawk" topograph, alpha band; arcs rise into the crest, scalp coloured by weighted degree, 2 modules = the injected clusters |
| `demo_delta_mohawk.png` | `plothead.m` | 3D mohawk, delta band (no coupling injected) |
| `demo_alpha_network.png` | `plotgraph3d.m` | flat topographic network (supplementary view) |
| `demo_alpha_conn.png` | (connectivity matrix) | alpha dwPLI matrix; uniformly high, zero diagonal |

The 3D mohawk plots use a spherical scalp model coloured by weighted node
degree, with the strongest 30% of within-module edges drawn as arcs rising
above the head — the signature visualisation of Chennu et al. (2017),
reproduced here in Matplotlib.

> These examples use **synthetic** data only. No real or personal EEG recordings
> are included in this repository.
