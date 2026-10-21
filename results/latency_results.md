# Real latency measurements — released single-camera subset

- Generated: `2026-10-21T15:36:28.998478+00:00`
- Command: `python backend/scripts/measure_latency.py`
- Environment: CPU-only (`gpu_available=False`), 12 logical CPUs, Python 3.12.4, PyTorch 2.10.0+cpu, FAISS 1.12.0.
- Subset: 123 clips, 242 scene vectors, 1778 person descriptors, camera_0 only (2026-03-01 17:01–18:56).

Every value below is measured from operations executed by this script. No lognormal/simulated values are used. This is a **reproducibility component measurement**, not a reproduction of the paper's 4-camera / 312k-descriptor latency percentiles.

## Cold loads (in-process, local cache)

| Component | ms | Note |
| --- | --- | --- |
| openclip_vit_b32 | 10386.39 | in-process cold load from local cache; includes module import |
| minilm_l12_v2 | 15953.43 | in-process cold load from local cache; includes module import |
| scene_faiss_index | 0.42 | 242 vectors, d=512 |
| person_faiss_store | 12.39 | 1778 descriptors, d=1152; includes meta.json load |

## Component timings

| Component | n | Mean ms | p50 ms | p95 ms | Min ms | Max ms |
| --- | --- | --- | --- | --- | --- | --- |
| Scene OpenCLIP text encode | 24 | 36.84 | 37.16 | 40.45 | 31.86 | 40.74 |
| Scene FAISS search (242 vectors, k=10) | 24 | 0.07 | 0.07 | 0.09 | 0.06 | 0.11 |
| Scene encode + FAISS | 24 | 36.91 | 37.23 | 40.51 | 31.95 | 40.80 |
| Scene clip aggregation + normalise | 24 | 0.04 | 0.03 | 0.05 | 0.03 | 0.06 |
| Scene production search_unstructured | 12 | 37.16 | 37.08 | 39.40 | 30.52 | 39.57 |
| Scene merge_results (production candidates post-threshold) | 12 | 0.30 | 0.31 | 0.40 | 0.17 | 0.42 |
| Scene merge_results (real top-10 FAISS candidates) | 24 | 0.22 | 0.22 | 0.25 | 0.19 | 0.26 |
| Adaptive alpha computation | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Person MiniLM text encode | 24 | 14.08 | 13.05 | 18.61 | 9.47 | 19.04 |
| Person zero-visual fusion (1152-d) | 24 | 0.04 | 0.04 | 0.06 | 0.03 | 0.08 |
| Person FAISS search (1,778 vectors, k=50) | 24 | 0.89 | 0.68 | 1.71 | 0.50 | 1.75 |
| Person encode + fuse + FAISS | 24 | 15.01 | 13.76 | 19.88 | 10.09 | 19.96 |
| Person search_meta + nearest-clip mapping | 12 | 16.58 | 17.89 | 19.55 | 12.31 | 19.68 |
| Person merge_results (real candidates) | 12 | 0.59 | 0.60 | 0.82 | 0.32 | 0.94 |

Percentiles use NumPy linear interpolation over the measured samples. Scene FAISS is an exact `IndexFlatIP` over 242 vectors; person FAISS over 1,778 descriptors. Text-only person queries use zero visual + MiniLM text, matching Algorithm 2.

## End-to-end component composition

| Composed path | n | Mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- |
| Scene retrieval core (encode+FAISS+aggregate+merge) | 12 | 37.46 | 37.36 | 39.68 |
| Person retrieval core (search_meta+merge) | 12 | 17.17 | 18.53 | 20.35 |

- Scene component sum (encode+FAISS + aggregation + merge over real top-10 FAISS candidates): **37.16 ms** (mean components).
- Scene production sum (post-threshold candidates; several subset queries yield an empty semantic list): **37.25 ms**.
- Person component sum (search_meta + merge): **17.17 ms** (mean components).

Excluded from these numbers: LLM answer generation (live chat answer path); clip stitching / ffmpeg clip generation; MongoDB structured detection retrieval (MongoDB is not running); detection + embedding ingestion (YOLO/OpenVINO/SigLIP indexing); frontend/network round-trip; 4-camera, ~312k-descriptor full deployment; 123-clip camera_0 subset only.

## Concurrency probe (OpenCLIP text encode + scene FAISS)

| Workers | Jobs | Throughput jobs/s | p50 ms | p95 ms | Mean ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 26.52 | 37.35 | 39.56 | 37.54 |
| 2 | 8 | 34.47 | 55.32 | 63.45 | 56.50 |
| 4 | 16 | 39.01 | 93.68 | 117.15 | 95.76 |
| 8 | 32 | 39.47 | 211.50 | 265.43 | 193.55 |

CPU-only, shared model/index. Throughput is end-to-end wall-clock for this job mix and includes CPU contention; it is not a 4-camera ingestion throughput claim.


## Routing, multi-scale retrieval, and generalisation implications

- The released subset is single-camera and short. The measured scene branch uses 512-d frame-level OpenCLIP vectors aggregated to clips; the person branch uses 1152-d (SigLIP 768 + MiniLM 384) descriptors with a text-only query using zero visual, then maps the person timestamp to the nearest released scene clip. These are different index scales and are reported separately, not mixed.
- Person routing is intent-gated (`ENABLE_PERSON_ROUTING=true`): explicit person/attribute/colour queries with no action or count constraint go to the person index; missing/empty person index falls back to the scene index. Action and count queries stay on the scene/structured branches.
- The subset cannot estimate cross-camera domain shift, demographic bias, or 4-camera/312k scaling. Those require the unreleased deployment data or a new labelled capture; this report makes no such claim.
- The per-component numbers are reproducible on CPU from the shipped indices. Full-deployment 6 s/12 s totals remain the paper's primary measurements and are not replaced by this table.
