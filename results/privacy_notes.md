# Privacy, retention, and access-control notes

Generated: 2026-10-21. Scope: the code shipped in this repository, not a claim
about a deployed installation. Privacy and access-control switches default to
the previous open behaviour, so the bundled demo keeps working unless an
operator opts in. Person routing is enabled by default but falls back to the
scene index when the person index is absent or empty.

| Variable | Default | Effect |
| --- | --- | --- |
| `ENABLE_PERSON_ROUTING` | `true` | Route person/attribute queries to the 1152-d person index. |
| `ACCESS_CONTROL_ENABLED` | `false` | Enforce role/camera/zone headers; off preserves the open demo. |
| `AUDIT_MONGO_ENABLED` | `false` | Mirror JSONL audit records into MongoDB. |
| `RETENTION_ENABLED` | `false` | Allow `run_retention_policy()` to inspect/delete media. |
| `RETENTION_HOURS` | `8` | Age cutoff once retention is explicitly enabled. |
| `FACE_BLUR_ENABLED` | `false` | Allow `blur_faces_in_file()` to rewrite a still image. |
| `INCLUDE_GENDER_ATTRIBUTES` | `true` | Emit/retain gender terms in captions and person hits. |
| `DATA_EPOCH_DATE` | `2026-03-01` | Ground relative dates in the archive; blank uses "now". |

## What is implemented

### 1. Role and camera/zone access control — opt-in

`backend/app/core/security.py` provides:

- `Role` = `viewer` < `operator` < `admin`, with `Principal.can()`.
- `Principal` carrying `actor`, `allowed_cameras`, `allowed_zones`.
- `is_camera_allowed` / `is_zone_allowed` / `authorize_camera` /
  `authorize_zone` and `filter_authorized_results` for defence in depth.
- `principal_from_headers`: reads `X-Role`, `X-Actor`,
  `X-Allowed-Cameras`, `X-Allowed-Zones` (header names configurable).
- `AccessControlMiddleware`: when `ACCESS_CONTROL_ENABLED=true`, returns 401
  for a missing/unknown role, 403 for insufficient role or an out-of-grant
  camera/zone, and passes `OPTIONS` plus liveness/docs paths through.
  Read routes require `viewer`, writes `operator`, `/api/settings` writes
  `admin`. Wildcard `*` or an absent scope header means unrestricted;
  a present but unparseable camera list means deny-all.
- `require_role(...)`: FastAPI dependency factory for per-route checks.

`backend/app/main.py` registers the middleware. It is inert by default
(`ACCESS_CONTROL_ENABLED=false`), which is covered by
`test_access_control_disabled_preserves_open_behavior`.

**Deployment contract:** identity is header-based and must be injected by a
trusted reverse proxy that authenticates users and strips client-supplied
copies of these headers. The app does **not** validate bearer tokens, run an
identity provider, or manage credentials. Header trust without such a proxy
would be spoofable, so this is a policy hook, not a complete authn system.

### 2. Audit trail

`audit_log()` appends one JSON object per event to `data/logs/audit.jsonl`
(lock-protected, best-effort, never raises) and can mirror to Mongo when
`AUDIT_MONGO_ENABLED=true`. The middleware records allowed and denied
`/api` and `/media` requests with actor, role, method, path, status, camera,
zone, and duration. Audit logs are themselves personal data: the deployment
must set a retention/access policy for `audit.jsonl`.

### 3. Retention policy — callable, never automatic

`backend/app/core/privacy.py`:

- `prune_old_media(..., dry_run=True)` scans `CLIPS_DIR`, `SNAPSHOTS_DIR`,
  `RECORDINGS_DIR`, reports candidates by category, bytes reclaimable, and
  errors. `dry_run=False` unlinks only expired files; directories are kept.
  Symlinks are not followed.
- `run_retention_policy(dry_run=True)` refuses to inspect anything unless
  `RETENTION_ENABLED=true` (or an explicit `force=True` from an admin tool).
- **Nothing calls retention at startup.** This is deliberate: media deletion
  must be an explicit operator/cron action. `RETENTION_HOURS=8` is only the
  configured window; with `RETENTION_ENABLED=false` no file is touched.

Recommended deployment: run `run_retention_policy(dry_run=True)` on a schedule,
review the report, then enable `dry_run=False` under a systemd timer/cron with
its own audit logging.

### 4. Face blur — optional still-image primitive

`FACE_BLUR_ENABLED=false` by default. `detect_faces` /
`blur_faces_with_count` / `blur_faces_in_file` use OpenCV's bundled Haar
cascade (no download, no GPU). `blur_faces_in_file` refuses to modify a file
while the switch is false unless the caller passes `force=True`, and writes
atomically when replacing in place. Tests cover the switch/skip path, the
no-face path, grayscale/BGR/BGRA input, and file rewrite.

Limitations: Haar is a coarse frontal-face detector and misses profile,
occluded, small, or low-light faces; blur is not a formal anonymisation
guarantee. The helper is **not** wired into the live ingestion/streaming
pipeline or into clip generation — it is a deployment utility for snapshots.
Blurring must happen before any human review or export to be meaningful.

### 5. Gender-off switch

`INCLUDE_GENDER_ATTRIBUTES=true` by default (matching the paper's attribute
set). Setting it to `false`:

- stops `AttributeEncoder.attributes_to_text` emitting gender terms;
- strips gender tokens from person-search `attribute_text` and result
  captions via `strip_gender_from_results`, including nested `person_hits`,
  `frames`, `objects`, and `detection` records.

It does **not** retrain or rewrite the released person FAISS index, whose
stored captions may still contain gender text; the filter is applied on
output. Nor does it change the demographic composition of the training data
or provide a bias audit. Treat it as an output-minimisation control.

## Verification

```
python -m pytest backend/tests -q
# 63 passed
python backend/scripts/measure_latency.py
```

Relevant tests: `backend/tests/test_privacy.py` (role hierarchy, header
parsing, middleware 401/403/open-by-default, JSONL audit, retention
dry-run/real/switch, face blur, gender-off) and
`backend/tests/test_person_routing.py` (person routing, fallback, parser
grounding, fast Mongo import).

## What cannot be claimed from this repository

- **No demographic-bias measurement.** The released subset has no
  demographic labels; reporting accuracy/false-positive rates by gender,
  skin tone, or age would require labelled data and an IRB-style protocol.
- **No human inter-annotator agreement.** No second annotator or adjudication
  was run for this revision.
- **No full-deployment privacy/latency result.** The shipped indices are a
  single camera (`camera_0`), 123 clips, 242 scene vectors, 1,778 person
  descriptors. The paper's 4-camera / ~312k-descriptor deployment is not
  reproduced here, and retention/blur/access-control behaviour at that scale
  is untested.
- **No legal compliance claim.** Retention, audit, and access control are
  technical primitives; GDPR/DPDP/legal basis, consent, DPIA, and retention
  approval remain deployment responsibilities.
