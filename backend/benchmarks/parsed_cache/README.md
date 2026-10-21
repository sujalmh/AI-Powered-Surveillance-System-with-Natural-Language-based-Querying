# Parse cache (build artifact)

This directory holds the on-disk parse cache of the **end-to-end** track
(`backend/scripts/eval_end_to_end.py`).  It is a build artifact: it is
written and read by the evaluation run, it is not a benchmark input, and it
is not used by `build_subset.py` or `build_subset_control.py`.  The benchmark
inputs remain `../queries.jsonl` and `../splits.json`.

The parser track (`backend/scripts/eval_parser.py`) uses the same format in
`../parser_cache/`; both directories are excluded from the release archive.

## Why it exists

The natural-language parse is the one non-deterministic step of the released
track: the answer depends on the model, the provider and the provider's quota
at the time of the run.  Caching the parse makes the run

*   **reproducible** — a re-run reuses the stored filters and produces the
    same ranking without calling the model; and
*   **resumable** — the cache is saved after every query, so a run that is
    stopped by a provider usage-limit error can be continued later.

## Format

One file per model id.  The file name is the model id with `/` and `:` replaced
by `_`, for example `opencode-go_beam-free.json`.

The file is a single JSON object keyed by the query id and a digest of the
query text and the model (`<qid>:<sha1-12>`), so a stale parse is never reused
after the benchmark is rebuilt or the model changes:

```json
{
  "q_count_01:3f0c1d2ab9e4": {
    "parsed": {
      "timestamp": {"$gte": "2026-03-01T17:00:00", "$lte": "2026-03-01T17:02:00"},
      "objects.object_name": "person",
      "query_type": "visual"
    },
    "ok": true,
    "seconds": 7.69
  }
}
```

| Field | Meaning |
|---|---|
| `<qid>:<digest>` | query id from `../queries.jsonl` plus the first 12 hex characters of the SHA-1 of the query text and the model id |
| `parsed` | the parsed filter in the backend filter shape (`timestamp`, `camera_id`, `objects.object_name`, `objects.color`, `zone`, `action`, `count_constraint`, `query_type`) |
| `ok` | `true` when the LLM answered; `false` when the parse failed and the deterministic heuristic fallback was used |
| `seconds` | wall-clock duration of that parse |
| `error` | present only when `ok` is `false`; the truncated exception text |

The cache is written by `backend/app/services/llm_client.ParseCache` (read /
write) and by `backend/scripts/eval_end_to_end.py` (one `save()` after every
query).  The parser track (`../parser_cache/`) uses the same value shape but
keys it by the plain query id.

## Regenerating it

```powershell
$env:PYTHONIOENCODING='utf-8'
python backend/scripts/eval_end_to_end.py            # fills the cache for the default model
python backend/scripts/eval_end_to_end.py --refresh  # re-parses and overwrites the cached entries
```

Deleting the file(s) and running the script again has the same effect as
`--refresh`.  A run that is stopped by a provider usage-limit error exits
with code 2 after the cache has been saved; running the script again resumes
at the first query that has no cache entry.

The opencode provider parses with `OPENCODE_MODEL`; set that variable (or
`backend/.env`) to parse with another model.  `--model <provider/model>`
selects the cache file that is used and the model id that is reported.

## Caveats

*   The cache records what the model answered, not what it *should* have
    answered.  It is never edited by hand; a wrong parse is fixed by
    re-parsing, not by editing the file.
*   A cached parse is tied to the prompt and the model that produced it.  If
    the parser prompt changes, delete the affected cache files.
*   The presence of a cache file says nothing about the quality of the run:
    a partial cache means the run was interrupted, and the end-to-end results
    are only written after the full query set has been processed.
*   The directory never ships: `backend/scripts/package_release.py` excludes
    `parsed_cache` from the release archive, and
    `backend/scripts/check_release.py` fails the release check when a
    `parsed_cache` path reaches the archive.  The cached parses are
    regenerable, so they are kept out of the bundle on purpose.
