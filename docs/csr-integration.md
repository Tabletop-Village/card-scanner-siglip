# Pinned inference for CSR

Set `CARD_SCANNER_CSR_MANIFEST=/absolute/path/to/bundle/manifest.json` to
select CSR pinned mode. Run from a clean, committed checkout whose HEAD
matches `scanner_revision`, with `csr_artifacts.py` included in that commit.
The manifest uses CSR's schema version 1 and `csr-siglip-rgb-v1` input contract.
Without this setting the existing local/Hub loading remains available, but
responses omit `X-CSR-Artifact-SHA256` and CSR refuses them.

## Bundle contents

Each of the five artifacts is one regular file with a relative path,
SHA-256 checksum and immutable 40-character source revision:

| Artifact | File contents |
| --- | --- |
| `detector` | The YOLO pose `.pt` checkpoint |
| `base_model` | Tar archive of the complete locally exported Transformers model directory, including config and weights/shards |
| `lora` | Tar archive of the complete locally exported PEFT adapter directory |
| `preprocessing` | Tar archive of the complete locally exported `AutoProcessor` directory, plus `csr-preprocessing.json` below |
| `gallery` | Torch `embeddings.pt` with `ids`, `embeds` and optional `aspect_ratios` |

Archive names are relative to the exported directory, without a wrapper
folder. Include only regular files and directories: no symlinks, hard links,
absolute paths, parent traversal or duplicate entries. A raw Hugging Face
cache directory often contains symlinks; export the model/processor with
`save_pretrained` or copy the complete snapshot into ordinary local files first.
Use deterministic archives and retain them unchanged for restart and rollback.

`csr-preprocessing.json` must contain this supported contract:

```json
{
  "schema_version": 1,
  "input_contract": "csr-siglip-rgb-v1",
  "crop_width": 400,
  "crop_height": 558,
  "resize_interpolation": "opencv_INTER_LINEAR",
  "channel_conversion": "BGR_to_RGB",
  "embedding_normalization": "L2",
  "match_margin_pool_size": 30
}
```

The API decodes color uploads as OpenCV BGR. `/identify` resizes to 400 × 558
with OpenCV linear interpolation; the encoder converts BGR to RGB before the
pinned processor and L2-normalizes the embedding. The processor archive pins
its resize, rescale and normalization settings. `match_margin_pool_size` must
remain 30 in pinned mode. CSR supplies `margin_pct=5`, `min_similarity=0.5`
and `verify=false`; these are retrieval settings, not calibrated confidence.

## Manifest

Use the same manifest on scanner and hub. Paths are relative to its directory;
checksums are lowercase SHA-256 of the actual artifact files (the archive
bytes for multi-file artifacts). Revisions are immutable source/model commits.
The following is a schema example; replace every placeholder before use:

```json
{
  "schema_version": 1,
  "scanner_revision": "<scanner Git commit>",
  "detector": {"path": "detector.pt", "sha256": "<hash>", "revision": "<detector commit>"},
  "preprocessing": {"path": "processor.tar", "sha256": "<hash>", "revision": "<preprocessing commit>"},
  "base_model": {"path": "base.tar", "sha256": "<hash>", "revision": "<base model commit>"},
  "lora": {"path": "lora.tar", "sha256": "<hash>", "revision": "<adapter commit>"},
  "gallery": {"path": "embeddings.pt", "sha256": "<hash>", "revision": "<gallery commit>"},
  "gallery_base_sha256": "<base.tar hash>",
  "gallery_lora_sha256": "<lora.tar hash>",
  "gallery_preprocessing_sha256": "<processor.tar hash>",
  "input_contract": "csr-siglip-rgb-v1"
}
```

The three `gallery_*_sha256` fields must identify the actual embedding producer
and match the pinned artifact checksums. Verify this against the gallery build
record; assigning matching hashes does not itself prove how embeddings were
produced. Rebuild the gallery if its producer cannot be established.

At startup the scanner verifies the source revision and all file checksums
while copying artifacts into a private snapshot, then safely extracts the
model/adapter/processor archives. All loaders use these verified local files;
Transformers and PEFT use `local_files_only=True`. A missing, corrupt,
incompatible or unloadable artifact fails startup without a network fallback.
An empty gallery also fails startup.

Only after all five artifacts load does the scanner expose
`X-CSR-Artifact-SHA256` on successful `/ready` and `/identify` responses,
including an empty candidate list. Its digest uses the same canonical JSON
field order and escaping as CSR's Go verifier; file whitespace and field order
do not change identity. Failed requests do not attest a usable result.
The scanner never reflects a caller's digest or a configured expected digest.

## Commissioning, refresh and rollback

1. Commit the scanner changes and select that clean checkout. Create the
   manifest with its actual Git HEAD and the artifact checksums/revisions.
2. In CSR, run `go run ./tools/scanner-bundle -manifest /path/to/manifest.json`.
   This validates declared compatibility and file integrity without inference.
3. Start the scanner with `CARD_SCANNER_CSR_MANIFEST` and API-key authentication.
   Compare the actual `/ready` response header to CSR's printed digest.
4. Configure CSR with that same manifest, `CSR_SIGLIP_URL` and
   `CSR_SIGLIP_API_KEY`. Test `/v1/scanner/identify` with a real color crop,
   checking raw candidates and stored evidence. Use TLS across hosts.
5. Measure real capture-domain accuracy, failure behavior and end-to-end
   capacity before enabling production or deadline-bound machine admission.

Pinned mode disables scheduled gallery reloads and rejects manual reloads.
A gallery/model change requires a new immutable bundle and process restart.
Retain the old bundle and clean scanner checkout for rollback; old CSR evidence
continues to reference the old digest. Database metadata/price updates can
continue, but never select or reload the inference artifacts.

Tests use lightweight model-loader doubles to exercise actual bundle
verification, startup selection, reload rejection, color preprocessing and API
headers. They do not claim a real model/GPU commissioning result.
