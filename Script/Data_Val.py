
from pathlib import Path
from datetime import datetime, timezone
import os
import uuid

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "PINPAD_DATA_ROOT",
    "/home/cse/pinpad/myenv/GNURadio_1/Correlation_Responses/OneSubcarrierLowDataRate",
)).expanduser().resolve()
OUTPUT = Path(os.environ.get(
    "PINPAD_OUTPUT_ROOT",
    "/home/cse/pinpad/myenv/GNURadio_1/PINPAD_Results",
)).expanduser().resolve()

N_BUTTONS, N_PER_BUTTON = 12, 200
RESPONSE_LENGTH = 3624
SEED = 42

SOURCE = [(c, "R1") for c in range(1, 6)]
ADAPT_CC = [(c, "R1") for c in range(6, 11)]
ADAPT_CR = [(c, "R2") for c in range(1, 6)]
ADAPT_CCR = [(c, "R2") for c in range(11, 16)]
TEST_CC = [(c, "R1") for c in range(16, 26)]
TEST_CR = [(c, "R3") for c in range(1, 6)]
TEST_CCR = [(c, "R3") for c in range(26, 36)]

ADAPT = {"Source": SOURCE, "CC": ADAPT_CC, "CR": ADAPT_CR, "CCR": ADAPT_CCR}
TEST = {"CC": TEST_CC, "CR": TEST_CR, "CCR": TEST_CCR}
GROUPS = {
    **{f"adapt_{name}": keys for name, keys in ADAPT.items()},
    **{f"test_{name}": keys for name, keys in TEST.items()},
}
EXPECTED_COUNTS = {
    "adapt_Source": 12000,
    "adapt_CC": 12000,
    "adapt_CR": 12000,
    "adapt_CCR": 12000,
    "test_CC": 24000,
    "test_CR": 12000,
    "test_CCR": 24000,
}
COLLECTION_MAP = ROOT / "collection_paths.csv"

adapt_keys = [key for keys in ADAPT.values() for key in keys]
test_keys = [key for keys in TEST.values() for key in keys]
EXPECTED = set(adapt_keys + test_keys)
if len(adapt_keys) != 20 or len(test_keys) != 25:
    raise ValueError("Expected 20 adaptation and 25 evaluation collections")
if len(EXPECTED) != 45 or set(adapt_keys) & set(test_keys):
    raise ValueError("Repeated or overlapping declared card-reader collections")
if len({card for card, _ in EXPECTED}) != 35:
    raise ValueError("Expected 35 distinct declared card identities")


def _paths():
    if COLLECTION_MAP.is_file():
        frame = pd.read_csv(COLLECTION_MAP, keep_default_na=False)
        if not {"card", "reader", "directory"} <= set(frame.columns):
            raise ValueError("collection_paths.csv requires card,reader,directory")
        mapping = {}
        for row in frame.itertuples(index=False):
            card_value = float(row.card)
            if not np.isfinite(card_value) or not card_value.is_integer():
                raise ValueError(f"Invalid card identifier: {row.card}")
            key = (int(card_value), str(row.reader).strip().upper())
            if key[1] not in ("R1", "R2", "R3"):
                raise ValueError(f"Invalid reader identifier: {key}")
            if key in mapping:
                raise ValueError(f"Duplicate collection: {key}")
            directory = str(row.directory).strip()
            if not directory:
                raise ValueError(f"Empty directory for collection {key}")
            path = Path(directory).expanduser()
            mapping[key] = (path if path.is_absolute() else ROOT / path).resolve()
        return mapping

    mapping = {}
    for reader in ("R1", "R2", "R3"):
        for card in range(1, 36):
            candidates = [
                ROOT / reader / f"card{card}",
                ROOT / reader.lower() / f"card{card}",
                ROOT / f"card{card}" / reader,
                ROOT / f"card{card}" / reader.lower(),
            ]
            matches = sorted({p.resolve() for p in candidates if p.is_dir()})
            if len(matches) == 1:
                mapping[(card, reader)] = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous paths for card {card}, {reader}: {matches}")
    return mapping


def _read_collection(key, collections):
    path = collections[key]
    parts = []
    for button in range(N_BUTTONS):
        candidates = [path / f"button{button}.npy", path / f"Button{button}.npy"]
        hits = sorted({p.resolve() for p in candidates if p.is_file()})
        if len(hits) != 1:
            raise FileNotFoundError(f"Expected one button{button}.npy in {path}; found {hits}")
        x = np.load(hits[0], allow_pickle=False)
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError(f"Non-numeric response array: {hits[0]}")
        if x.shape != (N_PER_BUTTON, RESPONSE_LENGTH):
            raise ValueError(
                f"Expected {(N_PER_BUTTON, RESPONSE_LENGTH)} at {hits[0]}; got {x.shape}"
            )
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite waveform data: {hits[0]}")
        parts.append(x)
    return np.concatenate(parts), np.repeat(np.arange(N_BUTTONS), N_PER_BUTTON)


def load_group(keys, collections=None):
    """Return waveforms, button labels, card IDs, and reader IDs."""
    if collections is None:
        collections = _paths()
    data = [_read_collection(key, collections) for key in keys]
    if not data:
        raise ValueError("Cannot load an empty group")
    x = np.concatenate([item[0] for item in data])
    y = np.concatenate([item[1] for item in data])
    cards = np.concatenate([
        np.full(len(item[1]), key[0], dtype=np.int64)
        for key, item in zip(keys, data)
    ])
    readers = np.concatenate([
        np.full(len(item[1]), key[1], dtype="U2")
        for key, item in zip(keys, data)
    ])
    return x, y, cards, readers


def validate_dataset():
    collections = _paths()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    run_output = OUTPUT / "dataset_validation" / run_id
    run_output.mkdir(parents=True, exist_ok=False)

    missing = sorted(key for key in EXPECTED if key not in collections or not collections[key].is_dir())
    if missing:
        template = run_output / "collection_paths_REQUIRED.csv"
        pd.DataFrame([
            {"card": card, "reader": reader, "directory": str(collections.get((card, reader), ""))}
            for card, reader in sorted(EXPECTED)
        ]).to_csv(template, index=False)
        raise FileNotFoundError(
            f"Missing {len(missing)} collections: {missing}. "
            f"Set PINPAD_DATA_ROOT or fill {COLLECTION_MAP} using {template}. "
            "Reader identity cannot be inferred from a card number alone."
        )

    directory_owners = {}
    for key in sorted(EXPECTED):
        path = collections[key].resolve()
        if path in directory_owners:
            raise ValueError(f"Same directory assigned to {directory_owners[path]} and {key}: {path}")
        directory_owners[path] = key

    rows, collection_rows = [], []
    for name, keys in GROUPS.items():
        samples, buttons = 0, set()
        # Validate one collection at a time to bound memory use.
        for key in keys:
            x, y = _read_collection(key, collections)
            samples += len(x)
            buttons.update(map(int, np.unique(y)))
            collection_rows.append({
                "domain": name, "card": key[0], "reader": key[1],
                "directory": str(collections[key]), "samples": len(x),
            })
            del x, y
        if samples != EXPECTED_COUNTS[name]:
            raise ValueError(f"{name}: got {samples}, expected {EXPECTED_COUNTS[name]}")
        rows.append({
            "domain": name, "collections": len(keys), "samples": samples,
            "buttons": len(buttons), "cards": sorted({c for c, _ in keys}),
            "readers": sorted({r for _, r in keys}),
        })
        print(f"Validated {name}: {len(keys)} collections, {samples:,} responses", flush=True)

    report = pd.DataFrame(rows)
    if int(report.samples.sum()) != 108000:
        raise ValueError("Expected 108,000 total responses")
    report.to_csv(run_output / "verified_dataset_collections.csv", index=False)
    pd.DataFrame(collection_rows).to_csv(run_output / "resolved_collection_paths.csv", index=False)
    print("\n" + report.to_string(index=False))
    print("\nDeclared inventory: 35 distinct cards; 3 reader configurations; 45 collections")
    print("Validated responses:", f"{int(report.samples.sum()):,}")
    print("Saved reports:", run_output)
    print("Reader identities follow the supplied map; physical identity and response-content independence are not established by this check.")
    return report


if __name__ == "__main__":
    report = validate_dataset()
