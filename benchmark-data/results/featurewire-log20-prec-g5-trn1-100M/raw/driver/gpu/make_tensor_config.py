#!/usr/bin/env python3
"""Derives the two files the parity arm needs, from the generated model configuration.

    make_tensor_config.py benchmark-data/generated/20m

hushar's wire contract is named features: a caller sends 210 of them and the service
transforms and concatenates them into the model's one `[rows, 690]` input. Triton's wire
contract is that tensor. Comparing the two as they stand would measure the servers AND the
210 transformations at once, so the parity arm gives hushar a configuration whose single
feature IS the finished vector:

    feature_order            ["features"]
    feature_transformations  identity, default_val 690 wide

Both stacks then receive the same bytes per row and call the graph once, and the only
difference left is the server. The real 210-feature configuration still runs, as its own
arm, and the gap between the two is what the feature API costs.

Two files are written beside the generated model:

    bench_model_tensor_config.json   hushar's parity configuration
    bench_tensor_feature_spec.json   the same input described for Triton's Python client

The width is READ from the real configuration rather than written down here, so the two
configurations cannot come to disagree about how wide a row is.
"""
import json
import os
import sys

TENSOR_INPUT = "features"
MODEL_ID = "bench-vectorized-tensor"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    gen = sys.argv[1].rstrip("/")
    src = os.path.join(gen, "bench_model_config.json")
    if not os.path.isfile(src):
        sys.exit(f"missing {src}; run scripts/generate_benchmark_models.py first")

    real = json.load(open(src))
    vc = real["vectorization_config"]
    transforms = vc["feature_transformations"]
    width = sum(len(transforms[name]["default_val"]) for name in vc["feature_order"])
    if width <= 0:
        sys.exit(f"{src} concatenates to {width} values per row")

    tensor = {
        "model_id": MODEL_ID,
        "model_path": real["model_path"],
        "execution_provider": real.get("execution_provider", "cpu"),
        "vectorization_config": {
            "data_type": vc.get("data_type", "float"),
            "feature_order": [TENSOR_INPUT],
            # identity, so the service copies the vector into the input and does nothing
            # else. The default is what settles the width, exactly as it does for a real
            # feature, and the client reads its length to know how many values to send.
            "feature_transformations": {
                TENSOR_INPUT: {"type": "identity", "default_val": [0.0] * width}
            },
        },
    }
    # A candidate arm is not measured here, but a roll-out cell would want it and deriving
    # it costs nothing, so the two configurations stay in step.
    candidate = json.loads(json.dumps(tensor))
    candidate["model_id"] = f"{MODEL_ID}-candidate"

    # Triton's Python client builds its payload from the signature the SERVER reports and
    # consults this file only for how to invent each value, so `width` here is
    # documentation; the authority is the graph. Written anyway, because the client
    # refuses an input it has no entry for rather than guessing.
    spec = {TENSOR_INPUT: {"kind": "float_array", "width": width}}

    written = {
        "bench_model_tensor_config.json": tensor,
        "bench_model_tensor_candidate_config.json": candidate,
        "bench_tensor_feature_spec.json": spec,
    }
    for name, body in written.items():
        path = os.path.join(gen, name)
        with open(path, "w") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")
        print(f"wrote {path}")

    print(f"  {len(vc['feature_order'])} real features -> {width} values per row")
    print(f"  parity arm sends 1 feature of {width} floats; "
          f"Triton sends {TENSOR_INPUT} FP32[rows,{width}]")
    print(f"  model {real['model_path']}")


if __name__ == "__main__":
    main()
