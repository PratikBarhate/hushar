#!/usr/bin/env python3
"""Reads the TensorRT, CUDA and cuDNN versions out of Triton image CONFIGS.

Only the manifest and the config blob are fetched -- a few kilobytes per tag -- so this
answers "which Triton ships TensorRT 10?" without pulling a 17 GB image per candidate.
"""
import json
import re
import sys
import urllib.request as u
from urllib.parse import urlencode

REPO = "nvidia/tritonserver"
ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])


def get(url, hdrs):
    return u.urlopen(u.Request(url, headers=hdrs), timeout=40).read()


def bearer():
    """The anonymous pull token, from the registry's own auth challenge.

    nvcr.io does not serve the `/v2/token` path Docker Hub does, so the realm is read out
    of the 401's `WWW-Authenticate` header rather than assumed. That makes this work on
    any OCI registry that allows anonymous pulls.
    """
    try:
        u.urlopen(u.Request(f"https://nvcr.io/v2/{REPO}/tags/list"), timeout=40)
        return None
    except u.HTTPError as e:
        if e.code != 401:
            raise
        challenge = e.headers.get("WWW-Authenticate", "")
    parts = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = parts.get("realm")
    if not realm:
        raise RuntimeError(f"no realm in challenge: {challenge!r}")
    query = {"scope": parts.get("scope") or f"repository:{REPO}:pull"}
    if parts.get("service"):
        query["service"] = parts["service"]
    body = json.loads(get(f"{realm}?{urlencode(query)}", {}))
    return body.get("token") or body.get("access_token")


def main():
    token = bearer()
    headers = {"Accept": ACCEPT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    tags = sys.argv[1:] or [f"26.{m:02d}-py3" for m in range(8, 0, -1)]
    for tag in tags:
        try:
            man = json.loads(get(f"https://nvcr.io/v2/{REPO}/manifests/{tag}", headers))
            if "manifests" in man:
                digest = next(x["digest"] for x in man["manifests"]
                              if x.get("platform", {}).get("architecture") == "amd64")
                man = json.loads(
                    get(f"https://nvcr.io/v2/{REPO}/manifests/{digest}", headers))
            config_digest = man["config"]["digest"]
            cfg = json.loads(
                get(f"https://nvcr.io/v2/{REPO}/blobs/{config_digest}", headers))
            env = {}
            for entry in cfg["config"].get("Env", []):
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    env[key] = value
            want = ("TRT_VERSION", "TENSORRT_VERSION", "CUDA_VERSION", "CUDNN_VERSION",
                    "NVIDIA_TRITON_SERVER_VERSION", "NVIDIA_TENSORRT_VERSION")
            found = {k: env[k] for k in want if k in env}
            trt = found.get("TRT_VERSION") or found.get("TENSORRT_VERSION") or "?"
            print(f"{tag:<12} tensorrt={trt:<14} "
                  f"cuda={found.get('CUDA_VERSION', '?'):<12} "
                  f"cudnn={found.get('CUDNN_VERSION', '?'):<10} "
                  f"triton={found.get('NVIDIA_TRITON_SERVER_VERSION', '?')}")
        except Exception as e:
            print(f"{tag:<12} ERR {type(e).__name__} {str(e)[:80]}")


if __name__ == "__main__":
    main()
