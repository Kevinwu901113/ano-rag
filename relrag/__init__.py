import os

# Automatically bypass proxy for localhost to prevent 502 errors with local services (e.g., vLLM)
# This is critical when http_proxy is set in the environment.
if "no_proxy" not in os.environ:
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
else:
    # Append to existing no_proxy if not present
    existing = os.environ["no_proxy"]
    additions = []
    for host in ["localhost", "127.0.0.1", "0.0.0.0"]:
        if host not in existing:
            additions.append(host)
    if additions:
        os.environ["no_proxy"] = existing + "," + ",".join(additions)
