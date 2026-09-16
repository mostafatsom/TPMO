"""Print the upstream FSGO module used by TPMO."""

from __future__ import annotations

import json

from tpmo import upstream_fsgo_info


if __name__ == "__main__":
    print(json.dumps(upstream_fsgo_info(), indent=2))
