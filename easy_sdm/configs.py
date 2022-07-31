from pathlib import Path

import yaml

configs = yaml.safe_load(open(Path(__file__).parent / "params.yaml", "r"))
