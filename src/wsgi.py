import os
import sys

project_home = os.path.dirname(__file__)

if project_home not in sys.path:
    sys.path.append(project_home)

os.environ["MODEL_PATH"] = os.path.join(
    project_home,
    "models",
    "electra_base",
    "checkpoint-4000",
)
os.environ["TOKENIZER_NAME"] = "google/electra-base-discriminator"
os.environ["THRESHOLD"] = "0.5"

from src.api.app import app as application