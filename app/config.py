"""All settings come from the environment (.env). Nothing is hard-coded in the app."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

# Model and limits
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()  # "gemini" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1500"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
# Cosine distance (0 = identical, 2 = opposite). Chunks further away than this
# are treated as "not relevant" and never sent to the model.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.8"))

# Guardrails
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
REFUSAL_MESSAGE = os.getenv("REFUSAL_MESSAGE", "That isn't in your documents.")

# Cost logging (USD per million tokens). Set these to your model's current prices.
PRICE_INPUT_PER_MTOK = float(os.getenv("PRICE_INPUT_PER_MTOK", "0"))
PRICE_OUTPUT_PER_MTOK = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "0"))

# Admin page
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
