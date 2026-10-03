import os
import sys

# Synced/network filesystems can keep coarse mtimes, which lets stale .pyc
# files shadow edited sources.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The tests never reach a real model; make any accidental network call fail loudly.
os.environ.setdefault("GOOGLE_API_KEY", "offline-test-key-not-valid")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
