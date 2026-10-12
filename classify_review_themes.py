"""Root entry point for Thematic Issue Classification Pipeline.

Delegates execution to model.classify_review_themes.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.classify_review_themes import main

if __name__ == "__main__":
    main()
