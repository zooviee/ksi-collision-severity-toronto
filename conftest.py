import sys
from pathlib import Path

# Add src/ to Python path so all test files can import from src
# regardless of whether they use:
#   from eda_visualizations import ...        (our style)
#   from src.eda_visualizations import ...    (teammate style)
sys.path.insert(0, str(Path(__file__).parent / "src"))