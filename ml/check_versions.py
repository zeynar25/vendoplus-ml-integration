"""
check_versions.py - show which Python and library versions this computer is using.

Run it before and after installing requirements.txt:
    python check_versions.py
"""

import platform
import sys

print("Python      :", platform.python_version(),
      "(OK)" if sys.version_info >= (3, 12) else "(too old: requirements.txt needs Python 3.12 or 3.13)")
print("Interpreter :", sys.executable)  # shows whether the virtual environment is active

libraries = [("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn"),
             ("scipy", "scipy"), ("joblib", "joblib"), ("matplotlib", "matplotlib"),
             ("pytest", "pytest")]

for display_name, import_name in libraries:
    try:
        module = __import__(import_name)
        print(f"{display_name:12s}: {module.__version__}")
    except ImportError:
        print(f"{display_name:12s}: not installed")
