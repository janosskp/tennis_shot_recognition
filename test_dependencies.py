#!/usr/bin/env python
"""Test script to verify all dependencies are installed"""

import sys

packages = [
    ("pandas", "Data manipulation"),
    ("numpy", "Numerical computing"),
    ("cv2", "Computer vision"),
    ("tensorflow", "Deep learning framework"),
    ("keras", "Neural network API"),
    ("tqdm", "Progress bars"),
]

print(f"Python {sys.version}")
print("-" * 60)

all_ok = True
for package_name, description in packages:
    try:
        __import__(package_name)
        print(f"✓ {package_name:20} - {description}")
    except ImportError as e:
        print(f"✗ {package_name:20} - {description} [FAILED]")
        all_ok = False

print("-" * 60)
if all_ok:
    print("✓ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    sys.exit(0)
else:
    print("✗ Some dependencies are missing")
    sys.exit(1)
