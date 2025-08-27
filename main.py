#!/usr/bin/env python3
"""
Redirects to main-gui.py for backward compatibility
"""
import sys
import subprocess
from pathlib import Path

def main():
    # Execute main-gui.py instead
    main_gui = Path(__file__).parent / "main-gui.py"
    subprocess.run([sys.executable, str(main_gui)] + sys.argv[1:])

if __name__ == '__main__':
    main()
