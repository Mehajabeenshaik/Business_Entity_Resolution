"""
python -m src.main run_all
"""
import subprocess
import sys

STEPS = [
    ["python", "-m", "src.load_data"],
    ["python", "-m", "src.blocking"],          # assumes it accepts train by default
    ["python", "-m", "src.train"],
    ["python", "-m", "src.evaluate"],
    # after re-running blocking on test:
    # ["python", "-m", "src.predict"],
    # ["python", "-m", "src.validate_output"],
]

if __name__ == "__main__":
    for cmd in STEPS:
        print(">>>", " ".join(cmd))
        subprocess.check_call(cmd)
