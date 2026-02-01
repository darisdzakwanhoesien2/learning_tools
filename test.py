from pathlib import Path
import os

CODE_DIR = "code_blocks"
os.makedirs(CODE_DIR, exist_ok=True)

# Add your function definition (if not already present)
with open(Path(CODE_DIR) / "f(x1,x2).py", "w") as f:
    f.write("""
def f(x1, x2):
    return 2 * x1**3 - 6 * x1 * x2 + 3 * x2**2
""")
