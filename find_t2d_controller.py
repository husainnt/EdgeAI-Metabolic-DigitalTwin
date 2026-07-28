"""
Locate T2DPancreaticController and Task 4/5 simglucose Patient Setup
"""

import os

search_dir = r"D:\FYP\CODE"
target_terms = ["T2DPancreaticController", "Vmx", "kp3", "u2ss", "T1DPatient"]

print(f"Scanning {search_dir} for T2DPancreaticController and simglucose setups...\n" + "=" * 75)

matches = {}

for root, dirs, files in os.walk(search_dir):
    # Skip virtual environments or cache folders
    if any(p in root for p in ['.venv', '__pycache__', 'site-packages']):
        continue
    for file in files:
        if file.endswith(".py"):
            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    found = [term for term in target_terms if term in content]
                    if found:
                        matches[full_path] = found
            except Exception as e:
                pass

if matches:
    print(f"✓ Found {len(matches)} matching file(s):\n")
    for path, terms in matches.items():
        print(f"📄 Path: {path}")
        print(f"   Matched Keywords: {', '.join(terms)}")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                print("   Key Snippets:")
                for line_no, line in enumerate(lines, 1):
                    if any(t in line for t in ["class T2DPancreaticController", "def ", "Vmx", "kp3", "u2ss"]):
                        print(f"     Line {line_no:03d}: {line.strip()}")
        except Exception:
            pass
        print("-" * 75)
else:
    print("❌ No direct matches found. Please check if Task 4/5 code is in another folder.")