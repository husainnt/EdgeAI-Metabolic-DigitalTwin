import os
import pandas as pd

# =====================================================
# BIG IDEAS DATASET INSPECTOR
# =====================================================

folder = r"D:\FYP\DATA_SET\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3\001"

print("=" * 80)
print("FILES FOUND")
print("=" * 80)

files = os.listdir(folder)

for f in files:
    print(f)

print("\n\n")

# =====================================================
# INSPECT EACH CSV
# =====================================================

for filename in files:

    if not filename.lower().endswith(".csv"):
        continue

    filepath = os.path.join(folder, filename)

    print("\n")
    print("=" * 80)
    print("FILE:", filename)
    print("=" * 80)

    # -------------------------------------------------
    # Print first 20 raw lines
    # -------------------------------------------------

    print("\nFIRST 20 RAW LINES:\n")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i in range(20):
                line = f.readline()

                if not line:
                    break

                print(line.rstrip())

    except Exception as e:
        print("Could not open file:", e)

    print("\n")

    # -------------------------------------------------
    # Try reading normally
    # -------------------------------------------------

    print("Trying pandas.read_csv()...\n")

    try:

        df = pd.read_csv(filepath)

        print("SUCCESS")

        print("\nColumns:\n")
        print(df.columns.tolist())

        print("\nFirst 5 rows:\n")
        print(df.head())

    except Exception as e:

        print("FAILED")
        print(e)

        # -------------------------------------------------
        # Try skipping metadata rows
        # -------------------------------------------------

        print("\nTrying skiprows...\n")

        success = False

        for skip in range(1, 15):

            try:

                df = pd.read_csv(filepath, skiprows=skip)

                print(f"SUCCESS with skiprows={skip}")

                print("\nColumns:\n")
                print(df.columns.tolist())

                print("\nFirst 5 rows:\n")
                print(df.head())

                success = True
                break

            except:
                pass

        if not success:
            print("Could not automatically determine header.")

print("\n")
print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
