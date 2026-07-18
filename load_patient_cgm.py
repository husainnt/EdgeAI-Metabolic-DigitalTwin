import os

# path to the wearable blood glucose directory
cgm_dir = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_blood_glucose"

print("=== RAW DIRECTORY LISTING (first 20 items) ===")

if os.path.exists(cgm_dir):
    all_items = os.listdir(cgm_dir)
    print(f"Total items: {len(all_items)}")
    for item in all_items[:20]:
        print("-", item)
else:
    print(f"[ERROR] Could not find path: {cgm_dir}")