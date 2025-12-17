import os
import csv 

raw_dir = os.path.join(os.path.dirname(__file__), 'raw_data')
processed_dir = os.path.join(os.path.dirname(__file__), 'processed_data')

labels = {
    "squat1.mp4": {"exercise": "squat", "form": "good"},
    "squat2.mp4": {"exercise": "squat", "form": "bad"},
    "deadlift1.mp4": {"exercise": "deadlift", "form": "good"},
    "deadlift2.mp4": {"exercise": "deadlift", "form": "bad"},
    "benchpress1.mp4": {"exercise": "bench press", "form": "good"},
    "benchpress2.mp4": {"exercise": "bench press", "form": "bad"},
}

metadata = []
for raw_file in os.listdir(raw_dir):
    raw_path = os.path.join(raw_dir, raw_file)
    processed_file = raw_file.replace(".mp4", ".npy")
    processed_path = os.path.join(processed_dir, processed_file)

    label_info = labels.get(raw_file, {"exercise": "unknown", "form": "unknown"})

    metadata.append({
        "id": raw_file.replace(".mp4", ""),
        "exercise": label_info["exercise"],
        "form": label_info["form"],
        "raw_path": raw_path,
        "processed_path": processed_path,
    })

    csv_file = os.path.join(os.path.dirname(__file__), 'metadata.csv')
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "exercise", "form", "raw_path", "processed_path"])
        writer.writeheader()
        for data in metadata:
            writer.writerow(data)

print(f"Metadata written to {csv_file}")