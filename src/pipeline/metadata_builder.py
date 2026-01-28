from pathlib import Path
"""
Scans a directory for processed video .mp4 files. Returns metadata list for each video

Args:
    folder_name (str | Path): Root directory to scan

Returns:
    list[dict[str, str]]: A list of metadata disctionaries, each with
    video ID (parent folder name), file path (from original directory), and dataset split
"""

def scan_processed(folder_name: str | Path) -> list[dict[str, str]]:
    path = Path(folder_name)

    if not path.exists() or not path.is_dir():
        return []

    contents = path.iterdir()
    vid_list: list[dict[str, str]] = []


    for child in contents:
        if child.is_dir():
            vid_list.extend(scan_processed(child))
        elif child.is_file() and child.suffix.lower() == ".mp4":
            vid_list.append(
                {
                    "id": str(child.parent.name),
                    "processed_path": str(child.as_posix()),
                    "split": "",
                }
            )


    return vid_list

