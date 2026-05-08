#!/usr/bin/env python3
"""Edit files, folders, and text lines with backups.
must: 所有代码修改不能使用自带的编辑器, 必须使用w_code.py进行编辑。


Examples:
    python w_code.py add demo.txt 3 "new line"
    python w_code.py delete demo.txt 3
    python w_code.py delete demo.txt 3 --end-line 5
    python w_code.py modify demo.txt 3 "replacement line"
    python w_code.py add-file demo.txt "file content"
    python w_code.py delete-file demo.txt
    python w_code.py add-dir demo_dir
    python w_code.py delete-dir demo_dir
    python w_code.py batch '[{"op":"modify","file":"a.txt","line":1,"content":"A"},{"op":"modify","file":"b.txt","line":1,"content":"B"}]'
    python w_code.py batch ops.json --jobs 4

Backups are created in ./bak by default before every edit.
Backup layout preserves the path relative to the nearest .gitignore root: ./bak/<target-dir>/<target-stem>/<timestamp>.*
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


BACKUP_DIR = Path(__file__).resolve().parent / "bak"


def read_lines(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"file does not exist: {file_path}")
    return file_path.read_text(encoding="utf-8").splitlines(keepends=True)


def normalize_content(content: str) -> str:
    return content if content.endswith("\n") else f"{content}\n"


def validate_line(line: int, minimum: int, maximum: int, label: str = "line") -> None:
    if line < minimum or line > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}, got {line}")


def find_gitignore_root(start_path: Path) -> Path | None:
    resolved = start_path.resolve(strict=False)
    current = resolved if resolved.is_dir() else resolved.parent
    for candidate in (current, *current.parents):
        if (candidate / ".gitignore").is_file():
            return candidate
    return None


def backup_target_relative_path(target_path: Path) -> Path:
    resolved = target_path.resolve(strict=False)
    gitignore_root = find_gitignore_root(resolved)
    if gitignore_root is not None:
        return resolved.relative_to(gitignore_root)

    try:
        return resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        if resolved.is_absolute():
            return Path("__abs__", *resolved.parts[1:])
        return target_path


def backup_leaf_dir_name(target_path: Path, relative_path: Path) -> str:
    if target_path.exists() and target_path.is_dir():
        return relative_path.name
    return relative_path.stem or relative_path.name


def backup_dir_for(target_path: Path) -> Path:
    relative_path = backup_target_relative_path(target_path)
    backup_dir = BACKUP_DIR / relative_path.parent / backup_leaf_dir_name(target_path, relative_path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def backup_path_for(target_path: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return backup_dir_for(target_path) / f"{timestamp}.{suffix}"


def backup_existing_path(target_path: Path) -> Path:
    if not target_path.exists():
        raise FileNotFoundError(f"path does not exist: {target_path}")
    if target_path.is_dir():
        backup_path = backup_path_for(target_path, "bak_dir")
        shutil.copytree(target_path, backup_path)
        return backup_path

    backup_path = backup_path_for(target_path, "bak")
    shutil.copy2(target_path, backup_path)
    return backup_path


def record_create_backup(target_path: Path, kind: str) -> Path:
    backup_path = backup_path_for(target_path, "create_record.txt")
    backup_path.write_text(
        "\n".join(
            [
                f"operation=add-{kind}",
                f"target={target_path.resolve()}",
                "before_state=not_exists",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return backup_path


def add_line(file_path: Path, line: int, content: str) -> None:
    lines = read_lines(file_path)
    validate_line(line, 1, len(lines) + 1)
    backup_existing_path(file_path)
    lines.insert(line - 1, normalize_content(content))
    file_path.write_text("".join(lines), encoding="utf-8")


def delete_lines(file_path: Path, start_line: int, end_line: int | None) -> None:
    lines = read_lines(file_path)
    end = end_line if end_line is not None else start_line
    validate_line(start_line, 1, len(lines), "start line")
    validate_line(end, start_line, len(lines), "end line")
    backup_existing_path(file_path)
    del lines[start_line - 1 : end]
    file_path.write_text("".join(lines), encoding="utf-8")


def modify_line(file_path: Path, line: int, content: str) -> None:
    lines = read_lines(file_path)
    validate_line(line, 1, len(lines))
    backup_existing_path(file_path)
    lines[line - 1] = normalize_content(content)
    file_path.write_text("".join(lines), encoding="utf-8")


def add_file(file_path: Path, content: str | None, force: bool) -> None:
    if file_path.exists():
        if not force:
            raise FileExistsError(f"file already exists: {file_path}")
        backup_existing_path(file_path)
    else:
        record_create_backup(file_path, "file")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content or "", encoding="utf-8")


def delete_file(file_path: Path) -> None:
    if not file_path.is_file():
        raise FileNotFoundError(f"file does not exist: {file_path}")
    backup_existing_path(file_path)
    file_path.unlink()


def add_dir(dir_path: Path, force: bool) -> None:
    if dir_path.exists():
        if not force:
            raise FileExistsError(f"directory already exists: {dir_path}")
        backup_existing_path(dir_path)
    else:
        record_create_backup(dir_path, "dir")

    dir_path.mkdir(parents=True, exist_ok=True)


def delete_dir(dir_path: Path) -> None:
    if not dir_path.is_dir():
        raise FileNotFoundError(f"directory does not exist: {dir_path}")
    backup_existing_path(dir_path)
    shutil.rmtree(dir_path)


def run_operation(op: dict[str, Any]) -> None:
    operation = str(op.get("op", op.get("operation", "")))
    if operation == "add":
        add_line(Path(op["file"]), int(op["line"]), str(op.get("content", "")))
    elif operation == "delete":
        end_line = op.get("end_line", op.get("end-line"))
        delete_lines(Path(op["file"]), int(op["line"]), int(end_line) if end_line is not None else None)
    elif operation == "modify":
        modify_line(Path(op["file"]), int(op["line"]), str(op.get("content", "")))
    elif operation == "add-file":
        add_file(Path(op["file"]), op.get("content"), bool(op.get("force", False)))
    elif operation == "delete-file":
        delete_file(Path(op["file"]))
    elif operation == "add-dir":
        add_dir(Path(op.get("dir", op.get("file"))), bool(op.get("force", False)))
    elif operation == "delete-dir":
        delete_dir(Path(op.get("dir", op.get("file"))))
    else:
        raise ValueError(f"unsupported batch operation: {operation!r}")


def load_batch_operations(source: str) -> list[dict[str, Any]]:
    stripped = source.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        raw = source
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"batch source does not exist: {source_path}")
        raw = source_path.read_text(encoding="utf-8")

    data = json.loads(raw)
    if isinstance(data, dict) and "operations" in data:
        data = data["operations"]
    if not isinstance(data, list):
        raise ValueError("batch input must be a JSON array or an object with an 'operations' array")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("each batch operation must be a JSON object")
    return data


def batch_target_key(op: dict[str, Any]) -> str:
    target = op.get("file", op.get("dir", ""))
    return str(Path(target).resolve()) if target else f"operation:{id(op)}"


def run_batch(operations: list[dict[str, Any]], jobs: int) -> None:
    if not operations:
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for op in operations:
        grouped[batch_target_key(op)].append(op)

    def run_group(group: list[dict[str, Any]]) -> None:
        for item in group:
            run_operation(item)

    workers = max(1, int(jobs))
    if workers == 1 or len(grouped) == 1:
        for group in grouped.values():
            run_group(group)
        return

    with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as executor:
        futures = [executor.submit(run_group, group) for group in grouped.values()]
        for future in as_completed(futures):
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit files, folders, and text lines with backups.")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    add_parser = subparsers.add_parser("add", help="insert one line before the target line")
    add_parser.add_argument("file", type=Path, help="target file path")
    add_parser.add_argument("line", type=int, help="1-based target line number")
    add_parser.add_argument("content", help="content to insert")

    delete_parser = subparsers.add_parser("delete", help="delete one line or a line range")
    delete_parser.add_argument("file", type=Path, help="target file path")
    delete_parser.add_argument("line", type=int, help="1-based start line number")
    delete_parser.add_argument("--end-line", type=int, help="end line for delete operation")

    modify_parser = subparsers.add_parser("modify", help="replace one line")
    modify_parser.add_argument("file", type=Path, help="target file path")
    modify_parser.add_argument("line", type=int, help="1-based target line number")
    modify_parser.add_argument("content", help="replacement content")

    add_file_parser = subparsers.add_parser("add-file", help="create a file")
    add_file_parser.add_argument("file", type=Path, help="file path to create")
    add_file_parser.add_argument("content", nargs="?", help="file content")
    add_file_parser.add_argument("--force", action="store_true", help="overwrite existing file after backup")

    delete_file_parser = subparsers.add_parser("delete-file", help="delete a file after backup")
    delete_file_parser.add_argument("file", type=Path, help="file path to delete")

    add_dir_parser = subparsers.add_parser("add-dir", help="create a directory")
    add_dir_parser.add_argument("dir", type=Path, help="directory path to create")
    add_dir_parser.add_argument("--force", action="store_true", help="keep existing directory after backup")

    delete_dir_parser = subparsers.add_parser("delete-dir", help="delete a directory after backup")
    delete_dir_parser.add_argument("dir", type=Path, help="directory path to delete")

    batch_parser = subparsers.add_parser("batch", help="run multiple edit operations from JSON")
    batch_parser.add_argument("source", help="JSON string, JSON file, or object with an operations array")
    batch_parser.add_argument("--jobs", type=int, default=4, help="parallel workers for different target paths")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.operation == "add":
            add_line(args.file, args.line, args.content)
        elif args.operation == "delete":
            delete_lines(args.file, args.line, args.end_line)
        elif args.operation == "modify":
            modify_line(args.file, args.line, args.content)
        elif args.operation == "add-file":
            add_file(args.file, args.content, args.force)
        elif args.operation == "delete-file":
            delete_file(args.file)
        elif args.operation == "add-dir":
            add_dir(args.dir, args.force)
        elif args.operation == "delete-dir":
            delete_dir(args.dir)
        elif args.operation == "batch":
            run_batch(load_batch_operations(args.source), args.jobs)
    except (FileExistsError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()