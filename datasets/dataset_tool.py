#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
================================================================================
 dataset_tool.py — инструмент для просмотра, анализа и редактирования датасетов
                   Heretic (формат Apache Arrow / Hugging Face `datasets`).
================================================================================

Утилита позволяет "подергать" датасеты: посмотреть содержимое, посчитать
статистику, найти промпты, а также перевести данные в удобно читаемый и
редактируемый формат (TXT / CSV / JSONL) и импортировать их обратно в .arrow.

--------------------------------------------------------------------------------
 ЗАПУСК (важно!)
--------------------------------------------------------------------------------
Нужна библиотека `datasets`. Запускай через окружение проекта `venv3.12`
ИЗ КОРНЯ ПРОЕКТА (.../heretic/):

    ./venv3.12/bin/python datasets/dataset_tool.py <команда> [аргументы]

Если окружения нет — создай его:

    python3.12 -m venv venv3.12
    ./venv3.12/bin/python -m pip install --upgrade pip
    ./venv3.12/bin/python -m pip install "datasets>=3.0"

--------------------------------------------------------------------------------
 ИМЯ ДАТАСЕТА
--------------------------------------------------------------------------------
Везде, где нужен <name>, указывай ПУТЬ К SPLIT-ПАПКЕ (в ней лежит state.json).
Имя можно писать коротко (относительно папки datasets/) или полным путём:

    harmful_behaviors/train           (коротко — относительно datasets/)
    datasets/harmful_behaviors/train  (от корня проекта)

--------------------------------------------------------------------------------
 КОМАНДЫ (полная инструкция — с примерами)
--------------------------------------------------------------------------------
1) list
   Показать все локальные датасеты (split-папки) внутри datasets/.
       ./venv3.12/bin/python datasets/dataset_tool.py list

2) info <name> [-n N] [-c COLUMN]
   Структура датасета: число строк, колонки, схема, первые N примеров.
       ... info harmful_behaviors/train
       ... info harmless_alpaca/train -n 10

3) head <name> [-n N] [-c COLUMN]
   Показать первые N значений колонки (по умолчанию N=10, колонка "text").
       ... head harmful_behaviors/train -n 20

4) sample <name> [-n N] [-c COLUMN] [--seed S]
   Показать N случайных промптов.
       ... sample harmless_alpaca/train -n 5

5) stats <name> [-c COLUMN]
   Аналитика колонки: всего/уникальных/дубликатов/пустых, длины (мин/сред/макс).
       ... stats harmful_behaviors/train

6) search <name> <query> [-c COLUMN] [-i] [--limit N]
   Найти строки, содержащие подстроку (-i = без учёта регистра).
       ... search harmful_behaviors/train hack -i

7) export <name> --to {txt,csv,jsonl} [-o OUT] [-c COLUMN]
   Перевести датасет в удобный формат для чтения/правки глазами.
   По умолчанию файл кладётся рядом: datasets/<name>.<формат>
       ... export harmful_behaviors/train --to txt
       ... export harmless_alpaca/test --to csv -o /tmp/test.csv

8) import <source_file> --out <disk_path> [-c COLUMN]
   Импортировать TXT/CSV/JSONL обратно в формат .arrow (save_to_disk).
   Для .txt: 1 строка = 1 промпт (пустые строки игнорируются).
       ... import my_prompts.txt --out datasets/my_prompts/train

9) add <name> (--text "..." [...] | --from-file FILE) [-c COLUMN]
   Добавить новые промпты в существующий датасет (и сохранить на месте).
       ... add harmful_behaviors/train --text "Новый промпт 1" "Новый промпт 2"
       ... add harmful_behaviors/train --from-file extra_prompts.txt

--------------------------------------------------------------------------------
 ТИПИЧНЫЙ СЦЕНАРИЙ РЕДАКТИРОВАНИЯ
--------------------------------------------------------------------------------
  1. export harmful_behaviors/train --to txt      # выгрузить в текст
  2. открыть datasets/harmful_behaviors/train.txt в редакторе, править строки
  3. import datasets/harmful_behaviors/train.txt --out datasets/harmful_behaviors/train
     (перезапишет .arrow правленными данными)

Подробнее о форматах и подключении локальных данных к Heretic — см.
datasets/HOW_TO_USE_DATASETS.MD
================================================================================


================================================================================
 ENGLISH — quick reference
================================================================================
Tool to view, analyze and edit Heretic datasets (Apache Arrow / HF `datasets`).
Run from the PROJECT ROOT with the project environment:

    ./venv3.12/bin/python datasets/dataset_tool.py <command> [args]

<name> is the path to a SPLIT folder (the one containing state.json), e.g.
`harmful_behaviors/train` (short, relative to datasets/) or the full path.

Commands: list | info | head | sample | stats | search | export | import | add
Run any command with -h for its options, e.g.:

    ./venv3.12/bin/python datasets/dataset_tool.py export -h

Typical edit flow: export ... --to txt  ->  edit the .txt  ->  import ... --out <split>
See datasets/HOW_TO_USE_DATASETS.MD for the full guide.
================================================================================
"""

import argparse
import os
import random
import shutil
import sys

try:
    from datasets import (
        Dataset,
        DatasetDict,
        concatenate_datasets,
        load_dataset,
        load_from_disk,
    )
except ImportError:
    sys.exit(
        "ОШИБКА: библиотека 'datasets' не установлена.\n"
        "Создай окружение и установи её:\n"
        "  python3.12 -m venv venv3.12\n"
        "  ./venv3.12/bin/python -m pip install 'datasets>=3.0'\n"
        "и запускай так:\n"
        "  ./venv3.12/bin/python datasets/dataset_tool.py <команда>"
    )

# Папка datasets/ (в ней лежит этот скрипт) — база для коротких имён.
DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = "state.json"  # маркер локального датасета save_to_disk


# ------------------------------- ХЕЛПЕРЫ ------------------------------------ #


def resolve_path(name: str) -> str:
    """Преобразует короткое имя или путь в существующий путь к папке датасета."""
    for candidate in (name, os.path.join(DATASETS_DIR, name)):
        if os.path.isdir(candidate):
            return candidate
    sys.exit(
        f"ОШИБКА: датасет/путь не найден: '{name}'\n"
        "Подсказка: выполни команду 'list', чтобы увидеть доступные датасеты."
    )


def load_single(path: str) -> Dataset:
    """Загружает split-датасет. Если это набор split-ов — подсказывает выбрать один."""
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        hint = os.path.join(path, splits[0]) if splits else "<split>"
        sys.exit(
            f"ОШИБКА: '{path}' — это набор из нескольких split-ов: {splits}\n"
            f"Укажи конкретный split, например: {hint}"
        )
    return ds


def pick_column(ds: Dataset, requested: str | None) -> str:
    """Выбирает колонку: заданную, либо 'text', либо первую доступную."""
    if requested:
        if requested not in ds.column_names:
            sys.exit(
                f"ОШИБКА: колонки '{requested}' нет. "
                f"Доступные колонки: {ds.column_names}"
            )
        return requested
    if "text" in ds.column_names:
        return "text"
    return ds.column_names[0]


def save_dataset(ds: Dataset, dest: str) -> None:
    """Сохраняет датасет в .arrow (save_to_disk), безопасно перезаписывая dest.

    save_to_disk нельзя писать в ту же папку, откуда читается датасет
    (конфликт memory-map), поэтому пишем во временную папку и заменяем.
    """
    dest = os.path.abspath(dest)
    tmp = dest + "__tmp_save"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    ds.save_to_disk(tmp)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(tmp, dest)


def first_values(ds: Dataset, column: str, n: int) -> list:
    """Быстро берёт первые n значений колонки."""
    n = min(n, ds.num_rows)
    return ds[:n][column]


# ------------------------------- КОМАНДЫ ------------------------------------ #


def cmd_list(args: argparse.Namespace) -> None:
    found = []
    for root, dirs, files in os.walk(DATASETS_DIR):
        if STATE_FILE in files:
            found.append(root)
            dirs[:] = []  # не спускаемся внутрь папки датасета
    if not found:
        print(f"Локальных датасетов не найдено в {DATASETS_DIR}")
        return
    print("Доступные локальные датасеты (split-папки):\n")
    for path in sorted(found):
        rel = os.path.relpath(path, DATASETS_DIR)
        try:
            ds = load_from_disk(path)
            info = f"{ds.num_rows:>7} строк | колонки: {ds.column_names}"
        except Exception as error:
            info = f"(ошибка чтения: {error})"
        print(f"  {rel:<34} {info}")
    print("\nИспользуй короткое имя (левая колонка) в других командах.")


def cmd_info(args: argparse.Namespace) -> None:
    path = resolve_path(args.name)
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        print(f"'{path}' — набор split-ов:")
        for name, split in ds.items():
            print(
                f"  {name:<8} {split.num_rows:>7} строк | колонки: {split.column_names}"
            )
        print("\nДля просмотра конкретного split укажи, например:",
              os.path.join(args.name, next(iter(ds.keys()))))
        return
    print(f"Путь    : {path}")
    print(f"Строк   : {ds.num_rows}")
    print(f"Колонки : {ds.column_names}")
    print(f"Схема   : {ds.features}")
    column = pick_column(ds, args.column)
    print(f"\nПервые примеры (колонка '{column}'):")
    for i, value in enumerate(first_values(ds, column, args.n)):
        print(f"  [{i}] {value!r}")


def cmd_head(args: argparse.Namespace) -> None:
    ds = load_single(resolve_path(args.name))
    column = pick_column(ds, args.column)
    for i, value in enumerate(first_values(ds, column, args.n)):
        print(f"[{i}] {value}")


def cmd_sample(args: argparse.Namespace) -> None:
    ds = load_single(resolve_path(args.name))
    column = pick_column(ds, args.column)
    if args.seed is not None:
        random.seed(args.seed)
    count = min(args.n, ds.num_rows)
    indices = sorted(random.sample(range(ds.num_rows), count))
    values = ds.select(indices)[column]
    for idx, value in zip(indices, values):
        print(f"[{idx}] {value}")


def cmd_stats(args: argparse.Namespace) -> None:
    ds = load_single(resolve_path(args.name))
    column = pick_column(ds, args.column)
    values = [str(v) for v in ds[column]]
    total = len(values)
    lengths = [len(v) for v in values]
    unique = len(set(values))
    empty = sum(1 for v in values if not v.strip())
    print(f"Датасет      : {resolve_path(args.name)}")
    print(f"Колонка      : {column}")
    print(f"Всего строк  : {total}")
    print(f"Уникальных   : {unique}")
    print(f"Дубликатов   : {total - unique}")
    print(f"Пустых       : {empty}")
    if lengths:
        print(f"Длина (симв.): мин {min(lengths)} | "
              f"сред {sum(lengths) / len(lengths):.1f} | макс {max(lengths)}")


def cmd_search(args: argparse.Namespace) -> None:
    ds = load_single(resolve_path(args.name))
    column = pick_column(ds, args.column)
    query = args.query.lower() if args.ignore_case else args.query
    matches = []
    for i, value in enumerate(ds[column]):
        haystack = str(value).lower() if args.ignore_case else str(value)
        if query in haystack:
            matches.append((i, value))
    print(f"Найдено совпадений: {len(matches)} (показаны первые {args.limit})\n")
    for i, value in matches[: args.limit]:
        print(f"[{i}] {value}")


def cmd_export(args: argparse.Namespace) -> None:
    path = resolve_path(args.name)
    ds = load_single(path)
    out = args.out or f"{path}.{args.to}"
    if args.to == "txt":
        column = pick_column(ds, args.column)
        with open(out, "w", encoding="utf-8") as file:
            for value in ds[column]:
                file.write(str(value) + "\n")
    elif args.to == "csv":
        ds.to_csv(out)
    elif args.to == "jsonl":
        ds.to_json(out)
    print(f"Экспортировано {ds.num_rows} строк -> {out}")


def cmd_import(args: argparse.Namespace) -> None:
    source = args.source
    if not os.path.isfile(source):
        sys.exit(f"ОШИБКА: файл не найден: {source}")
    column = args.column or "text"
    ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""

    if ext == "txt" or ext == "":
        with open(source, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
        ds = Dataset.from_dict({column: lines})
    elif ext == "csv":
        ds = load_dataset("csv", data_files=source)["train"]
    elif ext in ("jsonl", "json"):
        ds = load_dataset("json", data_files=source)["train"]
    else:
        sys.exit(f"ОШИБКА: неподдерживаемый формат '.{ext}'. Используй txt/csv/jsonl.")

    save_dataset(ds, args.out)
    print(f"Импортировано {ds.num_rows} строк -> {args.out} (формат .arrow)")


def cmd_add(args: argparse.Namespace) -> None:
    path = resolve_path(args.name)
    ds = load_single(path)
    column = pick_column(ds, args.column)

    new_values: list[str] = []
    if args.text:
        new_values.extend(args.text)
    if args.from_file:
        if not os.path.isfile(args.from_file):
            sys.exit(f"ОШИБКА: файл не найден: {args.from_file}")
        with open(args.from_file, encoding="utf-8") as file:
            new_values.extend(line.strip() for line in file if line.strip())

    if not new_values:
        sys.exit("ОШИБКА: не переданы новые промпты (используй --text или --from-file).")

    # Новый датасет должен иметь те же колонки, что и исходный.
    extra_data = {col: [None] * len(new_values) for col in ds.column_names}
    extra_data[column] = new_values
    extra = Dataset.from_dict(extra_data, features=ds.features)

    merged = concatenate_datasets([ds, extra])
    save_dataset(merged, path)
    print(f"Добавлено {len(new_values)} строк: было {ds.num_rows} -> стало {merged.num_rows}")


# --------------------------------- CLI -------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_tool.py",
        description="Просмотр, анализ и редактирование датасетов Heretic (.arrow). "
        "Полная инструкция — в docstring этого файла и в datasets/HOW_TO_USE_DATASETS.MD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<команда>")

    p = sub.add_parser("list", help="показать все локальные датасеты")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="структура датасета и первые примеры")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("-n", type=int, default=5, help="сколько примеров показать (по умолч. 5)")
    p.add_argument("-c", "--column", help="колонка (по умолч. 'text')")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("head", help="первые N значений колонки")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("-n", type=int, default=10, help="сколько строк (по умолч. 10)")
    p.add_argument("-c", "--column", help="колонка (по умолч. 'text')")
    p.set_defaults(func=cmd_head)

    p = sub.add_parser("sample", help="N случайных промптов")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("-n", type=int, default=5, help="сколько строк (по умолч. 5)")
    p.add_argument("-c", "--column", help="колонка (по умолч. 'text')")
    p.add_argument("--seed", type=int, help="сид для воспроизводимой выборки")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("stats", help="статистика по колонке")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("-c", "--column", help="колонка (по умолч. 'text')")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="поиск строк по подстроке")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("query", help="искомая подстрока")
    p.add_argument("-c", "--column", help="колонка (по умолч. 'text')")
    p.add_argument("-i", "--ignore-case", action="store_true", help="без учёта регистра")
    p.add_argument("--limit", type=int, default=20, help="сколько совпадений показать")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("export", help="экспорт в txt/csv/jsonl")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("--to", required=True, choices=["txt", "csv", "jsonl"], help="формат")
    p.add_argument("-o", "--out", help="куда сохранить (по умолч. рядом с датасетом)")
    p.add_argument("-c", "--column", help="колонка для txt (по умолч. 'text')")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="импорт txt/csv/jsonl в .arrow")
    p.add_argument("source", help="исходный файл (.txt/.csv/.jsonl)")
    p.add_argument("--out", required=True, help="куда сохранить .arrow (split-папка)")
    p.add_argument("-c", "--column", help="имя колонки для txt (по умолч. 'text')")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("add", help="добавить промпты в датасет")
    p.add_argument("name", help="имя/путь split-датасета")
    p.add_argument("--text", nargs="+", help="один или несколько промптов")
    p.add_argument("--from-file", help="текстовый файл с промптами (1 на строку)")
    p.add_argument("-c", "--column", help="колонка (по умолч. 'text')")
    p.set_defaults(func=cmd_add)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
