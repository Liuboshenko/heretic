# HOW TO USE DATASETS — Heretic

> Исчерпывающее руководство по работе с датасетами промптов в проекте Heretic:
> формат хранения, чтение/анализ/редактирование, **как сделать чтение локальным
> (без выгрузки из сети)**, как расширять и добавлять промпты.
>
> Документ состоит из двух частей: сначала **на русском языке**, затем **на английском**.
>
> ⚠️ Все команды выполняются **из корня проекта** (`.../heretic/`), а Python-код —
> из окружения `venv3.12` (`./venv3.12/bin/python`), в котором установлена библиотека `datasets`.

---

# 🇷🇺 Часть 1. Русская версия

## 1. Что это за датасеты и где они лежат

Проекту нужны **две группы промптов** (метод строит «направление отказа» как разницу между ними):

| Директория | Источник (Hugging Face) | Роль |
| :--- | :--- | :--- |
| `datasets/harmless_alpaca/` | `mlabonne/harmless_alpaca` | 🟢 «безобидные» промпты (без отказов) |
| `datasets/harmful_behaviors/` | `mlabonne/harmful_behaviors` | 🔴 «вредные» промпты (с отказами) |

Внутри каждой директории — подпапки по split-ам, скачанные и сохранённые локально:

```
datasets/
├── HOW_TO_USE_DATASETS.MD        ← этот файл
├── harmless_alpaca/
│   ├── train/                    ← 25058 строк, колонка "text"
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── test/                     ← 6265 строк
└── harmful_behaviors/
    ├── train/                    ← 416 строк, колонка "text"
    └── test/                     ← 104 строки
```

- Колонка с промптами во всех наборах: **`text`**.
- `train` используется для расчёта направления отказа, `test` — для оценки качества.

## 2. Формат `.arrow` — что это

`data-00000-of-00001.arrow` — это **Apache Arrow**: бинарный **колоночный** формат таблиц,
на котором работает библиотека Hugging Face `datasets`.

- **Колоночный** — данные хранятся по столбцам → быстрые операции над колонками.
- **Memory-mapped** — датасет читается с диска по мере надобности, не грузится целиком в RAM.
- **Типизированный** — схема (типы колонок) хранится вместе с данными.
- **Бинарный** — открывать в текстовом редакторе бесполезно, нужен код или спец-инструмент.

Три файла в каждой split-папке работают вместе:

| Файл | Назначение |
| :--- | :--- |
| `data-00000-of-00001.arrow` | сами данные (таблица) |
| `dataset_info.json` | метаданные: описание, признаки (features), число строк |
| `state.json` | какие `.arrow`-файлы образуют датасет — **по этому файлу Heretic распознаёт локальный датасет** |

## 3. Окружение для работы с датасетами (`venv3.12`)

Для чтения/редактирования `.arrow` нужна библиотека `datasets`. Она установлена в отдельное
окружение `venv3.12` в корне проекта. Запуск любого кода ниже:

```bash
./venv3.12/bin/python твой_скрипт.py
# или интерактивно:
./venv3.12/bin/python
```

**Если окружения нет** (например, после клонирования на другой машине) — создать заново:

```bash
python3.12 -m venv venv3.12
./venv3.12/bin/python -m pip install --upgrade pip
./venv3.12/bin/python -m pip install "datasets>=3.0"
```

## 4. Как читать датасет

```python
from datasets import load_from_disk

# Загружаем split-папку (именно подпапку train/ или test/, не корень датасета)
ds = load_from_disk("datasets/harmful_behaviors/train")

ds.num_rows            # 416 — число строк
ds.column_names        # ['text'] — список колонок
ds.features            # {'text': Value('string')} — схема/типы

ds[0]                  # {'text': '...'} — первая строка целиком (dict)
ds['text'][0]          # значение колонки в первой строке
ds[:3]['text']         # список первых 3 промптов
```

## 5. Как анализировать (через pandas)

```python
df = ds.to_pandas()                          # вся таблица -> DataFrame
df.shape                                     # (416, 1)
df['text'].str.len().mean()                  # средняя длина промпта
df[df['text'].str.contains("hack")]          # поиск/фильтр по подстроке
df['text'].duplicated().sum()                # сколько дубликатов
```

## 6. Как редактировать и сохранять обратно в `.arrow`

Объект `Dataset` **иммутабелен** — каждая операция возвращает новый объект, который нужно
присвоить обратно, а затем сохранить через `save_to_disk`.

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/harmful_behaviors/train")

ds = ds.add_item({"text": "Мой новый промпт"})          # добавить одну строку
ds = ds.filter(lambda r: "bank" not in r["text"])       # удалить строки по условию
ds = ds.map(lambda r: {"text": r["text"].strip()})      # преобразовать колонку

# Сохранить ОБРАТНО в тот же формат .arrow.
# Важно: нельзя писать save_to_disk в ту же папку, из которой читали (конфликт memory-map).
# Сохраняем во временную папку и заменяем.
ds.save_to_disk("datasets/harmful_behaviors/train_new")
```

Затем в терминале заменить старую папку новой:

```bash
rm -rf datasets/harmful_behaviors/train
mv datasets/harmful_behaviors/train_new datasets/harmful_behaviors/train
```

## 7. Экспорт в удобные форматы и импорт обратно

Чтобы смотреть/править данные глазами — выгрузи в CSV / JSONL / TXT:

```python
ds.to_csv("harmful_train.csv")       # открыть в Excel / LibreOffice / VS Code
ds.to_json("harmful_train.jsonl")    # по одному JSON-объекту на строку

# Простой текст: 1 промпт = 1 строка (самый удобный для ручной правки).
with open("harmful_train.txt", "w", encoding="utf-8") as f:
    for row in ds["text"]:
        f.write(row + "\n")
```

Импорт обратно из файла в `Dataset`:

```python
from datasets import load_dataset

ds = load_dataset("csv",  data_files="harmful_train.csv")["train"]
ds = load_dataset("json", data_files="harmful_train.jsonl")["train"]
# затем при желании: ds.save_to_disk("datasets/harmful_behaviors/train_new")
```

## 8. 🎯 ГЛАВНОЕ: как сделать чтение ЛОКАЛЬНЫМ (без выгрузки из сети)

По умолчанию поля датасетов в проекте указывают на **Hugging Face ID** (`mlabonne/...`),
поэтому при запуске данные скачиваются из интернета. Чтобы проект читал **локальные** данные
из `datasets/` и никогда ничего не качал, есть два способа.

> Как это работает внутри: функция `load_prompts()` (`src/heretic/utils.py:224`) сначала
> проверяет `is_hf_path()` (`utils.py:189`). Если путь **существует на диске**, он считается
> локальным (а не HF-ID). Затем при наличии `state.json` в папке срабатывает ветка
> локальной загрузки `load_from_disk` (`utils.py:256`). Именно поэтому важно указывать путь
> на **split-подпапку** (`datasets/.../train`), где лежит `state.json`.

### Вариант A — через `config.toml` (без изменения кода, рекомендуется)

Создай файл `config.toml` в корне проекта (или дополни существующий) секциями:

```toml
[good_prompts]
dataset = "datasets/harmless_alpaca/train"
split   = "train[:400]"
column  = "text"

[bad_prompts]
dataset = "datasets/harmful_behaviors/train"
split   = "train[:400]"
column  = "text"

[good_evaluation_prompts]
dataset = "datasets/harmless_alpaca/test"
split   = "test[:100]"
column  = "text"

[bad_evaluation_prompts]
dataset = "datasets/harmful_behaviors/test"
split   = "test[:100]"
column  = "text"
```

Плюсы: не трогает исходный код. Минус: `config.toml` включён в `.gitignore` (не коммитится).

### Вариант B — изменить значения по умолчанию в коде (постоянно, «в проекте»)

Место: `src/heretic/config.py`, класс `Settings`, поля `good_prompts`, `bad_prompts`,
`good_evaluation_prompts`, `bad_evaluation_prompts` (строки ~475–513). Меняется только поле
`dataset` (со split для однозначности):

| Поле | Было (HF-ID) | Стало (локально) | `split` |
| :--- | :--- | :--- | :--- |
| `good_prompts` | `mlabonne/harmless_alpaca` | `datasets/harmless_alpaca/train` | `train[:400]` |
| `bad_prompts` | `mlabonne/harmful_behaviors` | `datasets/harmful_behaviors/train` | `train[:400]` |
| `good_evaluation_prompts` | `mlabonne/harmless_alpaca` | `datasets/harmless_alpaca/test` | `test[:100]` |
| `bad_evaluation_prompts` | `mlabonne/harmful_behaviors` | `datasets/harmful_behaviors/test` | `test[:100]` |

Пример правки блока `good_prompts` в `config.py`:

```python
    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="datasets/harmless_alpaca/train",   # было: "mlabonne/harmless_alpaca"
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        ...
    )
```

Аналогично для остальных трёх полей. Плюс: локальное чтение становится поведением проекта
по умолчанию и коммитится в репозиторий — выгрузка из сети больше не нужна никогда.

### ⚠️ Важно про относительные пути

Пути вроде `datasets/harmless_alpaca/train` — **относительные к директории, из которой
запускается `heretic`**. Поэтому:

- запускай Heretic **из корня проекта**, либо
- используй **абсолютные пути** (например, `/media/.../heretic/datasets/harmless_alpaca/train`),
  если запускаешь из другого места.

Если путь не будет существовать в момент запуска, Heretic сочтёт его за HF-ID и попытается
скачать из сети.

## 9. Как расширять датасеты (добавлять новые промпты)

### Способ 1 — через текстовый файл (самый простой)

Heretic умеет читать обычный `.txt` напрямую: **1 промпт = 1 строка**, пустые строки
игнорируются (`src/heretic/utils.py:231-241`). Порядок:

1. Выгрузи split в `.txt` (см. раздел 7) или создай файл с нуля.
2. Добавь/удали строки в любом редакторе.
3. Укажи `.txt` в конфиге (`column` не нужен, `split` необязателен):

```toml
[bad_prompts]
dataset = "datasets/harmful_behaviors/train.txt"
split   = "[:400]"     # опционально: взять первые 400 строк
```

### Способ 2 — программно, через `datasets` (сохраняя формат `.arrow`)

```python
from datasets import load_from_disk, concatenate_datasets, Dataset

base = load_from_disk("datasets/harmful_behaviors/train")

# Добавить список новых промптов:
new_prompts = ["Промпт 1", "Промпт 2", "Промпт 3"]
extra = Dataset.from_dict({"text": new_prompts})

# Схемы (колонки) должны совпадать: у обоих колонка "text".
merged = concatenate_datasets([base, extra])
print(base.num_rows, "->", merged.num_rows)

merged.save_to_disk("datasets/harmful_behaviors/train_new")
# затем заменить папку (см. раздел 6)
```

### Способ 3 — добавить полностью свой датасет

Создай новую директорию `datasets/my_prompts/` и сохрани туда `Dataset` через `save_to_disk`
(или положи текстовый файл), затем укажи путь в конфиге. Помни: у датасета должна быть
колонка `text` (или укажи свою в поле `column`).

## 10. Как заново скачать / обновить датасеты из сети

Если нужно перекачать данные (например, на новой машине), запусти скачивание через `venv3.12`.
Мини-скрипт `download_datasets.py`:

```python
from datasets import load_dataset

DATASETS = [
    ("mlabonne/harmless_alpaca",  "datasets/harmless_alpaca"),
    ("mlabonne/harmful_behaviors", "datasets/harmful_behaviors"),
]

for hf_id, target_dir in DATASETS:
    ds = load_dataset(hf_id)          # DatasetDict со всеми split-ами
    ds.save_to_disk(target_dir)       # создаст подпапки train/ test/
    print(f"{hf_id} -> {target_dir}")
```

```bash
./venv3.12/bin/python download_datasets.py
```

## 11. Частые вопросы и подводные камни

- **Путь указывает на корень датасета, а не на split** → ошибка. Указывай `.../train` или `.../test`
  (там лежит `state.json`).
- **Нет `column`/`split` для локального датасета** → `load_prompts` бросит ошибку: для
  датасетов (не txt) оба поля обязательны.
- **`save_to_disk` в ту же папку, откуда читали** → конфликт memory-map. Сохраняй во временную
  папку и заменяй.
- **Запуск не из корня проекта** → относительный путь не найдётся, Heretic полезет в сеть.
  Используй абсолютный путь или запускай из корня.
- **Правка `.arrow` в текстовом редакторе** → бесполезно, это бинарный формат. Используй
  `datasets`/pandas или экспорт в `.txt`/`.csv`.
- **`datasets/` и `venv3.12/` в git** → сейчас не игнорируются. Реши сам: данные можно
  коммитить (чтобы репозиторий был самодостаточным), окружение `venv3.12/` обычно не коммитят.

---

# 🇬🇧 Part 2. English version

## 1. What these datasets are and where they live

The project needs **two prompt groups** (the method builds the "refusal direction" as their difference):

| Directory | Source (Hugging Face) | Role |
| :--- | :--- | :--- |
| `datasets/harmless_alpaca/` | `mlabonne/harmless_alpaca` | 🟢 "harmless" prompts (no refusals) |
| `datasets/harmful_behaviors/` | `mlabonne/harmful_behaviors` | 🔴 "harmful" prompts (refusals) |

Each directory contains split subfolders, downloaded and saved locally:

```
datasets/
├── HOW_TO_USE_DATASETS.MD        ← this file
├── harmless_alpaca/
│   ├── train/                    ← 25058 rows, column "text"
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── test/                     ← 6265 rows
└── harmful_behaviors/
    ├── train/                    ← 416 rows, column "text"
    └── test/                     ← 104 rows
```

- The prompt column in all sets is **`text`**.
- `train` is used to compute the refusal direction, `test` is used for quality evaluation.

## 2. The `.arrow` format — what it is

`data-00000-of-00001.arrow` is **Apache Arrow**: a binary **columnar** table format that the
Hugging Face `datasets` library uses under the hood.

- **Columnar** — data is stored by column → fast column operations.
- **Memory-mapped** — read from disk on demand, not fully loaded into RAM.
- **Typed** — the schema (column types) is stored with the data.
- **Binary** — opening it in a text editor is pointless; you need code or a dedicated tool.

The three files in each split folder work together:

| File | Purpose |
| :--- | :--- |
| `data-00000-of-00001.arrow` | the data itself (the table) |
| `dataset_info.json` | metadata: description, features, number of rows |
| `state.json` | which `.arrow` files form the dataset — **Heretic detects a local dataset by this file** |

## 3. Environment for working with datasets (`venv3.12`)

Reading/editing `.arrow` needs the `datasets` library. It is installed in a separate `venv3.12`
environment in the project root. Run any code below with:

```bash
./venv3.12/bin/python your_script.py
# or interactively:
./venv3.12/bin/python
```

**If the environment is missing** (e.g. after cloning on another machine) — recreate it:

```bash
python3.12 -m venv venv3.12
./venv3.12/bin/python -m pip install --upgrade pip
./venv3.12/bin/python -m pip install "datasets>=3.0"
```

## 4. How to read a dataset

```python
from datasets import load_from_disk

# Load a split folder (the train/ or test/ subfolder, not the dataset root)
ds = load_from_disk("datasets/harmful_behaviors/train")

ds.num_rows            # 416 — number of rows
ds.column_names        # ['text'] — list of columns
ds.features            # {'text': Value('string')} — schema/types

ds[0]                  # {'text': '...'} — first row as a dict
ds['text'][0]          # column value in the first row
ds[:3]['text']         # list of the first 3 prompts
```

## 5. How to analyze (via pandas)

```python
df = ds.to_pandas()                          # full table -> DataFrame
df.shape                                     # (416, 1)
df['text'].str.len().mean()                  # average prompt length
df[df['text'].str.contains("hack")]          # search/filter by substring
df['text'].duplicated().sum()                # number of duplicates
```

## 6. How to edit and save back to `.arrow`

A `Dataset` object is **immutable** — each operation returns a new object that you must reassign
and then persist with `save_to_disk`.

```python
from datasets import load_from_disk

ds = load_from_disk("datasets/harmful_behaviors/train")

ds = ds.add_item({"text": "My new prompt"})             # add one row
ds = ds.filter(lambda r: "bank" not in r["text"])       # remove rows by condition
ds = ds.map(lambda r: {"text": r["text"].strip()})      # transform the column

# Save BACK to the same .arrow format.
# Note: do not save_to_disk into the same folder you read from (memory-map conflict).
# Save to a temporary folder and replace.
ds.save_to_disk("datasets/harmful_behaviors/train_new")
```

Then replace the old folder in the terminal:

```bash
rm -rf datasets/harmful_behaviors/train
mv datasets/harmful_behaviors/train_new datasets/harmful_behaviors/train
```

## 7. Export to convenient formats and import back

To view/edit data by hand — export to CSV / JSONL / TXT:

```python
ds.to_csv("harmful_train.csv")       # open in Excel / LibreOffice / VS Code
ds.to_json("harmful_train.jsonl")    # one JSON object per line

# Plain text: 1 prompt = 1 line (the most convenient for manual editing).
with open("harmful_train.txt", "w", encoding="utf-8") as f:
    for row in ds["text"]:
        f.write(row + "\n")
```

Import back from a file into a `Dataset`:

```python
from datasets import load_dataset

ds = load_dataset("csv",  data_files="harmful_train.csv")["train"]
ds = load_dataset("json", data_files="harmful_train.jsonl")["train"]
# then optionally: ds.save_to_disk("datasets/harmful_behaviors/train_new")
```

## 8. 🎯 KEY: how to make reading LOCAL (no download from the network)

By default the dataset fields in the project point to **Hugging Face IDs** (`mlabonne/...`),
so data is downloaded from the internet on launch. To make the project read **local** data from
`datasets/` and never download anything, there are two ways.

> How it works internally: `load_prompts()` (`src/heretic/utils.py:224`) first checks
> `is_hf_path()` (`utils.py:189`). If the path **exists on disk**, it is treated as local
> (not an HF ID). Then, if the folder contains `state.json`, the local-loading branch
> `load_from_disk` fires (`utils.py:256`). That is why you must point to a **split subfolder**
> (`datasets/.../train`), where `state.json` lives.

### Option A — via `config.toml` (no code change, recommended)

Create `config.toml` in the project root (or extend the existing one) with these sections:

```toml
[good_prompts]
dataset = "datasets/harmless_alpaca/train"
split   = "train[:400]"
column  = "text"

[bad_prompts]
dataset = "datasets/harmful_behaviors/train"
split   = "train[:400]"
column  = "text"

[good_evaluation_prompts]
dataset = "datasets/harmless_alpaca/test"
split   = "test[:100]"
column  = "text"

[bad_evaluation_prompts]
dataset = "datasets/harmful_behaviors/test"
split   = "test[:100]"
column  = "text"
```

Pros: does not touch the source code. Con: `config.toml` is in `.gitignore` (not committed).

### Option B — change the defaults in the code (permanent, "in the project")

Location: `src/heretic/config.py`, class `Settings`, fields `good_prompts`, `bad_prompts`,
`good_evaluation_prompts`, `bad_evaluation_prompts` (lines ~475–513). Only the `dataset` field
changes (with its split for disambiguation):

| Field | Was (HF ID) | Becomes (local) | `split` |
| :--- | :--- | :--- | :--- |
| `good_prompts` | `mlabonne/harmless_alpaca` | `datasets/harmless_alpaca/train` | `train[:400]` |
| `bad_prompts` | `mlabonne/harmful_behaviors` | `datasets/harmful_behaviors/train` | `train[:400]` |
| `good_evaluation_prompts` | `mlabonne/harmless_alpaca` | `datasets/harmless_alpaca/test` | `test[:100]` |
| `bad_evaluation_prompts` | `mlabonne/harmful_behaviors` | `datasets/harmful_behaviors/test` | `test[:100]` |

Example edit of the `good_prompts` block in `config.py`:

```python
    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="datasets/harmless_alpaca/train",   # was: "mlabonne/harmless_alpaca"
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        ...
    )
```

Do the same for the other three fields. Pro: local reading becomes the project's default behavior
and is committed to the repository — no network download is ever needed again.

### ⚠️ Important note about relative paths

Paths like `datasets/harmless_alpaca/train` are **relative to the directory from which `heretic`
is launched**. Therefore:

- run Heretic **from the project root**, or
- use **absolute paths** (e.g. `/media/.../heretic/datasets/harmless_alpaca/train`) if you launch
  from elsewhere.

If the path does not exist at launch time, Heretic will treat it as an HF ID and try to download it.

## 9. How to extend the datasets (add new prompts)

### Method 1 — via a text file (the simplest)

Heretic can read a plain `.txt` directly: **1 prompt = 1 line**, empty lines are ignored
(`src/heretic/utils.py:231-241`). Steps:

1. Export a split to `.txt` (see section 7) or create the file from scratch.
2. Add/remove lines in any editor.
3. Point the config at the `.txt` (`column` not needed, `split` optional):

```toml
[bad_prompts]
dataset = "datasets/harmful_behaviors/train.txt"
split   = "[:400]"     # optional: take the first 400 lines
```

### Method 2 — programmatically, via `datasets` (keeping the `.arrow` format)

```python
from datasets import load_from_disk, concatenate_datasets, Dataset

base = load_from_disk("datasets/harmful_behaviors/train")

# Add a list of new prompts:
new_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
extra = Dataset.from_dict({"text": new_prompts})

# Schemas (columns) must match: both have the "text" column.
merged = concatenate_datasets([base, extra])
print(base.num_rows, "->", merged.num_rows)

merged.save_to_disk("datasets/harmful_behaviors/train_new")
# then replace the folder (see section 6)
```

### Method 3 — add a completely new dataset

Create a new directory `datasets/my_prompts/` and save a `Dataset` there via `save_to_disk`
(or drop in a text file), then point the config at the path. Remember: the dataset must have a
`text` column (or specify your own in the `column` field).

## 10. How to re-download / update the datasets from the network

If you need to re-download the data (e.g. on a new machine), run the download via `venv3.12`.
A tiny `download_datasets.py` script:

```python
from datasets import load_dataset

DATASETS = [
    ("mlabonne/harmless_alpaca",  "datasets/harmless_alpaca"),
    ("mlabonne/harmful_behaviors", "datasets/harmful_behaviors"),
]

for hf_id, target_dir in DATASETS:
    ds = load_dataset(hf_id)          # DatasetDict with all splits
    ds.save_to_disk(target_dir)       # creates train/ test/ subfolders
    print(f"{hf_id} -> {target_dir}")
```

```bash
./venv3.12/bin/python download_datasets.py
```

## 11. FAQ and pitfalls

- **Path points to the dataset root instead of a split** → error. Point to `.../train` or
  `.../test` (that is where `state.json` lives).
- **No `column`/`split` for a local dataset** → `load_prompts` raises an error: for datasets
  (not txt) both fields are mandatory.
- **`save_to_disk` into the same folder you read from** → memory-map conflict. Save to a temporary
  folder and replace.
- **Launching not from the project root** → the relative path is not found and Heretic goes to the
  network. Use an absolute path or launch from the root.
- **Editing `.arrow` in a text editor** → pointless, it is a binary format. Use `datasets`/pandas
  or export to `.txt`/`.csv`.
- **`datasets/` and `venv3.12/` in git** → currently not ignored. Decide for yourself: the data can
  be committed (to make the repo self-contained), the `venv3.12/` environment is usually not committed.
