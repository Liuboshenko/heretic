# HOW TO USE — Heretic

> Практическое руководство по запуску проекта **Heretic** (`heretic-llm`) — инструмента
> для полностью автоматического снятия цензуры («safety alignment») с языковых моделей.
>
> Документ состоит из двух частей: сначала **на русском языке**, затем **на английском**.

---

# 🇷🇺 Часть 1. Русская версия

## 1. Что это за проект

**Heretic** — CLI-инструмент для автоматической «абляции» (directional ablation) трансформерных
языковых моделей. Он удаляет из весов модели «направление отказа» (refusal direction) **без дообучения**,
а оптимальные параметры подбирает сам с помощью байесовского оптимизатора (Optuna TPE), одновременно
минимизируя число отказов и KL-дивергенцию от оригинальной модели.

- Точка входа: команда `heretic`
- Управление зависимостями: **uv** (в репозитории есть `uv.lock`)
- Лицензия: AGPL-3.0-or-later

## 2. Требования к окружению

| Компонент | Требование |
| :--- | :--- |
| **Python** | `>= 3.10` (в проекте закреплён **3.12**; поддерживаются 3.10–3.13) |
| **PyTorch** | `>= 2.2` обязательно. Для моделей типа gpt-oss (MXFP4) — `>= 2.6`. Ставится **вручную** под ваше железо |
| **Ускоритель (GPU)** | Настоятельно рекомендуется. Поддержка: CUDA (NVIDIA), ROCm (AMD), XPU (Intel), MLU, SDAA, MUSA, NPU (Huawei), MPS (Apple). Без GPU работает, но очень медленно |
| **VRAM / RAM** | Под размер модели. Merge квантованной модели требует ~3× числа параметров в RAM |
| **Диск** | Под скачанные веса модели |
| **Интернет + Hugging Face** | Нужен для скачивания моделей и датасетов по умолчанию. Для выгрузки моделей нужен HF-токен. Можно работать и с локальными файлами |

> ⚠️ **Важно про PyTorch.** `torch` намеренно **не** входит в прямые зависимости пакета,
> потому что установка зависит от железа (CUDA / ROCm / CPU). При установке через `pip`
> сначала подготовьте окружение с PyTorch. При установке через `uv sync` из репозитория
> torch подтянется автоматически (он зафиксирован в `uv.lock` как транзитивная зависимость).

## 3. Установка

### Вариант A — через pip (простой)

Сначала подготовьте окружение с PyTorch, подходящим для вашего железа, затем:

```bash
pip install -U heretic-llm
```

### Вариант B — через uv из репозитория (воспроизводимый, рекомендуется)

Гарантирует те же версии пакетов, что использовали разработчики (из `uv.lock`):

```bash
git clone https://github.com/p-e-w/heretic.git
cd heretic
uv run heretic <модель>
```

### Дополнительно — research-возможности

Нужны только для флагов `--plot-residuals` и `--print-residual-geometry`
(подтягивает `matplotlib`, `pacmap`, `scikit-learn`, `geom-median`, `imageio`):

```bash
pip install -U "heretic-llm[research]"
# или через uv:
uv sync --all-extras
```

## 4. Быстрый старт

```bash
# через pip-установку
heretic Qwen/Qwen3-4B-Instruct-2507

# через uv
uv run heretic Qwen/Qwen3-4B-Instruct-2507
```

Замените `Qwen/Qwen3-4B-Instruct-2507` на любую модель (Hugging Face ID **или** путь к папке на диске).

Процесс полностью автоматический. Ориентир по времени: на **RTX 3090** обработка 4B-модели
с настройками по умолчанию (200 trials) занимает **~20–30 минут**.

## 5. Что происходит и что вам предложат в конце

После оптимизации Heretic покажет Pareto-оптимальные варианты (компромиссы «отказы ↔ KL-дивергенция»)
и предложит выбрать один. Для выбранного варианта доступны действия:

- **Save the model to a local folder** — сохранить модель в папку;
- **Upload the model to Hugging Face** — выгрузить в HF (нужен токен);
- **Chat with the model** — интерактивно проверить в чате;
- **Benchmark the model** — прогнать стандартные бенчмарки (MMLU, GSM8K, HellaSwag и др.).

При сохранении/выгрузке спрашивается стратегия экспорта: **merge** (влить LoRA в модель, полный файл)
или **adapter** (сохранить только LoRA-адаптер).

## 6. Конфигурация

Настройки читаются по приоритету (сверху вниз, верхнее переопределяет нижнее):

1. **Аргументы командной строки** — `heretic --n-trials 100 ...`
2. **Переменные окружения** с префиксом `HERETIC_` — например `HERETIC_QUANTIZATION=bnb_4bit`
3. **Файл `.env`**
4. **Файл `config.toml`** в рабочей директории

Способы посмотреть/задать параметры:

```bash
heretic --help          # полный список опций CLI
```

Или скопируйте шаблон конфига и отредактируйте под себя:

```bash
cp config.default.toml config.toml
```

> В репозитории есть готовые пресеты: `config.nohumor.toml`, `config.noslop.toml`.

### Полезные параметры

| Параметр (config.toml) | Флаг CLI | Назначение |
| :--- | :--- | :--- |
| `quantization = "bnb_4bit"` | `--quantization bnb_4bit` | 4-битное квантование → сильно меньше VRAM |
| `batch_size = 0` | `--batch-size 0` | Размер батча (0 = авто-подбор) |
| `n_trials = 200` | `--n-trials 200` | Число попыток оптимизации |
| `device_map = "auto"` | `--device-map auto` | Раскладка модели по устройствам |
| `max_memory` | — | Лимит памяти на устройство |
| `seed` | `--seed 42` | Фиксированный сид для воспроизводимости |
| `system_prompt` | `--system-prompt "..."` | Системный промпт |

## 7. Дополнительные режимы запуска

```bash
# Оценить готовую (чужую) абилитерированную модель относительно оригинала
heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic

# Воспроизвести модель по её reproduce.json (файл или URL)
heretic --reproduce <путь-или-URL к reproduce.json>

# Собрать все reproduce.json из публичных Heretic-репозиториев на HF в папку
heretic --collect-reproducibles <папка>

# [research] Напечатать таблицу геометрии residual-векторов
heretic <модель> --print-residual-geometry

# [research] Построить PaCMAP-визуализации residual-векторов (PNG + GIF)
heretic <модель> --plot-residuals
```

## 8. Советы и типичные проблемы

- **Мало VRAM** → включите `quantization = "bnb_4bit"`.
- **Ошибка загрузки модели по dtype** → Heretic сам перебирает `auto → float16 → bfloat16 → float32`.
- **Нет GPU** → всё будет работать медленно; предупреждение об этом выводится при старте.
- **Прерванный запуск** → прогресс сохраняется в `checkpoints/`; при повторном запуске той же модели
  предложат продолжить или начать заново.
- **Локальные модели/датасеты** → вместо HF-ID можно указывать путь к папке или к текстовому файлу
  (один промпт на строку).
- **Артефакты в .gitignore**: `config.toml`, `checkpoints/`, `plots/` — не коммитятся.

---

# 🇬🇧 Part 2. English version

## 1. What this project is

**Heretic** is a CLI tool that performs automatic directional ablation ("abliteration") of
transformer language models. It removes the "refusal direction" from the model weights
**without any post-training**, and automatically finds the best ablation parameters using a
Bayesian optimizer (Optuna TPE), co-minimizing the number of refusals and the KL divergence
from the original model.

- Entry point: the `heretic` command
- Dependency management: **uv** (the repository ships a `uv.lock`)
- License: AGPL-3.0-or-later

## 2. Environment requirements

| Component | Requirement |
| :--- | :--- |
| **Python** | `>= 3.10` (pinned to **3.12** in the repo; 3.10–3.13 supported) |
| **PyTorch** | `>= 2.2` required. For models like gpt-oss (MXFP4) — `>= 2.6`. Install it **manually** for your hardware |
| **Accelerator (GPU)** | Strongly recommended. Supported: CUDA (NVIDIA), ROCm (AMD), XPU (Intel), MLU, SDAA, MUSA, NPU (Huawei), MPS (Apple). Works without a GPU, but slowly |
| **VRAM / RAM** | Depends on model size. Merging a quantized model needs ~3× the parameter count in RAM |
| **Disk** | For the downloaded model weights |
| **Internet + Hugging Face** | Needed to download the default models and datasets. Uploading models needs an HF token. Local files also work |

> ⚠️ **Note about PyTorch.** `torch` is intentionally **not** a direct dependency of the package,
> because installation is hardware-specific (CUDA / ROCm / CPU). If you install via `pip`, prepare a
> PyTorch-enabled environment first. If you install via `uv sync` from the repository, torch is pulled
> in automatically (it is pinned in `uv.lock` as a transitive dependency).

## 3. Installation

### Option A — via pip (simple)

First prepare an environment with PyTorch appropriate for your hardware, then:

```bash
pip install -U heretic-llm
```

### Option B — via uv from the repository (reproducible, recommended)

Guarantees the exact package versions used by the developers (from `uv.lock`):

```bash
git clone https://github.com/p-e-w/heretic.git
cd heretic
uv run heretic <model>
```

### Optional — research features

Only needed for the `--plot-residuals` and `--print-residual-geometry` flags
(pulls in `matplotlib`, `pacmap`, `scikit-learn`, `geom-median`, `imageio`):

```bash
pip install -U "heretic-llm[research]"
# or with uv:
uv sync --all-extras
```

## 4. Quick start

```bash
# with the pip install
heretic Qwen/Qwen3-4B-Instruct-2507

# with uv
uv run heretic Qwen/Qwen3-4B-Instruct-2507
```

Replace `Qwen/Qwen3-4B-Instruct-2507` with any model (a Hugging Face ID **or** a path to a local folder).

The process is fully automatic. Rough timing: on an **RTX 3090**, processing a 4B model with the
default settings (200 trials) takes **~20–30 minutes**.

## 5. What happens and what you are offered at the end

After optimization Heretic shows the Pareto-optimal trials (trade-offs between "refusals ↔ KL divergence")
and lets you pick one. For the chosen trial the following actions are available:

- **Save the model to a local folder**
- **Upload the model to Hugging Face** (requires a token)
- **Chat with the model** — interactively test it
- **Benchmark the model** — run standard benchmarks (MMLU, GSM8K, HellaSwag, etc.)

When saving/uploading you are asked for an export strategy: **merge** (fold the LoRA into the model,
full weights) or **adapter** (save only the LoRA adapter).

## 6. Configuration

Settings are resolved by priority (top overrides bottom):

1. **Command-line arguments** — `heretic --n-trials 100 ...`
2. **Environment variables** prefixed with `HERETIC_` — e.g. `HERETIC_QUANTIZATION=bnb_4bit`
3. **`.env` file**
4. **`config.toml`** in the working directory

Ways to view/set parameters:

```bash
heretic --help          # full list of CLI options
```

Or copy the config template and edit it:

```bash
cp config.default.toml config.toml
```

> The repository also ships ready-made presets: `config.nohumor.toml`, `config.noslop.toml`.

### Useful parameters

| Parameter (config.toml) | CLI flag | Purpose |
| :--- | :--- | :--- |
| `quantization = "bnb_4bit"` | `--quantization bnb_4bit` | 4-bit quantization → much less VRAM |
| `batch_size = 0` | `--batch-size 0` | Batch size (0 = auto) |
| `n_trials = 200` | `--n-trials 200` | Number of optimization trials |
| `device_map = "auto"` | `--device-map auto` | Model placement across devices |
| `max_memory` | — | Per-device memory limit |
| `seed` | `--seed 42` | Fixed seed for reproducibility |
| `system_prompt` | `--system-prompt "..."` | System prompt |

## 7. Additional run modes

```bash
# Evaluate an existing (third-party) abliterated model against the original
heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic

# Reproduce a model from its reproduce.json (file or URL)
heretic --reproduce <path-or-URL to reproduce.json>

# Collect all reproduce.json files from public Heretic repos on HF into a folder
heretic --collect-reproducibles <folder>

# [research] Print a table of residual-vector geometry
heretic <model> --print-residual-geometry

# [research] Generate PaCMAP visualizations of residual vectors (PNG + GIF)
heretic <model> --plot-residuals
```

## 8. Tips and common issues

- **Low VRAM** → enable `quantization = "bnb_4bit"`.
- **dtype loading error** → Heretic automatically tries `auto → float16 → bfloat16 → float32`.
- **No GPU** → everything runs slowly; a warning is printed at startup.
- **Interrupted run** → progress is saved in `checkpoints/`; re-running the same model lets you
  continue or start over.
- **Local models/datasets** → instead of an HF ID you can pass a folder path or a plain text file
  (one prompt per line).
- **Ignored artifacts**: `config.toml`, `checkpoints/`, `plots/` are git-ignored.
