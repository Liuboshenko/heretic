# Beyond Censorship Removal — Potential Applications

> 🌐 **Language / Язык:** [Русский](#часть-1--русская-версия) · [English](#part-2--english-version)

> A short, reasoned exploration of what Heretic's technology can be used for *beyond*
> its default job of removing refusals. The central claim: Heretic is not really a
> "censorship remover" — it is a **general-purpose behavioral-direction editor**, and
> refusal removal is just one instance of it.
>
> This document is bilingual: Russian first, then English.

---

# Часть 1 — Русская версия

## Ключевая переформулировка

Механизм Heretic на самом деле общий. Он:

1. берёт **две группы промптов** (контрастную пару),
2. вычисляет **направление** в пространстве активаций как разницу средних
   (`bad_means − good_means`),
3. **удаляет** (ортогонализует) это направление из пишущих в остаточный поток
   проекций (`o_proj`, `down_proj`),
4. а оптимизатор подбирает силу так, чтобы **минимизировать побочный урон**
   (KL-дивергенцию от исходной модели).

«Направление отказа» — лишь частный случай. **Любую** поведенческую ось, которую можно
выделить контрастной парой наборов промптов, можно так же локализовать и ослабить/убрать.

> 🔑 **Практический вывод:** чтобы перенацелить Heretic на другую задачу, достаточно
> заменить два датасета (`good_prompts` / `bad_prompts` в `config.py` / `config.toml`).
> Вся остальная машина — поиск направления, оптимизация, сохранение способностей через
> KL, экспорт — переиспользуется как есть. `good_prompts` = базовое/сбалансированное
> поведение, `bad_prompts` = поведение, которое нужно убрать.

## Применение №1 — борьба с асимметрией и предвзятостью суждений

Модель обучается на данных, где одни ассоциации встречаются чаще других. В результате её
«судейские» способности могут стать **асимметричными**: модель переносит наблюдаемую
частоту на логическую структуру и совершает ошибку **незаконного обращения** — путает
«некоторые A суть B» с «все A суть B», игнорирует базовые частоты (base-rate neglect) и
достраивает вывод по стереотипному приору, а не по фактам. Классический *структурный*
пример такой ошибки: из «многие X обладают признаком Y» модель некорректно выводит «Y ⇒ X».
Это не только этическая проблема — это **деградация аналитических способностей**: логическая
симметрия и беспристрастная оценка субъектов страдают.

Идея: если этот перекос представлен в активациях как направление, его можно **измерить и
ослабить** тем же методом. Контрастная пара:

- `good_prompts` — примеры, где модель рассуждает **симметрично и по фактам** (корректно
  различает «некоторые/все», учитывает базовые частоты);
- `bad_prompts` — примеры, где проявляется **асимметричный/стереотипный** вывод.

Разница средних даёт «направление предвзятости»; его абляция должна восстанавливать
сбалансированное аналитическое рассуждение при минимальном уроне остальным способностям
(что как раз контролирует KL-дивергенция).

> ⚠️ **Важное ограничение дизайна:** контрастные наборы нужно строить так, чтобы
> **единственным** систематическим различием между ними была целевая ось. Иначе разница
> средних захватит посторонний сигнал (тему, длину, формат), и абляция ударит не туда.

## Полный потенциал: Heretic как инструмент representation engineering

Обобщённо Heretic — это лёгкий (без дообучения) инструмент **редактирования представлений**
модели. Его сильные стороны:

- **Не требует градиентов и переобучения** — работает на инференсе за минуты.
- **Оптимизирует компромисс** «убрать поведение ↔ сохранить интеллект» автоматически.
- **Неразрушающий** — всё через LoRA, базовые веса не трогаются; результат обратим.
- **Архитектурно-широкий** — dense, многие MoE, мультимодальные, гибридные модели.

Границы применимости (о чём честно сказать в документе):

- Работает, когда поведение **приближённо линейно представлено** (difference-of-means
  захватывает ось). Сложные, нелинейно закодированные свойства уберутся хуже.
- Нужны **качественные контрастные наборы** — это главный источник успеха/провала.
- Не работает на архитектурах без пишущих проекций (**чистые state-space** модели).
- Всегда есть компромисс с KL: слишком агрессивная абляция вредит модели.
- **Двойное назначение:** тот же механизм может как убирать предвзятость, так и снимать
  защиту — исход определяется выбором контрастных наборов. Применять ответственно.

## 5 очевидных вариантов использования

1. **Снятие предвзятости / асимметрии суждений (вариант пользователя).** Убрать «направление
   стереотипа», восстановив симметричное, фактологичное рассуждение и аналитический баланс.
2. **Снятие цензуры / избыточного отказа (исходное назначение).** Для ресёрча, ред-тиминга,
   несцензурированных ассистентов.
3. **Удаление «слопа», подхалимства и штампов ассистента.** В репозитории уже есть пресеты
   `config.noslop.toml` и `config.nohumor.toml` — абляция «boilerplate-направления»,
   льстивости, чрезмерного хеджирования, навязчивого позитива.
4. **Снижение ложных отказов в легитимных доменах.** Медицина, право, безопасность,
   творческое письмо — где осторожная модель отказывает на безобидных запросах. Вернуть
   полезность без полного «джейлбрейка».
5. **Управление персоной/тоном.** Ослаблять или смещать оси формальности, многословия,
   юмора (`nohumor.toml`) — лёгкая настройка поведения без дообучения.

## 5 неочевидных вариантов использования

1. **Интерпретируемость и «микроскоп» для модели.** Использовать поиск направления и
   встроенные research-инструменты (`--print-residual-geometry`, `--plot-residuals`) не для
   абляции, а для **изучения** того, как понятие представлено по слоям (геометрия, силуэт,
   PaCMAP-проекции).
2. **Аудит и измерение (без удаления).** Величина разницы средних, силуэтный коэффициент и
   «стоимость удаления в KL» — это **метрики того, насколько сильно поведение «вшито»** в
   модель. Можно сравнивать модели/версии по степени предвзятости или зацензурированности.
3. **Калибровка уверенности / борьба с галлюцинациями.** Контраст «уверенно-неверных» и
   «калиброванно-неуверенных» ответов → найти и ослабить «направление избыточной
   уверенности» (гипотеза, но правдоподобная в рамках метода).
4. **Обнаружение и нейтрализация бэкдоров/отравления данных.** Бэкдор — это выученное
   направление, срабатывающее на триггер. Разница средних между «триггерными» и «чистыми»
   активациями может **локализовать и убрать» бэкдор без переобучения.
5. **Целевое «забывание» (machine unlearning) и диффинг моделей.** Убрать конкретную
   запомненную способность/знание (например, воспроизведение защищённого текста или PII-шаблона)
   с минимальным сопутствующим уроном под контролем KL; а также **сравнивать** два чекпоинта
   по их поведенческим направлениям.

## Как попробовать на практике

Для любого сценария — соберите две контрастные группы промптов и подставьте их как
локальные датасеты (см. `datasets/HOW_TO_USE_DATASETS.MD`):

```toml
[good_prompts]      # базовое/сбалансированное поведение
dataset = "datasets/balanced_reasoning/train"
split = "train[:400]"
column = "text"

[bad_prompts]       # поведение, которое хотим убрать
dataset = "datasets/biased_reasoning/train"
split = "train[:400]"
column = "text"
```

Дальше всё работает как обычно: `heretic <модель>`.

---

# Part 2 — English version

## The key reframing

Heretic's mechanism is actually general. It:

1. takes **two prompt groups** (a contrast pair),
2. computes a **direction** in activation space as a difference of means
   (`bad_means − good_means`),
3. **removes** (orthogonalizes) that direction out of the residual-stream write
   projections (`o_proj`, `down_proj`),
4. and an optimizer tunes the strength to **minimize collateral damage** (KL divergence
   from the original model).

The "refusal direction" is just one special case. **Any** behavioral axis that can be
isolated by a contrasting pair of prompt sets can be localized and attenuated/removed the
same way.

> 🔑 **Practical takeaway:** to repurpose Heretic for a different task, you only swap the
> two datasets (`good_prompts` / `bad_prompts` in `config.py` / `config.toml`). The rest of
> the machine — direction finding, optimization, KL-based capability preservation, export —
> is reused as-is. `good_prompts` = baseline/balanced behavior, `bad_prompts` = the behavior
> to remove.

## Application #1 — fighting asymmetric / biased judgment

A model trained on data where some associations appear more often than others can develop
**asymmetric** judgment: it transfers observed frequency onto logical structure and commits
the **illicit conversion** fallacy — confusing "some A are B" with "all A are B", neglecting
base rates, and completing an inference from a stereotype prior rather than from facts. A
classic *structural* form of this error: from "many X have property Y" the model incorrectly
concludes "Y ⇒ X". This is not only an ethical problem — it is a **degradation of analytical
ability**: logical symmetry and impartial assessment of subjects suffer.

The idea: if this skew is represented in the activations as a direction, it can be **measured
and attenuated** by the same method. The contrast pair:

- `good_prompts` — examples where the model reasons **symmetrically and factually**
  (correctly distinguishes "some/all", respects base rates);
- `bad_prompts` — examples exhibiting the **asymmetric/stereotyped** inference.

The difference of means yields a "bias direction"; ablating it should restore balanced
analytical reasoning while minimizing damage to other capabilities (exactly what the KL
divergence controls).

> ⚠️ **Important design constraint:** the contrast sets must be built so that the **only**
> systematic difference between them is the target axis. Otherwise the difference of means
> captures a spurious signal (topic, length, format) and the ablation hits the wrong thing.

## Full potential: Heretic as a representation-engineering tool

Generalized, Heretic is a lightweight (training-free) **representation-editing** tool. Its
strengths:

- **No gradients or retraining** — runs at inference in minutes.
- **Optimizes the trade-off** "remove behavior ↔ keep intelligence" automatically.
- **Non-destructive** — everything via LoRA; base weights are untouched; reversible.
- **Architecture-broad** — dense, many MoE, multimodal, hybrid models.

Limits of applicability (worth stating honestly):

- Works when the behavior is **approximately linearly represented** (difference-of-means
  captures the axis). Complex, nonlinearly encoded properties are removed less cleanly.
- Requires **good contrast sets** — the main source of success or failure.
- Does not work on architectures without write projections (**pure state-space** models).
- There is always a KL trade-off: overly aggressive ablation harms the model.
- **Dual-use:** the same mechanism can remove bias *or* remove safety — the outcome is
  decided by the choice of contrast sets. Use responsibly.

## 5 obvious use cases

1. **Removing biased / asymmetric judgment (the user's case).** Ablate the "stereotype
   direction", restoring symmetric, factual reasoning and analytical balance.
2. **Censorship / over-refusal removal (the original purpose).** For research, red-teaming,
   uncensored assistants.
3. **Removing "slop", sycophancy, and assistant clichés.** The repo already ships
   `config.noslop.toml` and `config.nohumor.toml` — ablate the "boilerplate direction",
   flattery, excessive hedging, forced positivity.
4. **Reducing false refusals in legitimate domains.** Medicine, law, security, creative
   writing — where a cautious model refuses benign requests. Restore utility without a full
   jailbreak.
5. **Persona / tone control.** Attenuate or steer axes of formality, verbosity, humor
   (`nohumor.toml`) — lightweight behavior tuning without fine-tuning.

## 5 non-obvious use cases

1. **Interpretability — a "microscope" for the model.** Use the direction finding and the
   built-in research tools (`--print-residual-geometry`, `--plot-residuals`) not to ablate,
   but to **study** how a concept is represented across layers (geometry, silhouette, PaCMAP
   projections).
2. **Auditing and measurement (without removal).** The magnitude of the difference of means,
   the silhouette coefficient, and the "KL cost of removal" are **metrics of how deeply a
   behavior is baked in**. You can compare models/versions by degree of bias or censorship.
3. **Confidence calibration / hallucination steering.** Contrast "confidently wrong" vs
   "calibrated-uncertain" answers → find and attenuate an "overconfidence direction"
   (speculative, but plausible within the method).
4. **Backdoor / data-poisoning detection and neutralization.** A backdoor is a learned
   direction triggered by a specific input. The difference of means between "triggered" and
   "clean" activations can **localize and remove** a backdoor without retraining.
5. **Targeted unlearning (machine unlearning) and model diffing.** Remove a specific
   memorized capability/knowledge (e.g., reproducing a copyrighted text or a PII pattern)
   with minimal collateral damage under KL control; and **compare** two checkpoints by their
   behavioral directions.

## How to try it in practice

For any scenario — assemble two contrasting prompt groups and plug them in as local datasets
(see `datasets/HOW_TO_USE_DATASETS.MD`):

```toml
[good_prompts]      # baseline / balanced behavior
dataset = "datasets/balanced_reasoning/train"
split = "train[:400]"
column = "text"

[bad_prompts]       # the behavior we want to remove
dataset = "datasets/biased_reasoning/train"
split = "train[:400]"
column = "text"
```

Then run as usual: `heretic <model>`.

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) / [ARCHITECTURE_RU.md](ARCHITECTURE_RU.md) for
how the pipeline works, and [../datasets/HOW_TO_USE_DATASETS.MD](../datasets/HOW_TO_USE_DATASETS.MD)
for building the contrast datasets.*
