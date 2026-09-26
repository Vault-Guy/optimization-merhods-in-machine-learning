# Методы оптимизации в машинном обучении

Материалы курса **«Методы оптимизации в машинном обучении»** для совместной программы СМОЛ ГУ и ФКН НИУ ВШЭ, 2026/27 учебный год.

**Автор:** Никита Лукьяненко, старший преподаватель ФКН НИУ ВШЭ.

## Лекции

| № | Тема | Материалы |
|---|---|---|
| 1 | Оптимизация как язык машинного обучения | [LaTeX](lectures/lecture01/lecture01_optimization_ml.tex) |
| 2 | Геометрия оптимизации: производная, градиент и гессиан | [LaTeX](lectures/lecture02/lecture02_geometry.tex) · [графики](lectures/lecture02/generate_figures.py) |
| 3 | Квадратичная оптимизация: геометрия, спектр и точное решение | [LaTeX](lectures/lecture03/lecture03_quadratic.tex) · [графики](lectures/lecture03/generate_figures.py) |
| 4 | Обусловленность и градиентный спуск: почему алгоритм сходится? | [LaTeX](lectures/lecture04/lecture04_conditioning_gd.tex) · [графики](lectures/lecture04/generate_figures.py) |

## Семинары

| № | Тема | Задачи |
|---|---|---|
| 2 | Геометрия оптимизации | [8 задач](seminars/seminar02/tasks.md) |
| 3 | Квадратичная оптимизация, обусловленность и градиентный спуск | [10 задач](seminars/seminar03/tasks.md) |

## Домашние лабораторные

Каталог [`labs/`](labs/) подготовлен для домашних лабораторных работ. Условия, стартовые ноутбуки, данные и критерии оценивания будут добавляться по мере прохождения курса.

## Структура репозитория

```text
lectures/
  lecture01/
    lecture01_optimization_ml.tex
  lecture02/
    lecture02_geometry.tex
    generate_figures.py
  lecture03/
    lecture03_quadratic.tex
    generate_figures.py
  lecture04/
    lecture04_conditioning_gd.tex
    generate_figures.py
seminars/
  seminar02/tasks.md
  seminar03/tasks.md
labs/
```

## Сборка

Подробные инструкции находятся в [BUILD.md](BUILD.md).

Лекция 1 рассчитана на **XeLaTeX**, лекции 2–4 — на **pdfLaTeX**. Для оформления используется HSE Beamer theme через `HSE-theme/beamerthemeHSE.sty`. Графики для лекций 2–4 генерируются соответствующими Python-скриптами.

Python-зависимости перечислены в [requirements.txt](requirements.txt).

## Статус курса

Репозиторий обновляется в течение 2026/27 учебного года. Здесь хранятся актуальные версии уже подготовленных лекций и семинарских заданий; лабораторные и следующие темы будут добавляться по мере готовности.
