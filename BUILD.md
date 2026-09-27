# Сборка материалов

## Автоматическая сборка на GitHub

Workflow `.github/workflows/build-pdfs.yml` запускается автоматически при изменениях в:

- `lectures/**/*.tex`;
- `lectures/**/generate_figures.py`;
- `requirements.txt`;
- самом workflow.

Его также можно запустить вручную через **Actions → Build lecture PDFs → Run workflow**.

Workflow:

1. устанавливает Python-зависимости и TeX Live;
2. загружает HSE Beamer theme из зафиксированного коммита внешнего шаблона;
3. генерирует рисунки для лекций 2–4;
4. собирает лекцию 1 через XeLaTeX, лекции 2–4 через pdfLaTeX;
5. загружает четыре PDF как GitHub Actions artifact;
6. при изменении PDF автоматически коммитит их в каталог `pdf/` ветки `main`.

Коммит, который обновляет только `pdf/*.pdf`, не запускает workflow повторно.

## Локальная сборка

Лекция 1 использует XeLaTeX. Лекции 2–4 используют pdfLaTeX.

Для лекций 2–4 сначала сгенерируйте рисунки:

```bash
python lectures/lecture02/generate_figures.py
python lectures/lecture03/generate_figures.py
python lectures/lecture04/generate_figures.py
```

Python-зависимости:

```bash
pip install -r requirements.txt
```

Для локальной LaTeX-сборки необходим каталог `HSE-theme/` рядом с соответствующим `.tex`-файлом. Автоматическая сборка GitHub Actions получает тему самостоятельно и не требует хранить её копию в репозитории.

Примеры:

```bash
cd lectures/lecture01
latexmk -xelatex -interaction=nonstopmode -halt-on-error lecture01_optimization_ml.tex
```

```bash
cd lectures/lecture04
latexmk -pdf -interaction=nonstopmode -halt-on-error lecture04_conditioning_gd.tex
```
