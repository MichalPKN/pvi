# Počítačové vidění

## Přednášky

| #  | datum      | přednáška                                                       | obsah                                                              |
|----|------------|-----------------------------------------------------------------|--------------------------------------------------------------------|
| 1. | 24.9.2026  | [Úvod](slides/pvi-01-intro.pdf)                                 | podmínky, úvod, pořízení obrazu, barevné prostory                  |
| 2. | 1.10.2026  | [Lokální zájmové body a oblasti I](slides/pvi-02-keypoints.pdf) | Harris, LoG, DoG                                                   |
| 3. | 8.10.2026  | [Lokální zájmové body a oblasti II](slides/pvi-03-features.pdf) | SIFT, approx. NN, SURF, MSER, ORB a další                          |
| 4. | 15.10.2026 | [Zarovnání obrazů](slides/pvi-04-registration.pdf)              | geometrické transformace a odhad, RANSAC, Hough, detekce, panorama |


## Cvičení

- Odevzdání úlohy po termínu bude penalizováno odečtením 50 % bodů.
- Další podmínky a známkování jsou v přednášce [Úvod](slides/ans-00-intro.pdf).

| #  | úloha                                      | body    | deadline   |
|----|--------------------------------------------|---------|------------|
| 1. | [colors](assignments/colors.ipynb)         | 17      | 28.10.2026 |
| 2. | [sunflowers](assignments/sunflowers.ipynb) | 21 (+9) | 8.11.2026  |


## Nastavení

### 1. Balíčkovací manažer
- Předmět využívá [uv](https://docs.astral.sh/uv/) jako manažer balíčků.
- Lze jej nainstalovat příkazy v následující tabulce.

| Platforma                           | Příkaz                                                                                |
|-------------------------------------|---------------------------------------------------------------------------------------|
| Windows (PowerShell)                | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Windows (winget)                    | `winget install --id=astral-sh.uv -e`                                                 |
| Ubuntu                              | `sudo snap install astral-uv`                                                         |
| macOS (brew)                        | `brew install uv`                                                                     |
| Ostatní Linuxové distribuce a macOS | `wget -qO- https://astral.sh/uv/install.sh \| sh`                                     |

### 2. Virtuální prostředí a instalace potřebných balíků
- Potřebné balíčky a knihovny jsou uvedeny v souborech `pyproject.toml` a `uv.lock`.
- Vytvořte nový adresář, vložte do něj oba výše uvedené soubory, spusťte v něm příkazový řadek a zadejte  
  ```
  uv sync
  ```
- Příkaz vytvoří virtuální prostředí v pod-adresáři `./.venv` a do něj nainstaluje všechny potřebné balíčky.
- Všechny skripty se ve vytvořeném virtuáním prostředí spustí jako  
  ```
  uv run nazev_skriptu.py
  ```

### 3. Obsah předmětu
- Úlohy jsou ve formě [Jupyter notebooks](https://jupyter.org/).
- Doporučený nástroj na práci s kódy je [Visual Studio Code](https://code.visualstudio.com/) s rozšířeními [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) a [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
