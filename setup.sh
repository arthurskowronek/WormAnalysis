### ⚙️ `setup.sh` (automatisé pour macOS / Linux)

```bash
#!/bin/bash

set -e

# 🔍 Détection de l'OS
OS="$(uname)"
echo "🖥️  Système détecté : $OS"

# 📦 Dépendances pour macOS (Homebrew)
if [[ "$OS" == "Darwin" ]]; then
    echo "🍎 macOS : installation de pyenv et tcl-tk via Homebrew..."
    brew install pyenv tcl-tk

    export PATH="/opt/homebrew/opt/tcl-tk/bin:$PATH"
    export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"
    export PKG_CONFIG_PATH="/opt/homebrew/opt/tcl-tk/lib/pkgconfig"
fi

# 📦 Dépendances pour Linux (apt-based)
if [[ "$OS" == "Linux" ]]; then
    echo "🐧 Linux : vérification des dépendances..."
    if ! command -v pyenv &> /dev/null; then
        echo "❌ pyenv non trouvé. Veuillez installer pyenv."
        echo "➡️  https://github.com/pyenv/pyenv#installation"
        exit 1
    fi

    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
fi

# 📦 Installer Python avec tkinter
echo "🐍 Installation de Python 3.13.5 avec support Tkinter via pyenv..."
env \
  PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'" \
  pyenv install 3.13.5 --skip-existing

pyenv local 3.13.5

# ✅ Création de l’environnement virtuel
echo "📦 Création du virtualenv (.venv)..."
python -m venv .venv
source .venv/bin/activate

# 📦 Installation des dépendances
echo "📦 Installation des paquets de requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Installation terminée !"
echo "👉 Activez l’environnement avec : source .venv/bin/activate"
