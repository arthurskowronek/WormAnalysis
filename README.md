# WormAnalysis

Projet Python avec interface graphique (`tkinter`), nécessitant un environnement Python 3.13+ configuré avec `tkinter`.

---

## ⚡ Installation rapide (Linux / macOS)

> Nécessite [Homebrew](https://brew.sh) sur macOS, et `pyenv` sur Linux/macOS.

```bash
curl -sSL https://raw.githubusercontent.com/arthurskowronek/WormAnalysis/main/setup.sh | bash
```

---

## 🪟 Installation sous Windows

Ce projet utilise `tkinter`, qui est inclus par défaut avec Python sous Windows, **si vous avez coché l’option "tkinter" lors de l’installation** de Python.

### ✅ Étapes à suivre

Start by verifying if the computer already has a good version of python using : python --version

1. **Installer Python 3.12+**

   * Rendez-vous sur [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
   * Téléchargez l’installeur de **Python 3.12.x (64-bit)**
   * Lors de l’installation :

     * ✅ Cochez **“Add Python to PATH”**
     * ✅ Cliquez sur **“Customize installation”**, puis **vérifiez que "tcl/tk and IDLE" est bien coché** (c’est nécessaire pour `tkinter`)

2. **Créer un environnement virtuel `.venv`**

Ouvrez PowerShell (ou cmd), placez-vous dans le dossier du projet et exécutez :

```powershell
python -m venv .venv
```

3. **Activer l’environnement virtuel**

```powershell
.venv\Scripts\Activate.ps1
```

> Si vous avez une erreur liée à l'exécution de scripts PowerShell, tapez :
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

4. **Installer les dépendances**

```powershell
pip install -r requirements.txt
```

5. **Lancer le projet**

```powershell
python main.py
```


