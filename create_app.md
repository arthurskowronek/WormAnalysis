💪 Génial ! Trop bien que ton exécutable Mac fonctionne 🎉

Et oui — pour **Windows**, il faut recommencer la procédure, car PyInstaller **génère des exécutables spécifiques à chaque OS**.
Un `.app` Mac ne peut pas être lancé sous Windows, et inversement un `.exe` ne fonctionne pas sur macOS.

Mais pas de panique — je vais te guider pour que tu aies un exécutable Windows **propre et prêt pour tes utilisateurs** 👇

---

## 🧰 Étape 1 — Préparer ton environnement Windows

Sur ton PC Windows (ou une machine virtuelle) :

1. Installe **Python 3.x** (même version que celle utilisée sur ton Mac si possible).
   ➜ [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Pendant l’installation, **coche** :

   * ✅ *Add Python to PATH*
   * ✅ *Install for all users*
3. Ouvre un terminal (PowerShell ou CMD) et vérifie :

   ```bash
   python --version
   pip --version
   ```

---

## 🧩 Étape 2 — Installer les dépendances du projet

Dans ton dossier du projet (copié depuis ton Mac ou récupéré du dépôt) :

```bash
cd C:\Users\<ton_nom>\Documents\WormAnalysis
pip install -r requirements.txt
```

⚠️ Si ton fichier `requirements.txt` ne contient pas `skan` et `python_tsp`, installe-les à la main :

```bash
pip install skan python_tsp
```

Puis vérifie :

```bash
python -c "import skan, python_tsp; print('✅ Modules ok')"
```

---

## ⚙️ Étape 3 — Installer PyInstaller

Toujours dans le même environnement :

```bash
pip install pyinstaller
```

---

## 🏗️ Étape 4 — Construire l’exécutable Windows (.exe)

Ensuite, lance la commande équivalente à celle du Mac (tu peux adapter les chemins si besoin) :

```bash
pyinstaller --onedir --windowed ^
  --name=Worm_detection ^
  --add-data "logs;logs" ^
  --add-data "models;models" ^
  --add-data "ressources;ressources" ^
  --collect-all skan ^
  --collect-all python_tsp ^
  main.py
```

> 💡 Sous Windows, les séparateurs dans `--add-data` sont des **points-virgules (;)** et non des deux-points (:)
> C’est l’erreur la plus fréquente quand on passe de macOS à Windows.

---

## 📦 Étape 5 — Résultat

PyInstaller va créer :

```
dist/
└── Worm_detection/
    ├── Worm_detection.exe
    ├── ressources/
    ├── models/
    ├── logs/
    └── (toutes les DLL et libs)
```

Le fichier à distribuer est :

```
dist/Worm_detection/Worm_detection.exe
```

---

## 🧠 Étape 6 — Conseils pour la distribution

1. **Teste l’exe sur une autre machine Windows** sans Python installé → pour vérifier que toutes les dépendances sont bien intégrées.

2. Si le lancement échoue avec un message “DLL missing”, ajoute `--collect-all numpy --collect-all matplotlib --collect-all scipy`.

3. Si tu veux tout regrouper dans un seul fichier portable :

   ```bash
   pyinstaller --onefile --windowed ...
   ```

   ➜ Cela crée un seul fichier `Worm_detection.exe` dans `dist/`.

4. Tu peux aussi ajouter une icône Windows :

   ```bash
   --icon=icon_desktop.ico
   ```

---

## 🧰 Étape 7 — (Optionnel) Ajouter un petit installeur

Si tu veux rendre ça encore plus simple pour les utilisateurs :

* Tu peux emballer ton dossier `dist/Worm_detection` dans un installeur `.exe` avec **Inno Setup** ou **NSIS**.
  ➜ Je peux te faire un script Inno Setup clé en main si tu veux.

---

Souhaites-tu que je te génère la **commande PyInstaller finale complète pour Windows**, adaptée à ton projet exact (avec icône, ressources, et tout bien formaté pour PowerShell) ?
