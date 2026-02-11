# WormAnalysis 🪱🔬

**Author**: Arthur SKOWRONEK, BESSEREAU LAB, INMG, LYON  
**Contact**: arthur.skowronek13@gmail.com (Review subject: [WORM_ANALYSIS])  
**Python Version**: 3.8.10
**Interface**: Tkinter  
**Hardware Control**: Micro-Manager (pymmcore)  
**Deep Learning**: PyTorch / Ultralytics (YOLO)

---

## 📖 Description

**WormAnalysis** is a desktop application designed for the automated acquisition and analysis of microscopy images, specifically tailored for the detection and phenotyping of *C. elegans* worms.

The application enables:
1.  **Microscope Control**: Real-time control via Micro-Manager core.
2.  **Image Acquisition**: Live mode, Snap, and automated scanning.
3.  **Worm Analysis**: Deep Learning (YOLO) based detection and classification (WT vs. Mutant).
4.  **Data Management**: Automated file organization and usage statistics.
5.  **Model Training**: Integrated interface for re-training or fine-tuning models.

**User guide:** A pdf and a pptx files are available in the `docs/` directory, describing the application, its features, and how to use it.

---

## 🔬 Microscope & Hardware Setup

This project is deployed on two specific microscope setups: **Nikon** and **Macrozoom**.

> [!IMPORTANT]
> **Environment Differences**: The `.env` file configuration differs between the two setups. Code logic (such as objective recommendations in `WormAnalysisApp.py`) adapts based on the `$MICROSCOPE_CONFIG` environment variable.

> [!WARNING]
> **Model Incompatibility**: A Deep Learning model trained on one microscope's computer **cannot** be readily used on the other due to differences in optical properties (magnification, resolution, lighting). Models must be trained using data specific to the hardware.

---

## ⚙️ Technical Architecture

The project follows a component-based architecture isolating the GUI, system logic, and hardware abstraction.

### File Structure
```
WormAnalysis/
├── main.py                    # 🚀 Application Entry Point
├── config.py                  # 🔧 Global Configuration (Paths, Env Vars)
├── src/
│   ├── interface/             # 🖥️ GUI Layer (Tkinter)
│   │   ├── WormAnalysisApp.py # Main Application Window & PC Logic
│   │   ├── colorTheme.py      # Color theme management
│   │   └── Tooltip.py         # Tooltip management
│   └── system/                # 🧠 Business Logic
│       ├── dataset_manager.py # Data loading & preprocessing
│       ├── preprocessing.py   # Image segmentation pipeline
│       └── features.py        # Feature extraction
├── ressources/                # ⚙️ Static Resources
│   └── parameters.yaml        # User-adjustable scanning parameters
├── models/                    # 🤖 Application Models (YOLO .pt, Sklearn .pkl)
├── notebooks/                 # 📓 Jupyter Notebooks for analysis & training
├── logs/                      # 📝 Application logs & Error tracking
├── data/                      # 📂 Generated Data (Images, predictions)
├── training/                  # 🤖 Training data for prediction models
├── docs/                      # 📄 Documentation : user guide, pptx, pdf
└── Micro-Manager-2.0gamma/    # 🔬 Hardware Drivers (Required)
```

---

## 🖥️ Deep Dive: `WormAnalysisApp.py`

This file is the backbone of the GUI and orchestration logic. It is built using `tkinter` and handles the entire user lifecycle.

### Key Responsibilities:
1.  **UI Construction**:
    *   **Sidebar Navigation**: Creates the left-hand menu for switching between "Detection", "Analysis", and "Help" modes.
    *   **Dynamic Parameters Panel**: The right-hand panel changes based on the context (e.g., enabling/disabling objective selection during a scan).
    *   **Content Area**: The central frame allows swapping between different pages (`show_automatic_scan_page`, `show_load_position_page`, etc.) without opening new windows.

2.  **State Management**:
    *   It maintains the state of the application (current page, microscope connection status, loaded parameters).
    *   Handles the switching logic between "Live" camera view and static image review.

3.  **Hardware Interaction**:
    *   Connects the GUI actions (buttons) to the `pymmcore` instance loaded in `main.py`.
    *   Example: Clicking "Snap" triggers `self.CORE.snapImage()` and updates the canvas.

### "Launch Analysis" Logic (Under the Hood)
When triggering an analysis (visible in the "Load Position" workflow or batch processing), the system executes a pipeline similar to what is documented in `notebooks/show_worm_analysis.ipynb`:

1.  **Preprocessing (`src.system.preprocessing`)**:
    *   **YOLO Detection**: The image is passed through a YOLOv8 model to detect bounding boxes of worms.
    *   **Segmentation**: Within each bounding box, traditional computer vision techniques (thresholding, morphology) or secondary networks segment the worm body from the background.
    *   **Filtering**: Noise and non-worm objects are filtered out based on size and shape properties.

2.  **Feature Extraction (`src.system.features`)**:
    *   From the segmented mask, the system calculates morphological features (length, width, area, curvature).

3.  **Result Visualization**:
    *   The `WormAnalysisApp` then overlays these results on the image (bounding boxes, labels) allowing for user verification.

---

## 📓 Jupyter Notebooks Explained

The `notebooks/` directory contains critical tools for development and model maintenance:

*   **`show_worm_analysis.ipynb`**:
    *   **Purpose**: A step-by-step walkthrough of the analysis pipeline.
    *   **Use Case**: Debugging the image processing logic on a single image. It helps visualize how `Dataset_Manager`, `Preprocessing`, and `FeatureExtractor` work together.

*   **`fine_tuning_worm_detection.ipynb`**:
    *   **Purpose**: Script to fine-tune the YOLO object detection model.
    *   **Use Case**: Use this when the microscope setup changes or detection accuracy drops. It retrains the model on new labeled data.

*   **`fine_tuning_worm_segmentation.ipynb`**:
    *   **Purpose**: Dedicated to improving the pixel-level segmentation mask of the worms.

*   **`train_mutant_prediction_model.ipynb`**:
    *   **Purpose**: Trains the classification model (often a Random Forest or similar SVM) that distinguishes "Mutant" from "Wild-Type" based on the extracted features.

---

## 🚀 Installation & Setup

### Prerequisites
*   **Python 3.13+**
*   **Micro-Manager**: The `Micro-Manager-2.0gamma` folder acts as the driver layer and must be present.

### Quick Setup
1.  **Clone the repo**
2.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows: .venv\Scripts\Activate.ps1
    # Mac/Linux: source .venv/bin/activate
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch**:
    ```bash
    python main.py
    ```

### Building the Executable
To bundle the app for distribution (Freeze), use **PyInstaller**.
*   **Windows Build**:
    ```powershell
    pyinstaller --onefile --windowed `
      --name=Worm_detection `
      --icon=icon_desktop.ico `
      --add-data "logs;logs" `
      --add-data "models;models" `
      --add-data "ressources;ressources" `
      --collect-all skan `
      --collect-all python_tsp `
      --collect-all sklearn `
      --collect-all numpy `
      --collect-all scipy `
      main.py
    ```

---

## Computer Migration Guide

- **Copy the folder:** Copy the `WormAnalysis` folder to the new computer.
- **Clean up:** Delete the existing `.venv` folder.
- **Create environment:** In VSCode, create a new virtual environment by running:
    `python -m venv .venv`
- **Activate environment:** Activate it using:
    `.venv/Scripts/activate`
- **Install dependencies:** Reinstall the required packages:
    `pip install -r requirements.txt`
- **Hardware Configuration:** Create a new configuration file for the computer-to-camera connection using Micro-Manager (In the _Hardware Configuration Wizard_, select **'None'** to start from scratch).
- **Move configuration:** Add this newly created configuration file to the following directory:
    `WormAnalysis/Micro-Manager-2.0gamma`
- **Update .env file:** * Set the `CONFIG_FILE` variable to the name of your new configuration file.
    - Ensure the `NAME_CAMERA_CONFIG` variable matches exactly what was entered during the Micro-Manager configuration setup.
- **Calibrate hardware:** The `MICROSCOPE_STEP_SIZE` variable (representing the length of a single step) may need adjustment. You will need to perform slide scanning tests to determine the optimal step value for this specific microscope.


---

## ⚠️ Troubleshooting & FAQ

*   **Error `Load core failed`**:
    *   Check if the microscope is physically powered on.
    *   Ensure the `.env` variable `CONFIG_FILE` points to the correct `.cfg` file for your specific microscope (Nikon vs Macrozoom).

*   **UI Freezing**:
    *   Heavy analysis tasks run on the main thread currently. Be patient during "Batch Analysis".
