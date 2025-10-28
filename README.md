# Pricing Game Solution Template

Welcome to the Pricing Game solution template! This repository provides an interactive workflow for analyzing loss data, developing rating structures, and optimizing commercial premiums for the competition.

## Quick Start

1. **Setup Environment**: Install dependencies (see Setup section below)
2. **Place Input Files**: Add your data files to the `files/` directory
3. **Run Workflow**: Open `workflow.qmd` in Positron or VS Code to run `quarto render`
4. **Export Solution**: Your final rating structure saves to `solution/`

## About the Pricing Game

For complete rules, objectives, and scoring details, see [Pricing-Game-2025.md](Pricing-Game-2025.md).

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** (required)

### 1. Create Virtual Environment

**On Windows (recommended - Python Launcher):**
```cmd
py -m venv .venv
```

**On Windows (alternative):**
```cmd
python -m venv .venv
```
**On macOS/Linux:**
```bash
python3 -m venv .venv
```


### 2. Activate Virtual Environment

**On Windows:**

*Note: If you get an execution policy error, first run this in PowerShell as Administrator:*
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the virtual environment:
```cmd
.venv\Scripts\activate
```

*Alternative for PowerShell:*
```powershell
.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```


### 3. Install Dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4. Place Input Files

Before running the workflow, place your input files in the `files/` directory:

1. **Loss Data**: Place `p{N}_loss_data.csv` in the `files/` directory
2. **Rating Structure**: Place `p{N}_rating_structure.json` in the `files/` directory

Where `{N}` is your player number (1-5).

---

## Repository Structure

```
pricing-game-solution/
│
├── workflow.qmd           # 🎯 Main interactive notebook (start here!)
│
├── files/                 # 📥 Input directory
│   ├── p{N}_loss_data.csv          # Your loss data (place here)
│   ├── p{N}_rating_structure.json  # Initial rating structure (place here)
│   └── retention.json              # Retention model parameters
│
├── solution/              # 📤 Output directory
│   └── player_{N}.json    # Your final solution (generated)
│
├── outputs/               # 📊 Intermediate outputs
│   ├── bands_freq.json    # Frequency model bands
│   ├── bands_sev.json     # Severity model bands
│   └── bands_consolidated.json  # Combined bands
│
├── pg_utils/              # 🔧 Helper functions
│   ├── __init__.py
│   ├── methods.py         # Scoring and validation utilities
│   ├── charts.py          # Chart generation functions
│   └── glm_utils.py       # GLM modeling utilities
│
├── README.md              # This file
├── Pricing-Game-2025.md   # Game rules and objectives
├── requirements.txt       # Python dependencies
├── _quarto.yml            # Quarto configuration
└── LICENSE                # License file
```

---

## The Workflow Notebook

The `workflow.qmd` file is your main workspace. It guides you through:

### 1. **Data Loading**

- Reads `p{N}_loss_data.csv` containing historical claims and exposures
- Loads `p{N}_rating_structure.json` with rating factors and base rates
- Loads `retention.json` with retention model parameters (logistic regression model with age adjustments)
- Validates data integrity and displays key statistics

### 2. **Exploratory Analysis**

- Univariate analysis: Distribution of key rating variables
- Loss patterns: Pure premium by segment
- Exposure analysis: Portfolio composition
- Interactive Plotly visualizations

### 3. **Rating Structure Development**

- Score commercial premiums using your rating structure
- Calculate loss ratios by segment
- Identify profitable and unprofitable segments
- Test different factor combinations

### 4. **Multivariate Analysis**

- Variable selection tools
- Band optimization utilities
- Cross-validation of rating factors
- Interaction effects analysis

### 5. **Retention Modeling**

- Re-score portfolio with new rates
- Calculate premium changes (delta%)
- Project retention impacts
- Optimize for profitability vs. retention

### 6. **Solution Export**

- Validate your final rating structure
- Ensure JSON schema compliance
- Export to `solution/player_{N}.json`
- Generate profitability summary

---

Good luck with your pricing strategy!
