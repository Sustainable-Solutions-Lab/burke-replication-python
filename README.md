# Burke, Hsiang, and Miguel 2015 Replication

This project converts the original R and Stata code from Burke, Hsiang, and Miguel (2015) "Global non-linear effect of temperature on economic production" to Python.

The translation from Stata and R to python was done by Ken Caldeira with the help of Cursor and Claude Code. Use at your own risk. This translation could contain errors. If you find any mistake, please let me know (caldeira@stanford.edu).

## Project Overview

The original analysis examines the relationship between temperature and economic growth using historical data and projects future impacts under climate change scenarios. This Python implementation maintains the same processing steps and methodology while providing a modern, reproducible framework.

## Project Structure

```
burke-replication-python/
├── config.py                 # Configuration settings and file paths
├── main.py                   # Main orchestration script
├── step1_data_preparation.py # Data preparation and initial analysis
├── step2_climate_projections.py # Climate projections
├── step3_socioeconomic_scenarios.py # Socioeconomic scenarios
├── step4_impact_projections.py # Impact projections
├── step5_damage_function.py  # Damage function calculations
├── step6_figure_generation.py # Figure generation
├── requirements.txt          # Python dependencies
├── PROCESSING_FLOW.md        # Detailed processing steps documentation
├── CLAUDE.md                 # Development guidelines
├── data/
│   ├── input/               # Input data (tracked in git)
│   │   ├── GrowthClimateDataset.csv
│   │   ├── SSP/
│   │   ├── CCprojections/
│   │   └── IAMdata/
│   └── output/              # Generated outputs (not tracked in git)
│       └── output_{run_name}_{timestamp}/
│           ├── *.csv files
│           ├── *.pdf figures
│           ├── *.log files
│           ├── bootstrap/
│           └── projectionOutput/
└── BurkeHsiangMiguel2015_Replication/  # Original R/Stata code (reference)
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

Place the original Burke, Hsiang, and Miguel 2015 data files in `data/input/`:

```
data/input/
├── GrowthClimateDataset.csv
├── SSP/
│   ├── SSP_PopulationProjections.csv
│   └── SSP_GrowthProjections.csv
├── CCprojections/
│   └── CountryTempChange_RCP85.csv
└── IAMdata/
    └── ProcessedKoppData.csv
```

The original replication code (for reference) should be in:
```
BurkeHsiangMiguel2015_Replication/
├── data/
└── script/
```

## Usage

### Running the Complete Analysis

To run all processing steps with default settings:
```bash
python main.py
```

### Using Run Names

You can specify an optional run name to organize different experiments:
```bash
python main.py [run_name]
```

Examples:
```bash
python main.py              # Uses run_name="default"
python main.py test         # Uses run_name="test"
python main.py experiment1  # Uses run_name="experiment1"
```

The run name affects:
- **Output directory**: `data/output/output_{run_name}_{timestamp}/`
- **Output files**: All CSV, PDF, and log files include `_{run_name}` suffix
  - e.g., `estimatedGlobalResponse_test.csv`, `Figure2_test.pdf`

This allows you to run multiple experiments and keep their outputs separate.

### Skipping Completed Steps

You can modify the skip flags in `config.py` to bypass steps that have already been completed:

```python
# Processing flags
SKIP_STEP_1 = False  # Skip data preparation and initial analysis
SKIP_STEP_2 = False  # Skip climate projections
SKIP_STEP_3 = False  # Skip socioeconomic scenarios
SKIP_STEP_4 = False  # Skip impact projections
SKIP_STEP_5 = False  # Skip damage function
SKIP_STEP_6 = False  # Skip figure generation
```

### Running Individual Steps

You can run individual steps directly:

```bash
python step1_data_preparation.py
python step2_climate_projections.py
python step3_socioeconomic_scenarios.py
python step4_impact_projections.py
python step5_damage_function.py
python step6_figure_generation.py
```

## Processing Steps

### 1. Data Preparation and Initial Analysis
**Original Files:** `GenerateFigure2Data.do`, `GenerateBootstrapData.do`

#### 1.1 Baseline Regression Analysis
- **Input:** `GrowthClimateDataset.csv` (main dataset with temperature, precipitation, GDP growth data)
- **Process:**
  - Run baseline quadratic temperature response regression
  - Estimate global response function with temperature and temperature squared
  - Generate marginal effects and confidence intervals using full variance-covariance matrix (matching Stata's `margins` command)
- **Output:**
  - `estimatedGlobalResponse_{run_name}.csv` (response function data)
  - `estimatedCoefficients_{run_name}.csv` (regression coefficients)
  - `mainDataset_{run_name}.csv` (cleaned dataset)

#### 1.2 Heterogeneity Analysis
- **Process:**
  - Analyze rich vs poor country responses (GDP percentile < 50)
  - Analyze agricultural vs non-agricultural GDP growth
  - Analyze early vs late period responses (pre/post 1990)
- **Output:**
  - `EffectHeterogeneity_{run_name}.csv` (rich/poor, agricultural responses)
  - `EffectHeterogeneityOverTime_{run_name}.csv` (temporal heterogeneity)

#### 1.3 Bootstrap Analysis
- **Process:**
  - Bootstrap regression coefficients (1000 replicates by default, configurable)
  - Sample countries with replacement
  - Run multiple model specifications:
    - Pooled model (no lags)
    - Rich/poor model (no lags)
    - Pooled model (5 lags)
    - Rich/poor model (5 lags)
- **Output:**
  - `bootstrap/bootstrap_noLag_{run_name}.csv`
  - `bootstrap/bootstrap_richpoor_{run_name}.csv`
  - `bootstrap/bootstrap_5Lag_{run_name}.csv`
  - `bootstrap/bootstrap_richpoor_5lag_{run_name}.csv`

### 2. Climate Projections
**Original Files:** `getTemperatureChange.R`

- **Input:** CMIP5 RCP8.5 ensemble mean temperature data, population data, country shapefiles
- **Process:** Calculate population-weighted country-specific temperature changes
- **Output:** `CountryTempChange_RCP85_{run_name}.csv`

### 3. Socioeconomic Scenarios
**Original Files:** `ComputeMainProjections.R` (first part)

- **Input:** SSP data, UN population projections, historical growth rates
- **Process:** Interpolate 5-year SSP projections to annual data, create baseline and SSP scenarios
- **Output:**
  - `projectionOutput/popProjections_{run_name}.Rdata`
  - `projectionOutput/growthProjections_{run_name}.Rdata`

### 4. Impact Projections
**Original Files:** `ComputeMainProjections.R` (main projection section)

- **Input:** Bootstrap coefficients, population/growth projections, temperature changes
- **Process:** Project GDP per capita with and without climate change (2010-2099)
- **Output:**
  - `projectionOutput/GDPcapCC_*_*.Rdata`
  - `projectionOutput/GDPcapNoCC_*_*.Rdata`
  - `projectionOutput/GlobalChanges_*_*.Rdata`

### 5. Damage Function
**Original Files:** `ComputeDamageFunction.R`

- **Input:** Impact projections, IAM temperature scenarios
- **Process:** Calculate damages for different global temperature increases (0.8°C to 6°C)
- **Output:** `projectionOutput/DamageFunction_*.Rdata`

### 6. Figure Generation
**Original Files:** `MakeFigure*.R`, `MakeExtendedDataFigure*.R`

- **Input:** All output data from previous steps
- **Process:** Generate main figures (2-5) and extended data figures
- **Output:**
  - `Figure2_{run_name}.pdf`
  - `Figure3_{run_name}.pdf`
  - `Figure4_{run_name}.pdf`
  - `Figure5_{run_name}.pdf`

## Key Features

### Accurate Replication of Stata's `margins` Command
The confidence interval calculations use the full variance-covariance matrix with sample means for all covariates, matching Stata's `margins, at()` command. This ensures proper uncertainty quantification across the full temperature range.

### Run Name Support
Organize multiple experiments with unique run names that propagate to all output files and directories.

### Skip Logic
The implementation includes intelligent skip logic that:
- Checks if output files already exist
- Respects skip flags in configuration
- Provides warnings if skip flags are set but files are missing

### Modular Design
- Each step is implemented as a separate module
- Clear interfaces between steps
- Easy to test individual components

### Comprehensive Logging
- All runs generate timestamped log files
- Detailed progress tracking
- Error handling with informative messages

## Configuration

The `config.py` file contains all configuration settings:

- **RUN_NAME**: Name for the current run (set via command line)
- **Paths**: File paths for input and output data
- **Flags**: Skip flags for each processing step
- **Settings**: Bootstrap parameters (N_BOOTSTRAP), temperature ranges, etc.
- **Models**: Model specifications and scenarios

Key settings:
```python
N_BOOTSTRAP = 1000  # Number of bootstrap replicates (set to 10 for testing)
RANDOM_SEED = 8675309  # Same as original Stata code
VERBOSITY_LEVEL = 3  # 0=quiet, 1=main steps, 2=detailed, 3=debug
```

## Output Files

All output files are placed in `data/output/output_{run_name}_{timestamp}/` and include the run_name suffix.

### Step 1 Outputs
- `estimatedGlobalResponse_{run_name}.csv`: Global response function
- `estimatedCoefficients_{run_name}.csv`: Regression coefficients
- `mainDataset_{run_name}.csv`: Cleaned main dataset
- `EffectHeterogeneity_{run_name}.csv`: Rich/poor heterogeneity
- `EffectHeterogeneityOverTime_{run_name}.csv`: Temporal heterogeneity
- Bootstrap files in `bootstrap/` subdirectory

### Step 2-5 Outputs
- Various CSV and Rdata files in main directory and `projectionOutput/` subdirectory

### Step 6 Outputs
- `Figure2_{run_name}.pdf` through `Figure5_{run_name}.pdf`

### Log Files
- `burke_replication_{timestamp}_{run_name}.log`: Complete processing log

## Dependencies

- pandas
- numpy
- statsmodels
- scipy
- matplotlib
- seaborn
- tqdm

## License

This project is for research purposes. Please cite the original paper:

Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. Nature, 527(7577), 235-239.

## Contributing

This is a replication project. The goal is to maintain consistency with the original analysis while providing a modern Python implementation. See `CLAUDE.md` for development guidelines.
