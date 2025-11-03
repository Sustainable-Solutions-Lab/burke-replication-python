# Processing Flow Documentation

This document provides a detailed explanation of the data processing flow for the Burke, Hsiang, and Miguel (2015) replication, with special attention to the meaning and interpretation of each variable and processing step.

## Overview

This analysis examines **how temperature affects economic growth** across countries over time. The central question is: Does temperature have a non-linear (curved) relationship with GDP growth? And if so, how does this relationship vary across different types of countries and sectors?

---

## Step 1: Data Preparation and Initial Analysis

**Purpose**: Estimate the historical relationship between temperature and economic growth, then use bootstrap methods to quantify uncertainty in these estimates.

### Input Data

**File**: `GrowthClimateDataset.csv`

This dataset contains annual observations for multiple countries spanning several decades (typically 1960-2010).

**Key Variables**:
- `year`: Calendar year of observation
- `iso_id`: ISO country code (identifies each country uniquely)
- `countryname`: Human-readable country name
- `growthWDI`: **Annual GDP growth rate** (percent change in real GDP per capita from World Development Indicators)
- `UDel_temp_popweight`: **Population-weighted annual temperature** in degrees Celsius for each country
  - This is not a simple average temperature; it weights different regions by population density
  - Higher values mean warmer years; this is the primary climate variable of interest
- `UDel_precip_popweight`: **Population-weighted annual precipitation** in millimeters
  - Controls for rainfall effects on agriculture and economy
- `GDPpctile_WDIppp`: **GDP percentile rank** of the country (0-100)
  - Based on GDP per capita adjusted for purchasing power parity
  - Used to classify countries as "rich" (≥50th percentile) or "poor" (<50th percentile)
- `Pop`: **Total population** of the country
- `TotGDP`: **Total GDP** (not per capita)
- `continent`: Geographic continent
- `AgrGDPgrowthCap`: **Agricultural GDP growth per capita**
- `NonAgrGDPgrowthCap`: **Non-agricultural GDP growth per capita**

### Data Preparation Steps

#### 1.1 Create Time Variables

**Purpose**: Allow for country-specific trends in growth that are unrelated to temperature.

```python
time = year - 1960  # Number of years since 1960 baseline
time2 = time ** 2   # Squared term for quadratic trends
```

**Why reference year 1960?**
- 1960 is near the start of the dataset
- Centering time variables improves numerical stability in regressions
- Allows interpretation of intercepts as values "at baseline period"

**Interpretation**:
- `time`: Linear time trend (constant growth rate over time)
- `time2`: Quadratic time trend (allows for acceleration/deceleration in growth)

#### 1.2 Create Country-Specific Time Trends

**What variable do these trends describe?**
These are time trends in the **dependent variable: annual GDP per capita growth rate** (percent change in real GDP per capita). They control for the fact that each country's per capita growth rate may be trending up or down over time for reasons unrelated to temperature.

**Purpose**: Each country has its own unique trajectory in their **per capita GDP growth rates** over time:
- China: Per capita growth rates **accelerating** over time (1% in 1960 → 10% in 2000)
- Japan: Per capita growth rates **decelerating** over time (8% in 1960 → 1% in 2000)
- USA: Per capita growth rates relatively **stable** over time (~2-3% throughout)

We need to separate these country-specific secular trends in per capita growth rates from temperature effects.

These are **control variables** (regression controls), not the outcome variable itself. They allow each country to have its own baseline time trend in per capita growth rates that is unrelated to temperature.

```python
_yi_[country]: Linear time trend for each country (e.g., constant growth rate increase/decrease)
_y2_[country]: Quadratic time trend for each country (e.g., growth acceleration/deceleration)
```

**What these represent**:
- These are interaction terms: `country_dummy × time` and `country_dummy × time²`
- They control for each country's underlying development trajectory
- For country i: `_yi_i × time` allows country i to have its own linear growth trend
- For country i: `_y2_i × time²` allows country i to have its own growth acceleration/deceleration

**Example**:
- `_yi_USA × time`: Captures that USA had steady, roughly linear GDP growth 1960-2010
- `_y2_CHN × time²`: Captures that China had accelerating GDP growth (quadratic pattern - slow at first, then explosive)

**Why do we need these?**
- Without country-specific trends, we might confuse long-term development patterns with temperature effects
- Example: If a country is industrializing (growing faster over time) AND getting warmer, we need to separate:
  - The warming effect (what we want to estimate)
  - The general industrialization trend (what we want to control for)
- These time trends "soak up" country-specific growth trajectories, leaving only year-to-year growth fluctuations to identify the temperature effect

#### 1.3 Create Temperature and Precipitation Squared Terms

```python
UDel_temp_popweight_2 = UDel_temp_popweight ** 2
UDel_precip_popweight_2 = UDel_precip_popweight ** 2
```

**Purpose**: Capture non-linear (curved) relationships.

**Economic Interpretation**:
- **Linear term** (`UDel_temp_popweight`): The average effect of temperature
- **Squared term** (`UDel_temp_popweight_2`): Allows the effect to change at different temperature levels
  - Example: A hot country getting hotter might suffer more than a cool country getting warmer
  - This creates a parabolic (∩-shaped or ∪-shaped) response curve

**The parabola means**:
- There may be an "optimal temperature" for economic growth
- Countries too far above or below this optimum suffer reduced growth

#### 1.4 Create Interaction Variables

**Rich vs Poor Countries**:
```python
poorWDIppp = 1 if GDPpctile_WDIppp < 50 else 0
```

**Purpose**: Test whether rich and poor countries respond differently to temperature.

**Hypothesis**: Poor countries may be more vulnerable because:
- Greater dependence on agriculture (weather-sensitive)
- Less access to adaptation technologies (air conditioning, irrigation)
- Less financial resilience to weather shocks

**Early vs Late Period**:
```python
early = 1 if year < 1990 else 0
```

**Purpose**: Test whether the temperature-growth relationship has changed over time.

**Hypothesis**: The relationship might change due to:
- Technological progress (air conditioning, better crop varieties)
- Structural economic shifts (less agriculture, more services)
- Climate adaptation investments

#### 1.5 Create Fixed Effects (Dummy Variables)

**Year Fixed Effects** (`year_1961`, `year_1962`, ...):
- Controls for global shocks that affect all countries in a given year
- Examples: Oil price shocks (1973, 1979), global financial crisis (2008)
- Ensures we're comparing countries within the same year, not across different time periods

**Country Fixed Effects** (`iso_USA`, `iso_CHN`, `iso_BRA`, ...):
- Controls for time-invariant country characteristics
- Examples: Geographic location, culture, institutions, natural resource endowments
- Ensures we're looking at variation *within* each country over time, not *between* countries

**Why both?**
- **Without country fixed effects**: We might confuse the fact that hot countries are generally poorer with a causal effect of temperature
- **Without year fixed effects**: We might confuse global trends (like the productivity boom of the 1990s) with temperature effects

---

### 1.1 Baseline Regression Analysis

**Model Specification**:
```
growthWDI = β₁·temperature + β₂·temperature² + β₃·precipitation + β₄·precipitation²
            + country_fixed_effects + year_fixed_effects
            + country_specific_linear_trends + country_specific_quadratic_trends
```

**What this regression answers**:
"After controlling for country-specific development paths, global economic conditions, and rainfall, what is the relationship between temperature and economic growth?"

**Key Outputs**:

1. **Regression Coefficients** (`estimatedCoefficients.csv`):
   - `β₁` (temp): Linear temperature effect
   - `β₂` (temp2): Quadratic temperature effect

2. **Optimal Temperature**:
   - Calculated as: `T_optimal = -β₁ / (2·β₂)`
   - This is the temperature at which growth is maximized
   - **Interpretation**: Countries at this temperature see the highest growth; deviations in either direction reduce growth

3. **Global Response Function** (`estimatedGlobalResponse.csv`):
   - Shows predicted growth impact at every temperature from -5°C to 35°C
   - Includes 90% confidence intervals (uncertainty bounds)
   - **Interpretation**: This is the "damage function" for temperature—how much growth changes at each temperature level

**Statistical Details**:
- **Clustering by country**: Standard errors account for correlation of observations within the same country over time
  - Why? A shock in year t might affect year t+1 for the same country
  - Prevents underestimating uncertainty
- **R-squared**: Proportion of growth variation explained by the model
  - High R² means the model fits historical data well
  - Does NOT prove causation (need to worry about omitted variables)

---

### 1.2 Heterogeneity Analysis

**Purpose**: Test whether the temperature-growth relationship differs across:
1. **Rich vs Poor countries** (by GDP level)
2. **Agricultural vs Non-agricultural sectors**
3. **Early period (pre-1990) vs Late period (post-1990)**

#### Rich vs Poor Analysis

**Model with Interactions**:
```
growthWDI = β₁·temp + β₂·temp² + β₃·(temp × poor) + β₄·(temp² × poor) + ...
```

**Interpretation of Coefficients**:
- For **rich countries** (poor = 0): Effect = β₁·temp + β₂·temp²
- For **poor countries** (poor = 1): Effect = (β₁ + β₃)·temp + (β₂ + β₄)·temp²

**What we're testing**:
- Is β₃ ≠ 0? (Do poor countries have a different linear temperature effect?)
- Is β₄ ≠ 0? (Do poor countries have a different curvature in their response?)
- Combined: Do poor countries have a fundamentally different temperature-growth relationship?

**Economic Interpretation**:
- If poor countries are more sensitive, we expect larger negative coefficients
- This would suggest climate change disproportionately harms development

#### Agricultural vs Non-Agricultural Analysis

**Separate Models**:
1. Agricultural GDP growth ~ temperature
2. Non-agricultural GDP growth ~ temperature

**Purpose**: Test the mechanism—is the temperature effect primarily through agriculture?

**Hypothesis**:
- Agriculture is directly weather-dependent (crops need specific temperatures)
- Services and manufacturing should be less affected (indoor, climate-controlled)
- If true, agricultural growth should show stronger temperature sensitivity

#### Temporal Analysis

**Model**:
```
growthWDI = β₁·temp + β₂·temp² + β₃·(temp × early) + β₄·(temp² × early) + ...
```

**What we're testing**: Has the relationship changed over time?

**Possible Reasons for Change**:
- **Adaptation**: Better technology, infrastructure, early warning systems
- **Structural Change**: Economies shifting from agriculture to services
- **Climate Change**: The climate has already changed, so responses might differ

**Output**: `EffectHeterogeneityOverTime.csv`

---

### 1.3 Bootstrap Analysis

**Purpose**: Quantify uncertainty in our coefficient estimates for use in future projections.

**Why Bootstrap?**
- Standard regression gives us **one** set of coefficients
- But these coefficients are uncertain (sampling variability)
- For projections, we need to propagate this uncertainty forward
- Bootstrap creates 1,000 alternative coefficient sets by resampling

**Method**:
1. **Resample countries with replacement**:
   - Draw 166 countries from the original 166 (some appear multiple times, some not at all)
   - This mimics drawing a new "alternative world" from the same data-generating process

2. **Re-run regression** on the resampled data

3. **Store coefficients** from this run

4. **Repeat 1,000 times**

**Result**: 1,000 sets of coefficients that represent uncertainty in our estimates

**Four Bootstrap Specifications**:

#### 1. Pooled Model (No Lags)
**File**: `bootstrap_noLag.csv`

**Model**: Same as baseline regression

**Use**: Simple projections assuming all countries respond the same to temperature

#### 2. Rich/Poor Model (No Lags)
**File**: `bootstrap_richpoor.csv`

**Model**: Allows rich and poor countries to have different temperature responses

**Use**: Projections that account for heterogeneity by development level

**Stored Coefficients**:
- `temp`, `temppoor`: Linear effects for rich and poor
- `temp2`, `temp2poor`: Quadratic effects for rich and poor

#### 3. Pooled Model (5 Lags)
**File**: `bootstrap_5Lag.csv`

**Model**: Includes current temperature plus 5 years of lagged temperatures
```
growth_t = f(temp_t, temp_{t-1}, temp_{t-2}, ..., temp_{t-5})
```

**Why Lags?**
- Temperature shocks might have **persistent effects**
- Example: A hot year destroys crops → less savings → less investment → lower growth for several years
- Lags capture this dynamic adjustment process

**Stored Coefficients**:
- `temp`, `L1temp`, `L2temp`, ..., `L5temp`: Effects at different lags
- `temp2`, `L1temp2`, ..., `L5temp2`: Squared term effects
- `tlin`: Sum of all linear lag coefficients (total long-run linear effect)
- `tsq`: Sum of all squared lag coefficients (total long-run quadratic effect)

**Interpretation**:
- `temp`: Immediate (contemporaneous) effect of temperature
- `L1temp`: Effect of last year's temperature on this year's growth
- `tlin + tsq`: Total cumulative effect after all adjustment dynamics play out

#### 4. Rich/Poor Model (5 Lags)
**File**: `bootstrap_richpoor_5lag.csv`

**Model**: Combines heterogeneity (rich/poor) with dynamics (lags)

**Why?**
- Rich and poor countries might differ in both immediate and persistent responses
- Example: Rich countries might recover faster from heat shocks (better insurance, credit markets)

**Stored Coefficients**:
- Separate linear and quadratic effects for rich and poor at each lag
- `tlin_rich`, `tsq_rich`: Long-run effects for rich countries
- `tlin_poor`, `tsq_poor`: Long-run effects for poor countries

---

### 1.4 Output Files Summary

| File | Content | Purpose |
|------|---------|---------|
| `estimatedGlobalResponse.csv` | Temperature-growth curve with confidence intervals | Plotting the damage function |
| `estimatedCoefficients.csv` | Baseline regression coefficients | Reference estimates |
| `mainDataset.csv` | Cleaned input data | Use in later steps |
| `EffectHeterogeneity.csv` | Rich/poor and sectoral responses | Heterogeneity analysis (Figure 2) |
| `EffectHeterogeneityOverTime.csv` | Pre/post-1990 responses | Temporal heterogeneity (Figure 2) |
| `bootstrap_noLag.csv` | 1,000 coefficient sets (pooled) | Uncertainty in projections |
| `bootstrap_richpoor.csv` | 1,000 coefficient sets (rich/poor) | Heterogeneous projections |
| `bootstrap_5Lag.csv` | 1,000 coefficient sets with lags (pooled) | Dynamic projections |
| `bootstrap_richpoor_5lag.csv` | 1,000 coefficient sets with lags (rich/poor) | Dynamic heterogeneous projections |

---

## Step 2: Climate Projections

**Purpose**: Calculate how much each country's temperature is expected to change under climate change scenarios.

### Input Data
- **CMIP5 climate model outputs**: Global temperature projections under RCP8.5 (high emissions scenario)
- **Population grids**: Spatial distribution of population within countries
- **Country boundaries**: Shapefiles defining country borders

### Processing

1. **Population-weighting**: Not all temperature changes are equal—changes where people live matter more
   - Example: Warming in Siberia (low population) vs warming in India (high population)

2. **Country aggregation**: Convert grid-cell temperature changes to country-level averages
   - Weight each grid cell by its population
   - Sum across all grid cells within each country

3. **Conversion factors**:
   - Input: Global mean temperature change (e.g., +3°C globally)
   - Output: Country-specific temperature change (e.g., +4°C in USA, +2°C in Brazil)
   - Accounts for spatial heterogeneity in warming patterns

### Output
**File**: `CountryTempChange_RCP85.csv`

**Content**: For each country, the projected temperature change from 1986-2005 baseline to 2080-2100

**Use**: These temperature changes will be applied to the regression coefficients from Step 1 to project GDP impacts

---

## Step 3: Socioeconomic Scenarios

**Purpose**: Project future GDP and population without climate change (the "baseline" scenario).

### Input Data
- **SSP (Shared Socioeconomic Pathways)**: Standard future scenarios used in climate research
  - SSP1: Sustainable development
  - SSP2: Middle of the road
  - SSP3: Fragmented world
  - SSP4: Inequality
  - SSP5: Fossil-fueled development

- **UN population projections**: Expected population growth by country

- **Historical growth rates (1980-2010)**: Baseline economic growth trends

### Processing

1. **Interpolate SSP data**: SSP data is provided in 5-year intervals; interpolate to annual

2. **Create baseline scenario**: Assumes countries maintain their historical growth rates

3. **Project GDP per capita**: Combine population and growth projections
   ```
   GDP_{t+1} = GDP_t × (1 + growth_rate)
   ```

### Output
- `popProjections.Rdata`: Annual population by country, 2010-2099
- `growthProjections.Rdata`: Annual growth rates by country under different SSP scenarios

**Use**: These are the "no climate change" counterfactual projections. Step 4 will add climate impacts.

---

## Step 4: Impact Projections

**Purpose**: Calculate how climate change affects future GDP by combining:
1. Temperature-growth relationship (from Step 1 bootstrap)
2. Future temperature changes (from Step 2)
3. Baseline GDP projections (from Step 3)

### Method

For each bootstrap run (1,000 total) and each year (2010-2099):

1. **Get temperature change** for that year and country
   - Interpolate between current climate and 2080-2100 projection
   - Example: Year 2050 is halfway between 2010 and 2090, so use 50% of total warming

2. **Calculate growth impact** using regression coefficients:
   ```
   impact = β₁·ΔT + β₂·(ΔT)²
   ```
   where ΔT is the temperature change from baseline

3. **Adjust growth rate**:
   ```
   growth_with_CC = baseline_growth + impact
   ```

4. **Compound forward**:
   ```
   GDP_with_CC_{t+1} = GDP_with_CC_t × (1 + growth_with_CC)
   GDP_no_CC_{t+1} = GDP_no_CC_t × (1 + baseline_growth)
   ```

5. **Calculate damages**:
   ```
   damage = (GDP_no_CC - GDP_with_CC) / GDP_no_CC × 100%
   ```

### Four Model Specifications

Each bootstrap coefficient set gets run:
1. **Pooled (no lags)**: Simplest model
2. **Rich/poor (no lags)**: Allows heterogeneity
3. **Pooled (5 lags)**: Includes dynamics
4. **Rich/poor (5 lags)**: Heterogeneity + dynamics (most complex)

### Output Files

For each SSP scenario and model specification:
- `GDPcapCC_[SSP]_[model].Rdata`: GDP per capita with climate change
- `GDPcapNoCC_[SSP]_[model].Rdata`: GDP per capita without climate change
- `GlobalChanges_[SSP]_[model].Rdata`: Global aggregate impacts

**Each file contains**:
- 1,000 projection paths (one per bootstrap run)
- Annual data from 2010-2099
- Country-level detail

**Interpretation**:
- **Median projection**: Most likely outcome (50th percentile across bootstrap runs)
- **Uncertainty bounds**: 5th and 95th percentiles show range of plausible outcomes
- **By year**: Shows how damages accumulate over time
- **By country**: Shows which countries are most affected

---

## Step 5: Damage Function

**Purpose**: Create a simplified relationship between global temperature change and economic damages for use in integrated assessment models (IAMs).

### Input
- Impact projections from Step 4 (GDP with and without climate change)
- IAM temperature scenarios (DICE, FUND, PAGE models)

### Processing

1. **Calculate damages at different warming levels**:
   - For each global mean temperature increase (0.8°C, 1°C, 1.5°C, ..., 6°C)
   - Find the corresponding year in climate projections
   - Calculate GDP loss at that year across all countries

2. **Aggregate globally**:
   - Weight countries by their GDP
   - Sum to get total global damages as % of world GDP

3. **Fit damage function**:
   - Fit a smooth curve through (temperature increase, damage %) points
   - Common form: `Damage = a·T + b·T²` or `Damage = a·T^b`

### Output
`DamageFunction_[model].Rdata`: For each model specification, parameters of the damage function

**Use**: IAM models can use these simplified functions instead of running full country-by-country projections

**Example Interpretation**:
- "Each 1°C of global warming reduces global GDP by 1.2% (with range of 0.5% to 2.3% across uncertainty)"
- This simplification allows cost-benefit analysis of climate policies in IAMs

---

## Step 6: Figure Generation

**Purpose**: Create publication-ready visualizations of results.

### Main Figures

**Figure 2**: Historical temperature-growth relationship
- Panel A: Global response function (∩-shaped curve)
- Panel B: Rich vs poor country responses
- Panel C: Temporal heterogeneity (early vs late period)
- Panel D: Agricultural vs non-agricultural responses

**Figure 3**: Bootstrap uncertainty
- Distribution of estimated optimal temperature
- Confidence intervals on response curves

**Figure 4**: Future projections
- Time series of projected GDP with and without climate change
- Damages by country/region over time
- Comparison across SSP scenarios

**Figure 5**: Damage functions
- Global damages vs temperature increase
- Comparison to other models
- Uncertainty bounds

### Extended Data Figures

Additional robustness checks, sensitivity analyses, and supplementary results

---

## Key Econometric Concepts

### Panel Data
- **Cross-sectional dimension**: Different countries
- **Time-series dimension**: Multiple years per country
- **Advantage**: Can control for unobserved country characteristics and global time trends

### Fixed Effects
- **Country FE**: Controls for all time-invariant differences between countries
  - Example: Geography, culture, institutions
- **Year FE**: Controls for global shocks affecting all countries
  - Example: Oil shocks, global recessions

### Clustering
- Observations within the same country are correlated over time
- Clustering adjusts standard errors to account for this
- Without clustering, we'd underestimate uncertainty (too many "false positives")

### Bootstrap Resampling
- Non-parametric method to estimate uncertainty
- Doesn't assume coefficients follow a normal distribution
- Preserves complex correlation structure in the data

### Identification Strategy
The key assumption for causal interpretation:
- **Year-to-year temperature variation is effectively random** (conditional on country and year fixed effects)
- Countries can't choose their temperature in a given year
- Temperature variation is driven by weather (random), not by economic decisions

**Threats to identification**:
1. Omitted variables that vary within country over time and correlate with temperature
   - Example: Air pollution episodes that are both hot and economically harmful
2. Adaptation: Countries might adapt to temperature, biasing historical estimates
3. Non-stationarity: Climate change might trigger different responses than historical weather variation

---

## Statistical Output Interpretation Guide

### Regression Coefficients

**Temperature (linear term)**:
- Positive → hotter temperatures increase growth (at low baseline temperatures)
- Negative → hotter temperatures reduce growth

**Temperature squared**:
- Negative → inverted U-shape (growth peaks at moderate temperature)
- Positive → U-shape (growth lowest at moderate temperature)

**Typical finding**:
- Linear term positive, squared term negative
- → Optimal temperature around 13°C
- → Countries below 13°C benefit from warming; above 13°C are harmed

### Confidence Intervals

**90% CI**: Range containing the true parameter with 90% probability

**Interpretation**:
- **Narrow CI**: Precise estimate, low uncertainty
- **Wide CI**: Imprecise estimate, high uncertainty
- **CI excludes zero**: Statistically significant at 10% level

### R-squared

**Definition**: Fraction of variation in growth explained by the model

**Example**: R² = 0.35 means 35% of growth variation is explained

**Interpretation**:
- R² depends on fixed effects included
- High R² doesn't prove causation
- Low R² doesn't mean effects aren't important (growth has many drivers)

---

## Interpretation Guidelines for Users

### What This Analysis Shows

1. **Historical relationship**: How temperature has affected growth in the past
2. **Non-linearity**: Effects depend on baseline temperature (hot countries harmed more)
3. **Heterogeneity**: Poor countries are more vulnerable
4. **Future projections**: Plausible range of economic impacts under climate change

### What This Analysis Does NOT Show

1. **Adaptation**: Historical responses may not reflect future adaptation capacity
2. **Catastrophic risks**: Doesn't capture extreme climate tipping points
3. **Non-market damages**: Focuses on GDP, ignoring health, ecosystems, mortality
4. **Migration/conflict**: Second-order effects not directly modeled

### Critical Assumptions

1. **Stationarity**: Future responses similar to historical responses
2. **No structural breaks**: The temperature-growth relationship is stable
3. **Linearity in warming**: Each degree of warming has similar effects
4. **No strategic adaptation**: Countries don't change their sensitivity over time

### Recommended Use

- **For**: Understanding plausible range of economic impacts
- **For**: Comparing across scenarios and model specifications
- **For**: Informing cost-benefit analysis of mitigation
- **Not for**: Precise point predictions decades in advance
- **Not for**: Ignoring uncertainty (always show ranges, not just means)

---

## Data Flow Summary

```
Step 1: Historical Data
    → Estimate temperature-growth relationship
    → Bootstrap for uncertainty
    ↓
Step 2: Climate Models
    → Project future temperature changes
    ↓
Step 3: Socioeconomic Scenarios
    → Project baseline GDP (no climate change)
    ↓
Step 4: Combine Steps 1-3
    → Project GDP with climate change
    → Calculate damages = (GDP_no_CC - GDP_CC)
    ↓
Step 5: Simplify to Damage Function
    → Fit curve: Damage ~ f(Global_Temperature)
    ↓
Step 6: Visualize
    → Create figures and tables
```

Each step builds on previous steps, creating a chain of analysis from historical data to future projections.

---

## Variable Naming Conventions

### Prefixes
- `UDel_`: University of Delaware climate dataset
- `WDI`: World Development Indicators (World Bank)
- `L1`, `L2`, ...: Lags (L1 = 1-year lag, L2 = 2-year lag, etc.)
- `_yi_`: Linear country-specific time trend
- `_y2_`: Quadratic country-specific time trend

### Suffixes
- `_2`: Squared term
- `_popweight`: Population-weighted spatial average
- `poor`: Interaction term for poor countries
- `Cap`: Per capita values

### Common Abbreviations
- `GDP`: Gross Domestic Product
- `GDPcap`: GDP per capita
- `CC`: Climate Change
- `NoCC`: No Climate Change
- `SSP`: Shared Socioeconomic Pathway
- `RCP`: Representative Concentration Pathway
- `temp`: Temperature
- `prec`: Precipitation
- `Agr`: Agricultural
- `NonAgr`: Non-agricultural

---

## References

**Original Paper**:
Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527(7577), 235-239.

**Key Methodological Papers**:
- Dell, Jones, & Olken (2012): "Temperature shocks and economic growth" - foundational panel methods
- Hsiang (2016): "Climate econometrics" - statistical methods for climate impacts
- Burke, Davis, & Diffenbaugh (2018): "Large potential reduction in economic damages under UN mitigation targets" - damage function applications
