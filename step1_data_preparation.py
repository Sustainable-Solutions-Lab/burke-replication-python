"""
Step 1: Data Preparation and Initial Analysis

PURPOSE:
This module estimates the historical relationship between temperature and economic growth using
panel regression methods. It answers the question: "How does temperature affect GDP growth?"
while controlling for country-specific development paths, global economic shocks, and precipitation.

The analysis produces:
1. A baseline estimate of the temperature-growth relationship (non-linear, parabolic curve)
2. Tests for heterogeneity (do rich and poor countries respond differently?)
3. Bootstrap uncertainty estimates (1,000 alternative coefficient sets for projections)

ECONOMIC QUESTION:
Is there an "optimal temperature" for economic productivity? Do hotter or colder deviations from
this optimum harm growth? How does this relationship vary across countries and over time?

STATISTICAL APPROACH:
- Panel regression with country and year fixed effects
- Non-linear specification (quadratic in temperature)
- Country-specific time trends to control for development paths
- Clustered standard errors to account for within-country correlation
- Bootstrap resampling for uncertainty quantification

ORIGINAL STATA FILES:
- GenerateFigure2Data.do: Main regression analysis and heterogeneity analysis
- GenerateBootstrapData.do: Bootstrap analysis with various specifications

See PROCESSING_FLOW.md for detailed documentation of the processing steps and variable meanings.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
import logging
from tqdm import tqdm
import os
from config import INPUT_FILES, OUTPUT_FILES, N_BOOTSTRAP

# Set up logging
from config import setup_logging
logger = setup_logging()

# Constants
RANDOM_SEED = 8675309  # Same as Stata

class BurkeDataPreparation:
    """Replicate Burke, Hsiang, and Miguel (2015) data preparation and analysis."""
    
    def __init__(self):
        self.data = None
        self.results = {}
    
    def load_data(self):
        """Load the main dataset."""
        logger.info("Loading data...")
        self.data = pd.read_csv(INPUT_FILES['main_dataset'], encoding='latin-1')
        logger.info(f"Data loaded: {self.data.shape}")
        return self.data
    
    def prepare_data(self):
        """
        Prepare data for regression analysis by creating derived variables.

        This function creates:
        1. Time trend variables (for country-specific growth trajectories)
        2. Non-linear terms (squared temperature and precipitation)
        3. Interaction indicators (rich/poor, early/late periods)
        4. Fixed effects (country and year dummy variables)
        """
        logger.info("Preparing data...")

        # Create time variables centered at 1960
        # WHY: Centering improves numerical stability and allows interpretation of intercepts
        # time = 0 represents year 1960 (near start of dataset)
        # time = 50 represents year 2010
        logger.info("Creating time variables with reference year 1960...")
        self.data['time'] = self.data['year'] - 1960  # Linear time trend
        self.data['time2'] = self.data['time'] ** 2   # Quadratic time trend (allows acceleration/deceleration)

        # Create temperature squared term to capture non-linear (parabolic) relationship
        # WHY: Economic theory suggests productivity peaks at moderate temperatures
        # Too cold OR too hot reduces productivity → inverted U-shape
        self.data['UDel_temp_popweight_2'] = self.data['UDel_temp_popweight'] ** 2

        # Create "poor country" indicator: 1 if GDP per capita is below median, 0 otherwise
        # WHY: Test whether poorer countries are more vulnerable to temperature shocks
        # Hypothesis: Poor countries have less adaptive capacity (less AC, irrigation, etc.)
        self.data['poorWDIppp'] = (self.data['GDPpctile_WDIppp'] < 50).astype(int)
        # Preserve missing values (don't impute poor/rich status if GDP data is missing)
        self.data.loc[self.data['GDPpctile_WDIppp'].isna(), 'poorWDIppp'] = np.nan

        # Create "early period" indicator: 1 if before 1990, 0 if 1990 or later
        # WHY: Test whether temperature-growth relationship has changed over time
        # Possible reasons: technological change, structural economic shifts, adaptation
        self.data['early'] = (self.data['year'] < 1990).astype(int)
        
        # Create COUNTRY FIXED EFFECTS (dummy variables for each country)
        # WHY: Controls for all time-invariant country characteristics
        # Examples: geography, culture, institutions, natural resources, historical legacies
        # Without these: we might confuse "hot countries are poor" with "heat causes poverty"
        # With these: we only use within-country variation (year-to-year temperature fluctuations)
        logger.info("Creating country dummy variables...")

        country_codes = sorted(self.data['iso_id'].unique())
        logger.info(f"Found {len(country_codes)} unique countries")

        # Create dummy variables (1 if observation is from that country, 0 otherwise)
        country_dummies = pd.get_dummies(self.data['iso_id'], prefix='iso', dtype=int)

        # Drop one country as reference to avoid perfect collinearity
        # (All dummies summing to 1 would make the design matrix singular)
        first_country = country_codes[0]
        reference_col = f'iso_{first_country}'
        country_dummies = country_dummies.drop(columns=[reference_col])

        logger.info(f"Dropped '{reference_col}' as reference category")
        logger.info(f"Created {len(country_dummies.columns)} country dummy variables")

        self.data = pd.concat([self.data, country_dummies], axis=1)

        # Create YEAR FIXED EFFECTS (dummy variables for each year)
        # WHY: Controls for global shocks that affect all countries in the same year
        # Examples: oil price spikes (1973, 1979), global financial crisis (2008), tech booms
        # Without these: we might confuse global recessions with temperature effects
        # With these: we compare countries to each other within the same year
        logger.info("Creating year dummy variables...")
        year_codes = sorted(self.data['year'].unique())
        year_dummies = pd.get_dummies(self.data['year'], prefix='year', dtype=int)

        # Drop one year as reference (same collinearity reason as country FE)
        reference_year = year_codes[0]
        reference_col = f'year_{reference_year}'
        if reference_col in year_dummies.columns:
            year_dummies = year_dummies.drop(columns=[reference_col])
            logger.info(f"Dropped '{reference_col}' as reference year for dummies")
        else:
            logger.warning(f"Reference year column '{reference_col}' not found in year dummies")
        logger.info(f"Created {len(year_dummies.columns)} year dummy variables")
        self.data = pd.concat([self.data, year_dummies], axis=1)
        
        logger.info("Data preparation completed")
        return self.data
    
    def create_time_trends(self):
        """
        Create COUNTRY-SPECIFIC TIME TRENDS for regression analysis.

        WHY WE NEED THESE:
        Country fixed effects control for the average level of each country, but countries
        also have different growth TRAJECTORIES over time:
        - China: rapid acceleration (exponential growth path)
        - Japan: slowing growth (initially fast, then stagnant)
        - USA: steady growth (relatively linear)

        Without country-specific trends, we might misattribute these different development
        paths to temperature effects. For example:
        - If China is warming AND growing fast, is warming causing growth?
        - No! China is industrializing. We need to separate this trend from temperature effects.

        WHAT THIS CREATES:
        - _yi_[country]: Linear time trend specific to each country (e.g., steady growth)
        - _y2_[country]: Quadratic time trend specific to each country (e.g., acceleration)

        Each country gets its own slope and curvature in its growth path.

        Original Stata code:
        qui xi i.iso_id*time, pref(_yi_)  //linear country time trends
        qui xi i.iso_id*time2, pref(_y2_) //quadratic country time trend
        """
        logger.info("Creating time trends...")

        # Create time variables centered at 1960
        self.data['time'] = self.data['year'] - 1960
        self.data['time2'] = self.data['time'] ** 2
        
        # Create interaction terms: country_dummy × time and country_dummy × time²
        # For each country, these equal the time/time² values for that country, 0 for others
        # This allows each country to have its own linear and quadratic growth trajectory
        # Example: _yi_CHN will be (year-1960) for China observations, 0 for all other countries
        countries = self.data['iso_id'].unique()
        yi_cols = {}
        y2_cols = {}

        for country in countries:
            mask = self.data['iso_id'] == country
            # Linear trend: equals time for this country, 0 for others
            yi_cols[f'_yi_{country}'] = np.where(mask, self.data['time'], 0)
            # Quadratic trend: equals time² for this country, 0 for others
            y2_cols[f'_y2_{country}'] = np.where(mask, self.data['time2'], 0)
        
        # Add all columns at once to avoid fragmentation
        yi_df = pd.DataFrame(yi_cols, index=self.data.index)
        y2_df = pd.DataFrame(y2_cols, index=self.data.index)
        self.data = pd.concat([self.data, yi_df, y2_df], axis=1)
        
        # Drop base trends (like Stata: qui drop _yi_iso_id*; qui drop _y2_iso_id*)
        base_trends = [col for col in self.data.columns if '_yi_iso_id' in col or '_y2_iso_id' in col]
        if base_trends:
            self.data = self.data.drop(columns=base_trends)
        
        logger.info(f"Time trends created. Added {len(yi_cols)} linear and {len(y2_cols)} quadratic trend columns.")
    
    def run_regression(self, regression_type, data=None, **kwargs):
        """
        Unified regression function for all regression types.
        
        Args:
            regression_type (str): One of ['baseline', 'heterogeneity', 'temporal', 
                                  'bootstrap_pooled_no_lag', 'bootstrap_rich_poor_no_lag',
                                  'bootstrap_pooled_5_lag', 'bootstrap_rich_poor_5_lag']
            data (pd.DataFrame): Data to use (defaults to self.data)
            **kwargs: Additional parameters specific to regression type
                - dependent_var (str): Dependent variable name (default: 'growthWDI')
                - interaction_var (str): Variable for interactions (e.g., 'poorWDIppp', 'early')
                - use_lags (bool): Whether to use lagged variables (default: False)
                - create_time_trends (bool): Whether to create time trends (default: True)
        
        Returns:
            dict: Standardized results dictionary with keys:
                - 'results': statsmodels regression results object
                - 'params': dict of parameter estimates
                - 'rsquared': float
                - 'n_obs': int
                - 'regression_type': str
                - Additional keys specific to regression type
        """
        # Use provided data or default to self.data
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Extract kwargs
        dependent_var = kwargs.get('dependent_var', 'growthWDI')
        interaction_var = kwargs.get('interaction_var', None)
        use_lags = kwargs.get('use_lags', False)
        create_time_trends = kwargs.get('create_time_trends', True)
        
        #logger.info(f"Running {regression_type} regression...")
        
        # Create time trends if needed
        if create_time_trends:
            # Create time variables with 1960 reference
            data['time'] = data['year'] - 1960
            data['time2'] = data['time'] ** 2
            
            # Create time trends (optimized to avoid DataFrame fragmentation)
            countries = data['iso_id'].unique()
            yi_cols = {}
            y2_cols = {}
            for country in countries:
                mask = data['iso_id'] == country
                yi_cols[f'_yi_{country}'] = np.where(mask, data['time'], 0)
                y2_cols[f'_y2_{country}'] = np.where(mask, data['time2'], 0)
            
            # Add all columns at once to avoid fragmentation
            yi_df = pd.DataFrame(yi_cols, index=data.index)
            y2_df = pd.DataFrame(y2_cols, index=data.index)
            data = pd.concat([data, yi_df, y2_df], axis=1)
            
            # Drop base trends
            base_trends = [col for col in data.columns if '_yi_iso_id' in col or '_y2_iso_id' in col]
            if base_trends:
                data = data.drop(columns=base_trends)
        
        # Create lagged variables if needed
        if use_lags:
            data = self._create_lagged_variables(data)
        
        # Prepare dependent variable
        y = data[dependent_var]
        
        # Get fixed effects
        year_cols = [col for col in data.columns if col.startswith('year_')]
        iso_cols = [col for col in data.columns if col.startswith('iso_') and col != 'iso_id']
        trend_cols = [col for col in data.columns if col.startswith('_yi_') or col.startswith('_y2_')]
        
        # Prepare regression columns based on regression type
        regression_cols = []
        
        if regression_type in ['baseline', 'bootstrap_pooled_no_lag']:
            # Basic temperature and precipitation variables
            regression_cols = ['UDel_temp_popweight', 'UDel_temp_popweight_2', 
                              'UDel_precip_popweight', 'UDel_precip_popweight_2']
        
        elif regression_type in ['heterogeneity', 'bootstrap_rich_poor_no_lag']:
            # Basic variables plus interaction terms
            regression_cols = ['UDel_temp_popweight', 'UDel_temp_popweight_2', 
                              'UDel_precip_popweight', 'UDel_precip_popweight_2']
            
            # Create interaction terms
            if interaction_var and interaction_var in data.columns:
                interaction_data = data[interaction_var]
                data['temp_poor'] = data['UDel_temp_popweight'] * interaction_data
                data['temp2_poor'] = data['UDel_temp_popweight_2'] * interaction_data
                data['precip_poor'] = data['UDel_precip_popweight'] * interaction_data
                data['precip2_poor'] = data['UDel_precip_popweight_2'] * interaction_data
                
                regression_cols.extend(['temp_poor', 'temp2_poor', 'precip_poor', 'precip2_poor'])
        
        elif regression_type in ['temporal']:
            # Basic variables plus temporal interaction terms
            regression_cols = ['UDel_temp_popweight', 'UDel_temp_popweight_2', 
                              'UDel_precip_popweight', 'UDel_precip_popweight_2']
            
            # Create temporal interaction terms
            if interaction_var and interaction_var in data.columns:
                interaction_data = data[interaction_var]
                data['temp_early'] = data['UDel_temp_popweight'] * interaction_data
                data['temp2_early'] = data['UDel_temp_popweight_2'] * interaction_data
                data['precip_early'] = data['UDel_precip_popweight'] * interaction_data
                data['precip2_early'] = data['UDel_precip_popweight_2'] * interaction_data
                
                regression_cols.extend(['temp_early', 'temp2_early', 'precip_early', 'precip2_early'])
        
        elif regression_type in ['bootstrap_pooled_5_lag']:
            # Current and lagged variables
            regression_cols = []
            # Current and lagged temperature
            regression_cols.extend(['UDel_temp_popweight', 'L1temp', 'L2temp', 'L3temp', 'L4temp', 'L5temp'])
            # Current and lagged temperature squared
            regression_cols.extend(['UDel_temp_popweight_2', 'L1temp2', 'L2temp2', 'L3temp2', 'L4temp2', 'L5temp2'])
            # Current and lagged precipitation
            regression_cols.extend(['UDel_precip_popweight', 'L1prec', 'L2prec', 'L3prec', 'L4prec', 'L5prec'])
            # Current and lagged precipitation squared
            regression_cols.extend(['UDel_precip_popweight_2', 'L1prec2', 'L2prec2', 'L3prec2', 'L4prec2', 'L5prec2'])
        
        elif regression_type in ['bootstrap_rich_poor_5_lag']:
            # Current and lagged variables with interaction terms
            regression_cols = []
            # Current and lagged temperature
            regression_cols.extend(['UDel_temp_popweight', 'L1temp', 'L2temp', 'L3temp', 'L4temp', 'L5temp'])
            # Current and lagged temperature squared
            regression_cols.extend(['UDel_temp_popweight_2', 'L1temp2', 'L2temp2', 'L3temp2', 'L4temp2', 'L5temp2'])
            # Current and lagged precipitation
            regression_cols.extend(['UDel_precip_popweight', 'L1prec', 'L2prec', 'L3prec', 'L4prec', 'L5prec'])
            # Current and lagged precipitation squared
            regression_cols.extend(['UDel_precip_popweight_2', 'L1prec2', 'L2prec2', 'L3prec2', 'L4prec2', 'L5prec2'])
            
            # Create interaction terms for all lagged variables
            if interaction_var and interaction_var in data.columns:
                interaction_data = data[interaction_var]
                # Interaction terms for current variables
                data['temp_poor'] = data['UDel_temp_popweight'] * interaction_data
                data['temp2_poor'] = data['UDel_temp_popweight_2'] * interaction_data
                data['precip_poor'] = data['UDel_precip_popweight'] * interaction_data
                data['precip2_poor'] = data['UDel_precip_popweight_2'] * interaction_data
                
                # Interaction terms for lagged variables
                for lag in range(1, 6):
                    data[f'L{lag}temp_poor'] = data[f'L{lag}temp'] * interaction_data
                    data[f'L{lag}temp2_poor'] = data[f'L{lag}temp2'] * interaction_data
                    data[f'L{lag}prec_poor'] = data[f'L{lag}prec'] * interaction_data
                    data[f'L{lag}prec2_poor'] = data[f'L{lag}prec2'] * interaction_data
                
                # Add interaction terms to regression columns
                regression_cols.extend(['temp_poor', 'temp2_poor', 'precip_poor', 'precip2_poor'])
                for lag in range(1, 6):
                    regression_cols.extend([f'L{lag}temp_poor', f'L{lag}temp2_poor', 
                                         f'L{lag}prec_poor', f'L{lag}prec2_poor'])
        
        # Add fixed effects
        regression_cols.extend(year_cols)
        regression_cols.extend(trend_cols)
        regression_cols.extend(iso_cols)
        
        # Create X matrix
        X = data[regression_cols]
        X = sm.add_constant(X)
        
        # Remove missing values
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        y_clean = y[valid_mask]
        X_clean = X[valid_mask]
        
        # Convert boolean columns to integers
        bool_cols = X_clean.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X_clean.loc[:, col] = X_clean[col].astype(int)
        
        # DIAGNOSTIC: Check data types before regression (for baseline regression)
        if regression_type == 'baseline':
            logger.info("=== DIAGNOSTIC: Checking data types before regression ===")
            logger.info(f"X_clean shape: {X_clean.shape}")
            logger.info(f"X_clean dtypes:\n{X_clean.dtypes}")
            
            # Check for object dtype columns
            object_cols = X_clean.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                logger.error(f"Found object dtype columns: {list(object_cols)}")
                for col in object_cols:
                    logger.error(f"Column '{col}' unique values: {X_clean[col].unique()[:10]}")
            
            # Check for any non-numeric data
            for col in X_clean.columns:
                try:
                    pd.to_numeric(X_clean[col], errors='raise')
                except (ValueError, TypeError) as e:
                    logger.error(f"Column '{col}' contains non-numeric data: {e}")
                    logger.error(f"Sample values: {X_clean[col].head()}")
            
            # Convert any remaining object columns to numeric if possible
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    try:
                        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                        logger.info(f"Converted column '{col}' from object to numeric")
                    except Exception as e:
                        logger.error(f"Could not convert column '{col}' to numeric: {e}")
            
            # Final check
            logger.info(f"Final X_clean dtypes:\n{X_clean.dtypes}")
            logger.info("=== END DIAGNOSTIC ===")
        
        # Run regression with clustering
        model = OLS(y_clean, X_clean)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': data.loc[valid_mask, 'iso_id']})
        
        # Create standardized results dictionary
        result_dict = {
            'results': results,
            'rsquared': results.rsquared,
            'n_obs': len(y_clean),
            'regression_type': regression_type,
            'params': results.params.to_dict()
        }
        
        # Add regression-specific parameters
        if regression_type in ['bootstrap_pooled_no_lag', 'bootstrap_rich_poor_no_lag']:
            coefs = results.params
            if regression_type == 'bootstrap_pooled_no_lag':
                result_dict.update({
                    'temp': coefs['UDel_temp_popweight'],
                    'temp2': coefs['UDel_temp_popweight_2'],
                    'prec': coefs['UDel_precip_popweight'],
                    'prec2': coefs['UDel_precip_popweight_2']
                })
            else:  # bootstrap_rich_poor_no_lag
                result_dict.update({
                    'temp': coefs['UDel_temp_popweight'],
                    'temppoor': coefs['temp_poor'],
                    'temp2': coefs['UDel_temp_popweight_2'],
                    'temp2poor': coefs['temp2_poor'],
                    'prec': coefs['UDel_precip_popweight'],
                    'precpoor': coefs['precip_poor'],
                    'prec2': coefs['UDel_precip_popweight_2'],
                    'prec2poor': coefs['precip2_poor']
                })
        
        elif regression_type in ['bootstrap_pooled_5_lag']:
            coefs = results.params
            # Calculate tlin and tsq (sums of lagged coefficients)
            tlin = (coefs['UDel_temp_popweight'] + coefs['L1temp'] + coefs['L2temp'] + 
                    coefs['L3temp'] + coefs['L4temp'] + coefs['L5temp'])
            tsq = (coefs['UDel_temp_popweight_2'] + coefs['L1temp2'] + coefs['L2temp2'] + 
                   coefs['L3temp2'] + coefs['L4temp2'] + coefs['L5temp2'])
            
            result_dict.update({
                'temp': coefs['UDel_temp_popweight'],
                'L1temp': coefs['L1temp'],
                'L2temp': coefs['L2temp'],
                'L3temp': coefs['L3temp'],
                'L4temp': coefs['L4temp'],
                'L5temp': coefs['L5temp'],
                'temp2': coefs['UDel_temp_popweight_2'],
                'L1temp2': coefs['L1temp2'],
                'L2temp2': coefs['L2temp2'],
                'L3temp2': coefs['L3temp2'],
                'L4temp2': coefs['L4temp2'],
                'L5temp2': coefs['L5temp2'],
                'tlin': tlin,
                'tsq': tsq
            })
        
        elif regression_type in ['bootstrap_rich_poor_5_lag']:
            coefs = results.params
            # Calculate tlin and tsq for rich and poor separately
            # Rich (no interaction)
            tlin_rich = (coefs['UDel_temp_popweight'] + coefs['L1temp'] + coefs['L2temp'] + 
                         coefs['L3temp'] + coefs['L4temp'] + coefs['L5temp'])
            tsq_rich = (coefs['UDel_temp_popweight_2'] + coefs['L1temp2'] + coefs['L2temp2'] + 
                        coefs['L3temp2'] + coefs['L4temp2'] + coefs['L5temp2'])
            
            # Poor (with interaction)
            tlin_poor = tlin_rich + (coefs['temp_poor'] + coefs['L1temp_poor'] + coefs['L2temp_poor'] + 
                                    coefs['L3temp_poor'] + coefs['L4temp_poor'] + coefs['L5temp_poor'])
            tsq_poor = tsq_rich + (coefs['temp2_poor'] + coefs['L1temp2_poor'] + coefs['L2temp2_poor'] + 
                                  coefs['L3temp2_poor'] + coefs['L4temp2_poor'] + coefs['L5temp2_poor'])
            
            result_dict.update({
                'temp': coefs['UDel_temp_popweight'],
                'L1temp': coefs['L1temp'],
                'L2temp': coefs['L2temp'],
                'L3temp': coefs['L3temp'],
                'L4temp': coefs['L4temp'],
                'L5temp': coefs['L5temp'],
                'temp2': coefs['UDel_temp_popweight_2'],
                'L1temp2': coefs['L1temp2'],
                'L2temp2': coefs['L2temp2'],
                'L3temp2': coefs['L3temp2'],
                'L4temp2': coefs['L4temp2'],
                'L5temp2': coefs['L5temp2'],
                'temppoor': coefs['temp_poor'],
                'L1temppoor': coefs['L1temp_poor'],
                'L2temppoor': coefs['L2temp_poor'],
                'L3temppoor': coefs['L3temp_poor'],
                'L4temppoor': coefs['L4temp_poor'],
                'L5temppoor': coefs['L5temp_poor'],
                'temp2poor': coefs['temp2_poor'],
                'L1temp2poor': coefs['L1temp2_poor'],
                'L2temp2poor': coefs['L2temp2_poor'],
                'L3temp2poor': coefs['L3temp2_poor'],
                'L4temp2poor': coefs['L4temp2_poor'],
                'L5temp2poor': coefs['L5temp2_poor'],
                'tlin_rich': tlin_rich,
                'tsq_rich': tsq_rich,
                'tlin_poor': tlin_poor,
                'tsq_poor': tsq_poor
            })
        
        # Log R-squared to file only, not console
        from config import log_file_only
        log_file_only(f"{regression_type} regression completed. R-squared: {results.rsquared:.4f}")
        return result_dict
    
    def baseline_regression(self):
        """
        Run baseline regression to estimate the global temperature-growth relationship.

        REGRESSION MODEL:
        GDP_growth = β₁·temperature + β₂·temperature² + β₃·precipitation + β₄·precipitation²
                     + country_fixed_effects + year_fixed_effects
                     + country_specific_time_trends + error

        WHAT WE'RE ESTIMATING:
        - β₁ (linear temperature effect): How growth changes with each 1°C increase
        - β₂ (quadratic temperature effect): Whether this effect changes at different temperatures
        - Combined: These create a parabolic relationship (∩-shaped curve)

        INTERPRETATION:
        - If β₁ > 0 and β₂ < 0: Growth increases with temperature up to an optimum, then decreases
        - Optimal temperature = -β₁ / (2·β₂)
        - This optimal temperature maximizes economic growth

        CONTROLS:
        - Precipitation (linear and squared): Controls for rainfall effects
        - Country fixed effects: Controls for time-invariant country differences
        - Year fixed effects: Controls for global shocks
        - Country time trends: Controls for country-specific development paths

        STANDARD ERRORS:
        - Clustered by country: Accounts for within-country correlation over time
        - This prevents underestimating uncertainty

        Original Stata code from GenerateFigure2Data.do:
        reg growthWDI c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id, cluster(iso_id)
        """
        logger.info("Running baseline regression...")
        
        # Use unified regression function
        result_dict = self.run_regression('baseline')
        
        # Store results for compatibility
        self.results['baseline'] = result_dict['results']
        
        # Log R-squared to file only, not console
        from config import log_file_only
        log_file_only(f"Baseline regression completed. R-squared: {result_dict['rsquared']:.4f}")
        return result_dict['results']
    
    def generate_global_response(self, results):
        """
        Generate the global temperature-growth response curve ("damage function").

        PURPOSE:
        Create a curve showing how GDP growth varies with temperature across the full range
        of observed temperatures (-5°C to 35°C annual average).

        WHAT THIS FUNCTION DOES:
        1. Calculate optimal temperature (where growth is maximized)
        2. For each temperature from -5°C to 35°C:
           - Predict growth rate using full model with other covariates at sample means
           - Calculate 90% confidence interval using FULL variance-covariance matrix
        3. Normalize by subtracting maximum (like R code does)
        4. Save the curve for plotting (this becomes Figure 2, Panel A)

        MATCHING STATA'S MARGINS COMMAND:
        Stata's `margins, at(temp=(-5(1)35))` computes predictions by:
        1. Setting temp to specified values
        2. Setting ALL OTHER covariates to their sample means
        3. Using the FULL VCE matrix for standard error calculation

        This implementation replicates that approach by:
        - Constructing full gradient vectors with sample means for non-temperature variables
        - Using the complete variance-covariance matrix from the regression

        Original Stata code:
        margins, at(temp=(-5(1)35)) post noestimcheck level(90)
        parmest, norestore level(90)
        """
        logger.info("Generating global response function...")

        # Get coefficients
        temp_coef = results.params['UDel_temp_popweight']
        temp2_coef = results.params['UDel_temp_popweight_2']

        # Calculate optimal temperature (where growth is maximized)
        optimal_temp = -temp_coef / (2 * temp2_coef)

        # Generate temperature range for response function (like Stata: margins, at(temp=(-5(1)35)))
        temp_range = np.arange(-5, 36, 1)

        # Get all parameter names from the regression results
        param_names = list(results.params.index)
        n_params = len(param_names)

        # Compute sample means for constructing the gradient vector
        # For each covariate, we need its mean value in the regression sample
        # Note: For FE dummies, the mean is the proportion of observations with that FE=1
        sample_means = {}
        for name in param_names:
            if name == 'const':
                sample_means[name] = 1.0  # Constant always = 1
            elif name == 'UDel_temp_popweight':
                sample_means[name] = None  # Will be set by at() values
            elif name == 'UDel_temp_popweight_2':
                sample_means[name] = None  # Will be set by at() values
            elif name in self.data.columns:
                # For variables in the data, use sample mean
                sample_means[name] = self.data[name].mean()
            else:
                # For derived variables (time trends, etc.), compute from data
                if name.startswith('_yi_'):
                    country = name[4:]
                    if country in self.data['iso_id'].values:
                        mask = self.data['iso_id'] == country
                        sample_means[name] = (self.data.loc[mask, 'year'] - 1960).mean() * mask.mean()
                    else:
                        sample_means[name] = 0.0
                elif name.startswith('_y2_'):
                    country = name[4:]
                    if country in self.data['iso_id'].values:
                        mask = self.data['iso_id'] == country
                        sample_means[name] = ((self.data.loc[mask, 'year'] - 1960)**2).mean() * mask.mean()
                    else:
                        sample_means[name] = 0.0
                elif name.startswith('iso_'):
                    # Country FE dummy - mean is proportion of obs from that country
                    country = name[4:]
                    sample_means[name] = (self.data['iso_id'] == int(country)).mean() if country.isdigit() else 0.0
                elif name.startswith('year_'):
                    # Year FE dummy - mean is proportion of obs from that year
                    year = name[5:]
                    sample_means[name] = (self.data['year'] == int(year)).mean() if year.isdigit() else 0.0
                else:
                    sample_means[name] = 0.0

        # Get full variance-covariance matrix
        full_cov = results.cov_params().values

        # Step 1: Calculate predicted growth rates and standard errors
        predictions_raw = []
        se_predictions = []

        for temp in temp_range:
            # Construct full gradient vector (like Stata's margins)
            # For each parameter: gradient = value of corresponding covariate at this temp
            grad = np.zeros(n_params)
            for i, name in enumerate(param_names):
                if name == 'const':
                    grad[i] = 1.0
                elif name == 'UDel_temp_popweight':
                    grad[i] = temp
                elif name == 'UDel_temp_popweight_2':
                    grad[i] = temp**2
                else:
                    # Use sample mean for all other covariates
                    grad[i] = sample_means.get(name, 0.0)

            # Compute prediction: y = gradient' * beta
            pred = grad @ results.params.values
            predictions_raw.append(pred)

            # Compute variance: Var(y) = gradient' * VCE * gradient
            pred_var = grad.T @ full_cov @ grad
            se_predictions.append(np.sqrt(max(0, pred_var)))

        predictions_raw = np.array(predictions_raw)
        se_predictions = np.array(se_predictions)

        # Step 2: Calculate confidence intervals for raw predictions (90% CI like Stata)
        # Note: Stata uses t-distribution, but with large df it's very close to normal
        ci_factor = stats.norm.ppf(0.95)  # 90% CI
        lower_ci_raw = predictions_raw - ci_factor * se_predictions
        upper_ci_raw = predictions_raw + ci_factor * se_predictions

        # Step 3: Normalize by subtracting the maximum estimate (matching R code exactly)
        # R code: mx = max(resp$estimate); est = resp$estimate - mx
        #         min90 = resp$min90 - mx; max90 = resp$max90 - mx
        mx = np.max(predictions_raw)
        predictions = predictions_raw - mx
        lower_ci = lower_ci_raw - mx
        upper_ci = upper_ci_raw - mx

        # Create response function dataframe
        response_data = pd.DataFrame({
            'x': temp_range,
            'estimate': predictions,
            'min90': lower_ci,
            'max90': upper_ci
        })

        # Save results
        response_data.to_csv(OUTPUT_FILES['estimated_global_response'], index=False)

        # Save coefficients (like Stata: mat b = e(b); mat b = b[1,1..2])
        coef_data = pd.DataFrame({
            'temp': [temp_coef],
            'temp2': [temp2_coef]
        })
        coef_data.to_csv(OUTPUT_FILES['estimated_coefficients'], index=False)

        logger.info(f"Global response function saved. Optimal temperature: {optimal_temp:.2f}°C")
        return response_data
    
    def heterogeneity_analysis(self):
        """
        Analyze heterogeneity in temperature responses (Figure 2, panels B, D, E).

        MATCHING STATA'S MARGINS COMMAND:
        Stata's `margins, over(interact) at(temp=(0(1)30))` computes predictions by:
        1. Setting temp to specified values for each interact group
        2. Setting ALL OTHER covariates to their sample means
        3. Using the FULL VCE matrix for standard error calculation

        Original Stata code:
        loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap
        foreach var of loc vars  {
        ...
        qui reg `var' interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
        margins, over(interact) at(temp=(0(1)30)) post noestimcheck force level(90)
        """
        logger.info("Running heterogeneity analysis...")

        results_list = []

        # Variables to analyze (like Stata: loc vars growthWDI AgrGDPgrowthCap NonAgrGDPgrowthCap)
        variables = ['growthWDI', 'AgrGDPgrowthCap', 'NonAgrGDPgrowthCap']

        for var in variables:
            if var not in self.data.columns:
                logger.warning(f"Variable {var} not found in dataset, skipping...")
                continue

            logger.info(f"Analyzing heterogeneity for {var}...")

            # Use unified regression function for heterogeneity analysis
            result_dict = self.run_regression('heterogeneity',
                                           dependent_var=var,
                                           interaction_var='poorWDIppp')

            results = result_dict['results']

            # Get all parameter names and full VCE matrix
            param_names = list(results.params.index)
            n_params = len(param_names)
            full_cov = results.cov_params().values

            # Compute sample means for all covariates (like Stata's margins)
            sample_means = {}
            for name in param_names:
                if name == 'const':
                    sample_means[name] = 1.0
                elif name in ['UDel_temp_popweight', 'UDel_temp_popweight_2',
                             'temp_poor', 'temp2_poor']:
                    sample_means[name] = None  # Will be set by at() and over() values
                elif name in self.data.columns:
                    sample_means[name] = self.data[name].mean()
                else:
                    # Handle derived variables (time trends, FE dummies)
                    if name.startswith('_yi_'):
                        country = name[4:]
                        if country in self.data['iso_id'].values:
                            mask = self.data['iso_id'] == country
                            sample_means[name] = (self.data.loc[mask, 'year'] - 1960).mean() * mask.mean()
                        else:
                            sample_means[name] = 0.0
                    elif name.startswith('_y2_'):
                        country = name[4:]
                        if country in self.data['iso_id'].values:
                            mask = self.data['iso_id'] == country
                            sample_means[name] = ((self.data.loc[mask, 'year'] - 1960)**2).mean() * mask.mean()
                        else:
                            sample_means[name] = 0.0
                    elif name.startswith('iso_'):
                        country = name[4:]
                        sample_means[name] = (self.data['iso_id'] == int(country)).mean() if country.isdigit() else 0.0
                    elif name.startswith('year_'):
                        year = name[5:]
                        sample_means[name] = (self.data['year'] == int(year)).mean() if year.isdigit() else 0.0
                    else:
                        sample_means[name] = 0.0

            # Generate response functions for rich and poor (like Stata: margins, over(interact) at(temp=(0(1)30)))
            temp_range = np.arange(0, 31, 1)
            ci_factor = stats.norm.ppf(0.95)  # 90% CI

            for interact in [0, 1]:  # 0 = rich, 1 = poor
                predictions_raw = []
                se_predictions = []

                for temp_val in temp_range:
                    # Construct full gradient vector (like Stata's margins)
                    grad = np.zeros(n_params)
                    for i, name in enumerate(param_names):
                        if name == 'const':
                            grad[i] = 1.0
                        elif name == 'UDel_temp_popweight':
                            grad[i] = temp_val
                        elif name == 'UDel_temp_popweight_2':
                            grad[i] = temp_val**2
                        elif name == 'temp_poor':
                            grad[i] = temp_val if interact == 1 else 0.0
                        elif name == 'temp2_poor':
                            grad[i] = temp_val**2 if interact == 1 else 0.0
                        elif name == 'precip_poor':
                            # Mean precip for poor countries if interact==1
                            if interact == 1:
                                grad[i] = sample_means.get('UDel_precip_popweight', 0.0)
                            else:
                                grad[i] = 0.0
                        elif name == 'precip2_poor':
                            if interact == 1:
                                grad[i] = sample_means.get('UDel_precip_popweight_2', 0.0)
                            else:
                                grad[i] = 0.0
                        else:
                            grad[i] = sample_means.get(name, 0.0)

                    # Compute prediction: y = gradient' * beta
                    pred = grad @ results.params.values
                    predictions_raw.append(pred)

                    # Compute variance: Var(y) = gradient' * VCE * gradient
                    pred_var = grad.T @ full_cov @ grad
                    se_predictions.append(np.sqrt(max(0, pred_var)))

                predictions_raw = np.array(predictions_raw)
                se_predictions = np.array(se_predictions)

                # Calculate raw CIs
                lower_ci_raw = predictions_raw - ci_factor * se_predictions
                upper_ci_raw = predictions_raw + ci_factor * se_predictions

                # Normalize by subtracting the maximum estimate (matching R code)
                mx = np.max(predictions_raw)
                predictions = predictions_raw - mx
                lower_ci = lower_ci_raw - mx
                upper_ci = upper_ci_raw - mx

                # Create result rows
                for i, temp_val in enumerate(temp_range):
                    results_list.append({
                        'x': temp_val,
                        'estimate': predictions[i],
                        'min90': lower_ci[i],
                        'max90': upper_ci[i],
                        'interact': interact,
                        'model': var
                    })

        # Save heterogeneity results
        heterogeneity_data = pd.DataFrame(results_list)
        heterogeneity_data.to_csv(OUTPUT_FILES['effect_heterogeneity'], index=False)

        logger.info("Heterogeneity analysis completed")
        return heterogeneity_data
    
    def temporal_heterogeneity(self):
        """
        Analyze temporal heterogeneity (Figure 2, panel C).

        MATCHING STATA'S MARGINS COMMAND:
        Stata's `margins, over(interact) at(temp=(0(1)30))` computes predictions by:
        1. Setting temp to specified values for each interact group (early/late)
        2. Setting ALL OTHER covariates to their sample means
        3. Using the FULL VCE matrix for standard error calculation

        Original Stata code:
        qui reg growthWDI interact#c.(c.temp##c.temp UDel_precip_popweight UDel_precip_popweight_2)  _yi_* _y2_* i.year i.iso_id, cl(iso_id)
        margins, over(interact) at(temp=(0(1)30)) post noestimcheck level(90)
        """
        logger.info("Running temporal heterogeneity analysis...")

        # Use unified regression function for temporal heterogeneity analysis
        result_dict = self.run_regression('temporal',
                                       dependent_var='growthWDI',
                                       interaction_var='early')

        results = result_dict['results']

        # Get all parameter names and full VCE matrix
        param_names = list(results.params.index)
        n_params = len(param_names)
        full_cov = results.cov_params().values

        # Compute sample means for all covariates (like Stata's margins)
        sample_means = {}
        for name in param_names:
            if name == 'const':
                sample_means[name] = 1.0
            elif name in ['UDel_temp_popweight', 'UDel_temp_popweight_2',
                         'temp_early', 'temp2_early']:
                sample_means[name] = None  # Will be set by at() and over() values
            elif name in self.data.columns:
                sample_means[name] = self.data[name].mean()
            else:
                # Handle derived variables (time trends, FE dummies)
                if name.startswith('_yi_'):
                    country = name[4:]
                    if country in self.data['iso_id'].values:
                        mask = self.data['iso_id'] == country
                        sample_means[name] = (self.data.loc[mask, 'year'] - 1960).mean() * mask.mean()
                    else:
                        sample_means[name] = 0.0
                elif name.startswith('_y2_'):
                    country = name[4:]
                    if country in self.data['iso_id'].values:
                        mask = self.data['iso_id'] == country
                        sample_means[name] = ((self.data.loc[mask, 'year'] - 1960)**2).mean() * mask.mean()
                    else:
                        sample_means[name] = 0.0
                elif name.startswith('iso_'):
                    country = name[4:]
                    sample_means[name] = (self.data['iso_id'] == int(country)).mean() if country.isdigit() else 0.0
                elif name.startswith('year_'):
                    year = name[5:]
                    sample_means[name] = (self.data['year'] == int(year)).mean() if year.isdigit() else 0.0
                else:
                    sample_means[name] = 0.0

        # Generate response functions for early and late periods
        temp_range = np.arange(0, 31, 1)
        ci_factor = stats.norm.ppf(0.95)  # 90% CI
        results_list = []

        for interact in [0, 1]:  # 0 = late, 1 = early
            predictions_raw = []
            se_predictions = []

            for temp_val in temp_range:
                # Construct full gradient vector (like Stata's margins)
                grad = np.zeros(n_params)
                for i, name in enumerate(param_names):
                    if name == 'const':
                        grad[i] = 1.0
                    elif name == 'UDel_temp_popweight':
                        grad[i] = temp_val
                    elif name == 'UDel_temp_popweight_2':
                        grad[i] = temp_val**2
                    elif name == 'temp_early':
                        grad[i] = temp_val if interact == 1 else 0.0
                    elif name == 'temp2_early':
                        grad[i] = temp_val**2 if interact == 1 else 0.0
                    elif name == 'precip_early':
                        # Mean precip for early period if interact==1
                        if interact == 1:
                            grad[i] = sample_means.get('UDel_precip_popweight', 0.0)
                        else:
                            grad[i] = 0.0
                    elif name == 'precip2_early':
                        if interact == 1:
                            grad[i] = sample_means.get('UDel_precip_popweight_2', 0.0)
                        else:
                            grad[i] = 0.0
                    else:
                        grad[i] = sample_means.get(name, 0.0)

                # Compute prediction: y = gradient' * beta
                pred = grad @ results.params.values
                predictions_raw.append(pred)

                # Compute variance: Var(y) = gradient' * VCE * gradient
                pred_var = grad.T @ full_cov @ grad
                se_predictions.append(np.sqrt(max(0, pred_var)))

            predictions_raw = np.array(predictions_raw)
            se_predictions = np.array(se_predictions)

            # Calculate raw CIs
            lower_ci_raw = predictions_raw - ci_factor * se_predictions
            upper_ci_raw = predictions_raw + ci_factor * se_predictions

            # Normalize by subtracting the maximum estimate (matching R code)
            mx = np.max(predictions_raw)
            predictions = predictions_raw - mx
            lower_ci = lower_ci_raw - mx
            upper_ci = upper_ci_raw - mx

            # Create result rows
            for i, temp_val in enumerate(temp_range):
                results_list.append({
                    'x': temp_val,
                    'estimate': predictions[i],
                    'min90': lower_ci[i],
                    'max90': upper_ci[i],
                    'interact': interact
                })

        # Save temporal heterogeneity results
        temporal_data = pd.DataFrame(results_list)
        temporal_data.to_csv(OUTPUT_FILES['effect_heterogeneity_time'], index=False)

        logger.info("Temporal heterogeneity analysis completed")
        return temporal_data
    
    def bootstrap_analysis(self):
        """
        Run bootstrap analysis to quantify uncertainty in regression coefficients.

        PURPOSE:
        The baseline regression gives us ONE set of coefficients, but these are uncertain
        (subject to sampling variability). For future climate projections, we need to
        propagate this uncertainty forward. Bootstrap creates 1,000 alternative coefficient
        sets by resampling the data.

        HOW BOOTSTRAP WORKS:
        1. Randomly sample countries WITH REPLACEMENT (some countries appear multiple times)
        2. Re-run the regression on this resampled dataset
        3. Store the coefficients from this run
        4. Repeat 1,000 times
        5. Result: 1,000 sets of plausible coefficients reflecting estimation uncertainty

        WHY RESAMPLE COUNTRIES (NOT OBSERVATIONS)?
        - We want to preserve the time-series structure within each country
        - Resampling at the country level respects the panel structure of the data
        - This is called "cluster bootstrap" (clusters = countries)

        FOUR MODEL SPECIFICATIONS:
        1. Pooled (no lags): All countries respond the same, immediate effects only
        2. Rich/Poor (no lags): Rich and poor countries respond differently
        3. Pooled (5 lags): All countries respond the same, effects persist for 5 years
        4. Rich/Poor (5 lags): Heterogeneous responses with persistent effects

        OUTPUT:
        Bootstrap CSV files containing 1,000 coefficient sets for each specification.
        These will be used in Step 4 to create uncertainty bounds on future projections.
        """
        logger.info("Starting bootstrap analysis...")

        np.random.seed(RANDOM_SEED)
        
        # Get unique countries
        countries = self.data['iso_id'].unique()
        n_countries = len(countries)
        
        # Bootstrap specifications
        bootstrap_specs = [
            ('no_lag', self._bootstrap_pooled_no_lag),
            ('rich_poor', self._bootstrap_rich_poor_no_lag),
            ('5_lag', self._bootstrap_pooled_5_lag),
            ('rich_poor_5_lag', self._bootstrap_rich_poor_5_lag)
        ]
        
        for spec_name, bootstrap_func in bootstrap_specs:
            logger.info(f"Running bootstrap for {spec_name}...")
            bootstrap_func(countries, n_countries)
    
    def _bootstrap_pooled_no_lag(self, countries, n_countries):
        """
        Bootstrap pooled model with no lags (matching Stata exactly).
        
        Original Stata code:
        cap postutil clear
        postfile boot run temp temp2 prec prec2 using data/output/bootstrap/bootstrap_noLag, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id 
        post boot (0) (_b[UDel_temp_popweight]) (_b[UDel_temp_popweight_2]) (_b[UDel_precip_popweight]) (_b[UDel_precip_popweight_2])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id)  //draw a sample of countries with replacement
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id 
        post boot (`nn') (_b[UDel_temp_popweight]) (_b[UDel_temp_popweight_2]) (_b[UDel_precip_popweight]) (_b[UDel_precip_popweight_2])
        }
        """
        results_list = []
        
        # Baseline run (no resampling) - like Stata: first run is baseline
        baseline_results = self._run_pooled_no_lag_regression(self.data)
        results_list.append({
            'run': 0,
            'temp': baseline_results['temp'],
            'temp2': baseline_results['temp2'],
            'prec': baseline_results['prec'],
            'prec2': baseline_results['prec2']
        })
        
        # Bootstrap runs (like Stata: forvalues nn = 1/1000)
        # Note: Using N_BOOTSTRAP = 10 for testing, set to 1000 for full replication
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Bootstrap pooled no lag"):
            # Sample countries with replacement (like Stata: bsample, cl(iso_id))
            sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
            
            # Build bootstrap sample with unique boot_cluster_id for each resampled country (idcluster equivalent)
            bootstrap_data = []
            cluster_counter = 0
            for country in sampled_countries:
                country_data = self.data[self.data['iso_id'] == country].copy()
                country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                bootstrap_data.append(country_data)
                cluster_counter += 1
            bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
            
            # Run regression
            try:
                results = self._run_pooled_no_lag_regression(bootstrap_sample)
                results_list.append({
                    'run': run,
                    'temp': results['temp'],
                    'temp2': results['temp2'],
                    'prec': results['prec'],
                    'prec2': results['prec2']
                })
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        # Save results
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'temppoor' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'temppoor', 'temp2', 'temp2poor', 'prec', 'precpoor', 'prec2', 'prec2poor']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_no_lag'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap pooled no lag completed: {len(results_list)} successful runs")
    
    def _run_pooled_no_lag_regression(self, data):
        """
        Run pooled regression with no lags (matching Stata specification).
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2 i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_pooled_no_lag', data=data, create_time_trends=True)
        return {
            'temp': result_dict['temp'],
            'temp2': result_dict['temp2'],
            'prec': result_dict['prec'],
            'prec2': result_dict['prec2']
        }
    
    def _bootstrap_rich_poor_no_lag(self, countries, n_countries):
        """
        Bootstrap rich/poor model with no lags.
        
        Original Stata code:
        cap postutil clear
        postfile boot run temp temppoor temp2 temp2poor prec precpoor prec2 prec2poor using data/output/bootstrap/bootstrap_richpoor, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (0) (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id)  //draw a sample of countries with replacement
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (`nn') (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8])
        }
        """
        results_list = []
        
        # Baseline run
        baseline_results = self._run_rich_poor_no_lag_regression(self.data)
        results_list.append({
            'run': 0,
            'temp': baseline_results['temp'],
            'temppoor': baseline_results['temppoor'],
            'temp2': baseline_results['temp2'],
            'temp2poor': baseline_results['temp2poor'],
            'prec': baseline_results['prec'],
            'precpoor': baseline_results['precpoor'],
            'prec2': baseline_results['prec2'],
            'prec2poor': baseline_results['prec2poor']
        })
        
        # Bootstrap runs (like Stata: forvalues nn = 1/1000)
        # Note: Using N_BOOTSTRAP = 10 for testing, set to 1000 for full replication
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Bootstrap rich/poor no lag"):
            sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
            
            bootstrap_data = []
            cluster_counter = 0
            for country in sampled_countries:
                country_data = self.data[self.data['iso_id'] == country].copy()
                country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                bootstrap_data.append(country_data)
                cluster_counter += 1
            bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
            
            try:
                results = self._run_rich_poor_no_lag_regression(bootstrap_sample)
                results_list.append({
                    'run': run,
                    'temp': results['temp'],
                    'temppoor': results['temppoor'],
                    'temp2': results['temp2'],
                    'temp2poor': results['temp2poor'],
                    'prec': results['prec'],
                    'precpoor': results['precpoor'],
                    'prec2': results['prec2'],
                    'prec2poor': results['prec2poor']
                })
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'temppoor' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'temppoor', 'temp2', 'temp2poor', 'prec', 'precpoor', 'prec2', 'prec2poor']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_rich_poor'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap rich/poor no lag completed: {len(results_list)} successful runs")
    
    def _run_rich_poor_no_lag_regression(self, data):
        """
        Run rich/poor regression with no lags (matching Stata specification).
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_rich_poor_no_lag', 
                                       data=data, 
                                       create_time_trends=True,
                                       interaction_var='poorWDIppp')
        return {
            'temp': result_dict['temp'],
            'temppoor': result_dict['temppoor'],
            'temp2': result_dict['temp2'],
            'temp2poor': result_dict['temp2poor'],
            'prec': result_dict['prec'],
            'precpoor': result_dict['precpoor'],
            'prec2': result_dict['prec2'],
            'prec2poor': result_dict['prec2poor']
        }
    
    def _create_lagged_variables(self, data):
        """
        Create lagged temperature and precipitation variables for dynamic models.

        PURPOSE:
        Temperature shocks may have PERSISTENT effects on growth, not just immediate impacts.
        For example:
        - A hot year destroys crops → less agricultural income
        - Less income → less savings and investment
        - Less investment → lower growth in subsequent years

        Lags capture this dynamic adjustment process.

        WHAT THIS CREATES:
        For each variable (temp, temp², precip, precip²):
        - L1: Value from 1 year ago
        - L2: Value from 2 years ago
        - L3: Value from 3 years ago
        - L4: Value from 4 years ago
        - L5: Value from 5 years ago

        INTERPRETATION IN REGRESSION:
        growth_t = β₀·temp_t + β₁·temp_{t-1} + β₂·temp_{t-2} + ... + β₅·temp_{t-5}

        - β₀: Immediate (contemporaneous) effect of current temperature
        - β₁: Effect of last year's temperature on this year's growth
        - Sum(β₀...β₅): Total long-run effect after all dynamics play out

        WHY 5 LAGS?
        - Economic shocks typically dissipate within 5 years
        - Beyond 5 years, effects become indistinguishable from noise
        - This follows standard practice in macroeconomic time series analysis

        Equivalent to Stata: xtset iso_id year
        """
        data_copy = data.copy()

        # Sort by country and year (required for lagging to work correctly)
        data_copy = data_copy.sort_values(['iso_id', 'year'])

        # Check for missing years within each country (gaps would break lag calculations)
        for country, group in data_copy.groupby('iso_id'):
            years = group['year'].values
            missing_years = set(range(years.min(), years.max()+1)) - set(years)
            if missing_years:
                logger.warning(f"Country {country} has missing years: {sorted(missing_years)}")
        
        # Create lagged variables for temperature and precipitation
        for lag in range(1, 6):  # L1 through L5
            # Temperature lags
            data_copy[f'L{lag}temp'] = data_copy.groupby('iso_id')['UDel_temp_popweight'].shift(lag)
            data_copy[f'L{lag}temp2'] = data_copy.groupby('iso_id')['UDel_temp_popweight_2'].shift(lag)
            # Precipitation lags
            data_copy[f'L{lag}prec'] = data_copy.groupby('iso_id')['UDel_precip_popweight'].shift(lag)
            data_copy[f'L{lag}prec2'] = data_copy.groupby('iso_id')['UDel_precip_popweight_2'].shift(lag)
        
        # For rich/poor model, also create interaction lags
        if 'poorWDIppp' in data_copy.columns:
            for lag in range(1, 6):
                # Poor interaction lags
                data_copy[f'L{lag}temppoor'] = data_copy[f'L{lag}temp'] * data_copy['poorWDIppp']
                data_copy[f'L{lag}temp2poor'] = data_copy[f'L{lag}temp2'] * data_copy['poorWDIppp']
                data_copy[f'L{lag}precpoor'] = data_copy[f'L{lag}prec'] * data_copy['poorWDIppp']
                data_copy[f'L{lag}prec2poor'] = data_copy[f'L{lag}prec2'] * data_copy['poorWDIppp']
        
        return data_copy
    
    def _run_pooled_5_lag_regression(self, data):
        """
        Run pooled regression with 5 lags (matching Stata specification).
        
        Original Stata code:
        xtset iso_id year
        qui reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_pooled_5_lag', 
                                       data=data, 
                                       create_time_trends=True,
                                       use_lags=True)
        return {
            'temp': result_dict['temp'],
            'L1temp': result_dict['L1temp'],
            'L2temp': result_dict['L2temp'],
            'L3temp': result_dict['L3temp'],
            'L4temp': result_dict['L4temp'],
            'L5temp': result_dict['L5temp'],
            'temp2': result_dict['temp2'],
            'L1temp2': result_dict['L1temp2'],
            'L2temp2': result_dict['L2temp2'],
            'L3temp2': result_dict['L3temp2'],
            'L4temp2': result_dict['L4temp2'],
            'L5temp2': result_dict['L5temp2'],
            'tlin': result_dict['tlin'],
            'tsq': result_dict['tsq']
        }
    
    def _run_rich_poor_5_lag_regression(self, data):
        """
        Run rich/poor regression with 5 lags (matching Stata specification).
        
        Original Stata code:
        xtset iso_id year
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id
        """
        # Use unified regression function
        result_dict = self.run_regression('bootstrap_rich_poor_5_lag', 
                                       data=data, 
                                       create_time_trends=True,
                                       use_lags=True,
                                       interaction_var='poorWDIppp')
        return {
            'temp': result_dict['temp'],
            'L1temp': result_dict['L1temp'],
            'L2temp': result_dict['L2temp'],
            'L3temp': result_dict['L3temp'],
            'L4temp': result_dict['L4temp'],
            'L5temp': result_dict['L5temp'],
            'temp2': result_dict['temp2'],
            'L1temp2': result_dict['L1temp2'],
            'L2temp2': result_dict['L2temp2'],
            'L3temp2': result_dict['L3temp2'],
            'L4temp2': result_dict['L4temp2'],
            'L5temp2': result_dict['L5temp2'],
            'temppoor': result_dict['temppoor'],
            'L1temppoor': result_dict['L1temppoor'],
            'L2temppoor': result_dict['L2temppoor'],
            'L3temppoor': result_dict['L3temppoor'],
            'L4temppoor': result_dict['L4temppoor'],
            'L5temppoor': result_dict['L5temppoor'],
            'temp2poor': result_dict['temp2poor'],
            'L1temp2poor': result_dict['L1temp2poor'],
            'L2temp2poor': result_dict['L2temp2poor'],
            'L3temp2poor': result_dict['L3temp2poor'],
            'L4temp2poor': result_dict['L4temp2poor'],
            'L5temp2poor': result_dict['L5temp2poor']
        }
    
    def _bootstrap_pooled_5_lag(self, countries, n_countries):
        """
        Bootstrap pooled model with 5 lags.
        
        Original Stata code:
        postfile boot run temp L1temp L2temp L3temp L4temp L5temp temp2 L1temp2 L2temp2 L3temp2 L4temp2 L5temp2  using data/output/bootstrap/bootstrap_5Lag, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        xtset iso_id year
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        mat b = e(b)
        post boot (0) (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id) idcluster(id) //draw a sample of countries with replacement
        xtset id year  //need to use the new cluster variable it creates. 
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2	
        qui reg growthWDI L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2) i.year _yi_* _y2_* i.iso_id
        mat b = e(b)
        post boot (`nn') (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12])
        }
        """
        logger.info("Running bootstrap pooled 5 lag...")
        
        results_list = []
        
        # Baseline run
        baseline_results = self._run_pooled_5_lag_regression(self.data)
        baseline_results['run'] = 0
        results_list.append(baseline_results)
        
        # Bootstrap runs
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Pooled 5-lag bootstrap"):
            try:
                # Sample countries with replacement
                sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
                
                # Build bootstrap sample with unique boot_cluster_id for each resampled country (idcluster equivalent)
                bootstrap_data = []
                cluster_counter = 0
                for country in sampled_countries:
                    country_data = self.data[self.data['iso_id'] == country].copy()
                    country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                    bootstrap_data.append(country_data)
                    cluster_counter += 1
                bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
                
                # Run regression
                results = self._run_pooled_5_lag_regression(bootstrap_sample)
                results['run'] = run
                results_list.append(results)
                
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'L1temp' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'L1temp', 'L2temp', 'L3temp', 'L4temp', 'L5temp', 'temp2', 'L1temp2', 'L2temp2', 'L3temp2', 'L4temp2', 'L5temp2']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_5_lag'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap pooled 5 lag completed: {len(results_list)} successful runs")
    
    def _bootstrap_rich_poor_5_lag(self, countries, n_countries):
        """
        Bootstrap rich/poor model with 5 lags.
        
        Original Stata code:
        postfile boot run temp temppoor L1temp L1temppoor L2temp L2temppoor L3temp L3temppoor L4temp L4temppoor L5temp L5temppoor ///
        temp2 temp2poor L1temp2 L1temp2poor L2temp2 L2temp2poor L3temp2 L3temp2poor L4temp2 L4temp2poor L5temp2 L5temp2poor ///
        using data/output/bootstrap/bootstrap_richpoor_5lag, replace
        set seed 8675309
        use data/input/GrowthClimateDataset, clear
        xtset iso_id year
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2
        qui gen poor = (GDPpctile_WDIppp<50)
        qui replace poor=. if GDPpctile_WDIppp==.
        qui reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (0) (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12]) (b[1,13]) (b[1,14]) (b[1,15]) (b[1,16]) (b[1,17]) (b[1,18]) (b[1,19]) (b[1,20]) (b[1,21]) (b[1,22]) (b[1,23]) (b[1,24])
        forvalues nn = 1/1000 {
        use data/input/GrowthClimateDataset, clear
        bsample, cl(iso_id) idcluster(id) //draw a sample of countries with replacement
        qui xtset id year  //need to use the new cluster variable it creates. 
        qui gen UDel_temp_popweight_2 = UDel_temp_popweight^2	
        qui reg growthWDI poor#c.(L(0/5).(UDel_temp_popweight UDel_temp_popweight_2 UDel_precip_popweight UDel_precip_popweight_2)) i.year _yi_* _y2_* i.iso_id 
        mat b = e(b)
        post boot (`nn') (b[1,1]) (b[1,2]) (b[1,3]) (b[1,4]) (b[1,5]) (b[1,6]) (b[1,7]) (b[1,8]) (b[1,9]) (b[1,10]) (b[1,11]) (b[1,12]) (b[1,13]) (b[1,14]) (b[1,15]) (b[1,16]) (b[1,17]) (b[1,18]) (b[1,19]) (b[1,20]) (b[1,21]) (b[1,22]) (b[1,23]) (b[1,24])
        }
        """
        logger.info("Running bootstrap rich/poor 5 lag...")
        
        results_list = []
        
        # Baseline run
        baseline_results = self._run_rich_poor_5_lag_regression(self.data)
        baseline_results['run'] = 0
        results_list.append(baseline_results)
        
        # Bootstrap runs
        for run in tqdm(range(1, N_BOOTSTRAP + 1), desc="Rich/poor 5-lag bootstrap"):
            try:
                # Sample countries with replacement
                sampled_countries = np.random.choice(countries, size=n_countries, replace=True)
                
                # Build bootstrap sample with unique boot_cluster_id for each resampled country (idcluster equivalent)
                bootstrap_data = []
                cluster_counter = 0
                for country in sampled_countries:
                    country_data = self.data[self.data['iso_id'] == country].copy()
                    country_data['boot_cluster_id'] = cluster_counter  # assign new cluster id
                    bootstrap_data.append(country_data)
                    cluster_counter += 1
                bootstrap_sample = pd.concat(bootstrap_data, ignore_index=True)
                
                # Run regression
                results = self._run_rich_poor_5_lag_regression(bootstrap_sample)
                results['run'] = run
                results_list.append(results)
                
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
        
        # DEBUG: Print all unique keys in results_list before DataFrame creation
        all_keys = set()
        for d in results_list:
            all_keys.update(d.keys())
        print(f"DEBUG: Unique keys in results_list before DataFrame creation: {sorted(all_keys)}")
        
        bootstrap_df = pd.DataFrame(results_list)
        # Define the exact column order as in Stata (adjust as needed for each bootstrap type)
        if 'temppoor' in bootstrap_df.columns:
            column_order = ['run', 'temp', 'temppoor', 'L1temp', 'L1temppoor', 'L2temp', 'L2temppoor', 'L3temp', 'L3temppoor', 'L4temp', 'L4temppoor', 'L5temp', 'L5temppoor', 'temp2', 'temp2poor', 'L1temp2', 'L1temp2poor', 'L2temp2', 'L2temp2poor', 'L3temp2', 'L3temp2poor', 'L4temp2', 'L4temp2poor', 'L5temp2', 'L5temp2poor']
        else:
            column_order = ['run', 'temp', 'temp2', 'prec', 'prec2']
        # Reorder columns if all are present
        bootstrap_df = bootstrap_df[[col for col in column_order if col in bootstrap_df.columns]]
        # Replace NaN with '.' for Stata compatibility (optional)
        bootstrap_df = bootstrap_df.where(pd.notnull(bootstrap_df), '.')
        # Save with explicit float format and no index, matching Stata's postfile output
        bootstrap_df.to_csv(OUTPUT_FILES['bootstrap_rich_poor_5_lag'], index=False, float_format='%.8f')
        logger.info(f"Bootstrap results saved with columns: {bootstrap_df.columns.tolist()} and 8 decimal precision (Stata postfile compatible)")
        logger.info(f"Bootstrap rich/poor 5 lag completed: {len(results_list)} successful runs")
    
    def save_main_dataset(self):
        """
        Save the main dataset for use in later steps.
        
        Original Stata code:
        use data/input/GrowthClimateDataset, clear
        keep UDel_temp_popweight Pop TotGDP growthWDI GDPpctile_WDIppp continent iso countryname year
        outsheet using data/output/mainDataset.csv, comma replace
        """
        logger.info("Saving main dataset...")
        
        # Select relevant columns (like Stata: keep UDel_temp_popweight Pop TotGDP growthWDI GDPpctile_WDIppp continent iso countryname year)
        main_cols = ['UDel_temp_popweight', 'Pop', 'TotGDP', 'growthWDI', 
                    'GDPpctile_WDIppp', 'continent', 'iso', 'countryname', 'year']
        
        main_data = self.data[main_cols].copy()
        main_data.to_csv(OUTPUT_FILES['main_dataset'], index=False)
        
        logger.info("Main dataset saved")
        return main_data

def run_step1():
    """Run Step 1: Data Preparation and Initial Analysis."""
    logger.info("Starting Step 1: Data Preparation and Initial Analysis")
    
    # Initialize
    processor = BurkeDataPreparation()
    
    # Load and prepare data
    processor.load_data()
    processor.prepare_data()
    
    # Run baseline regression
    baseline_results = processor.baseline_regression()
    
    # Generate global response function
    processor.generate_global_response(baseline_results)
    
    # Run heterogeneity analysis
    processor.heterogeneity_analysis()
    processor.temporal_heterogeneity()
    
    # Run bootstrap analysis
    processor.bootstrap_analysis()
    
    # Save main dataset
    processor.save_main_dataset()
    
    logger.info("Step 1 completed successfully")

if __name__ == "__main__":
    run_step1() 