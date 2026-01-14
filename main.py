"""
Main processing script for Burke, Hsiang, and Miguel 2015 replication.

This script orchestrates all the major processing steps and can skip steps
if output files already exist and skip flags are set.

Usage:
    python main.py [run_name]

    run_name: Optional name for this run (default: "default")
              Output directory will be: output_{run_name}_{timestamp}
              Output files will include _{run_name} suffix

Examples:
    python main.py              # Uses run_name="default"
    python main.py test         # Uses run_name="test"
    python main.py experiment1  # Uses run_name="experiment1"
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Parse command line arguments BEFORE importing config
def get_run_name():
    """Get run_name from command line arguments."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "default"

# Get run_name and initialize paths before other imports
run_name = get_run_name()

# Now import config and initialize paths with the run_name
import config
config.initialize_paths(run_name)

# Import the rest after config is initialized
from config import (
    PROJECT_ROOT, OUTPUT_PATH, FIGURES_PATH, OUTPUT_FILES,
    SKIP_STEP_1, SKIP_STEP_2, SKIP_STEP_3, SKIP_STEP_4, SKIP_STEP_5, SKIP_STEP_6,
    RUN_NAME, get_figure_filename
)
from step1_data_preparation import run_step1
from step2_climate_projections import run_step2
from step3_socioeconomic_scenarios import run_step3
from step4_impact_projections import run_step4
from step5_damage_function import run_step5
from step6_figure_generation import run_step6

def setup_logging():
    """Set up logging configuration."""
    from config import setup_logging as config_setup_logging
    return config_setup_logging()

def cleanup_timestamp_file():
    """Clean up the timestamp file after pipeline completion."""
    timestamp_file = PROJECT_ROOT / ".current_run_timestamp"
    if timestamp_file.exists():
        timestamp_file.unlink()
        logger.info("Cleaned up timestamp file")

def check_skip_condition(skip_flag, output_files, step_name):
    """Check if a step should be skipped based on flag only."""
    if skip_flag:
        logger.info(f"Skipping {step_name} - skip flag is set")
        return True
    return False

def main():
    """Main processing function."""
    global logger
    logger = setup_logging()

    logger.info("Starting Burke, Hsiang, and Miguel 2015 replication")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Run name: {RUN_NAME}")
    logger.info(f"Output directory: {OUTPUT_PATH}")


    # Step 1: Data Preparation and Initial Analysis
    if not check_skip_condition(SKIP_STEP_1, [
        OUTPUT_FILES['estimated_global_response'],
        OUTPUT_FILES['estimated_coefficients'],
        OUTPUT_FILES['main_dataset'],
        OUTPUT_FILES['effect_heterogeneity'],
        OUTPUT_FILES['effect_heterogeneity_time']
    ], "Step 1 (Data Preparation)"):
        logger.info("Running Step 1: Data Preparation and Initial Analysis")
        run_step1()

    # Step 2: Climate Projections
    if not check_skip_condition(SKIP_STEP_2, [
        OUTPUT_FILES['country_temp_change']
    ], "Step 2 (Climate Projections)"):
        logger.info("Running Step 2: Climate Projections")
        run_step2()

    # Step 3: Socioeconomic Scenarios
    if not check_skip_condition(SKIP_STEP_3, [
        OUTPUT_FILES['pop_projections'],
        OUTPUT_FILES['growth_projections']
    ], "Step 3 (Socioeconomic Scenarios)"):
        logger.info("Running Step 3: Socioeconomic Scenarios")
        run_step3()

    # Step 4: Impact Projections
    if not check_skip_condition(SKIP_STEP_4, [
        OUTPUT_PATH / "projectionOutput" / "GDPcapCC_pooled_base.Rdata",
        OUTPUT_PATH / "projectionOutput" / "GDPcapNoCC_pooled_base.Rdata"
    ], "Step 4 (Impact Projections)"):
        logger.info("Running Step 4: Impact Projections")
        run_step4()

    # Step 5: Damage Function
    if not check_skip_condition(SKIP_STEP_5, [
        OUTPUT_PATH / "projectionOutput" / "DamageFunction_pooled.Rdata",
        OUTPUT_PATH / "projectionOutput" / "DamageFunction_richpoor.Rdata"
    ], "Step 5 (Damage Function)"):
        logger.info("Running Step 5: Damage Function")
        run_step5()

    # Step 6: Figure Generation
    if not check_skip_condition(SKIP_STEP_6, [
        get_figure_filename("Figure2"),
        get_figure_filename("Figure3"),
        get_figure_filename("Figure4"),
        get_figure_filename("Figure5")
    ], "Step 6 (Figure Generation)"):
        logger.info("Running Step 6: Figure Generation")
        run_step6()

    logger.info("Burke replication processing completed successfully!")
    cleanup_timestamp_file()

if __name__ == "__main__":
    main()
