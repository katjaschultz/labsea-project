# labsea-project

This project provides tools and scripts for processing, analyzing, and visualizing oceanographic data from the Labrador Sea within 75km of the AR7W line, with a focus on Argo float observations and overturning transport calculations.

## Features

- Automated download of Argo float data for the Labrador Sea region
- Preprocessing and masking of profiles by season and year
- Calculation of gridded composite sections (e.g., specific volume anomaly, density, salinity, temperature)
- Calculation of absolute geostrophic velocities and overturning streamfunctions
- Transport diagnostics in depth and density space
- Plotting functions for velocity, overturning, and transport

## Project Structure

```
labsea_project/
    __init__.py
    my_readers.py      # Data loading and fetching routines
    writers.py         # Dataset creation and saving
    tools.py           # Core computational functions
    plotters.py        # Plotting and visualization
    reference_func.py  # Reference velocity and polynomial fitting
    utilities.py       # Helper functions (coordinates, rotation, etc.)
notebooks/
    demo.ipynb         # Example workflow notebook
data/
    ...                # Data files (downloaded, processed, masks, etc.)
scripts/
    ...                # Processing scripts
```

## Usage
Clone the repository and activate a new environment on your local machine

1. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

2. **Run the demo notebook:**
    Open `notebooks/demo.ipynb` in Jupyter or VS Code and follow the workflow to download, process, and analyze Argo data.

3. **Scripts and modules:**
    - Use functions from `labsea_project/` in your own scripts or notebooks for custom analyses.

## Notes

- Data download may take a while and requires an internet connection.
- The project is modular: you can use individual functions or the full workflow as needed.
- For questions or contributions, please contact the