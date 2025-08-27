# 🌊 labsea-project

This project provides tools and scripts for processing, analyzing, and visualizing oceanographic data from the Labrador Sea within 75km of the AR7W line, with a focus on Argo float observations and overturning transport calculations.

## 🚀 Features

* 📥 Automated download of Argo float data for the Labrador Sea region
* 🛠️ Preprocessing and masking of profiles by season and year
* 📊 Calculation of gridded composite sections (e.g., specific volume anomaly, density, salinity, temperature)
* 🌐 Calculation of absolute geostrophic velocities and overturning streamfunctions
* 🌡️ Transport diagnostics in depth and density space
* 📈 Plotting functions for velocity, overturning, and transport

## 📂 Project Structure

```
labsea_project/
    __init__.py
    my_readers.py      #  Data loading and fetching routines
    writers.py         #  Dataset creation and saving
    tools.py           #  Core computational functions
    plotters.py        #  Plotting and visualization
    reference_func.py  #  Reference velocity and polynomial fitting
    utilities.py       #  Helper functions (coordinates, rotation, etc.)
notebooks/
    demo.ipynb         # Example workflow notebook
demo data/             # data you need for the demo (coordinates, topography, etc.)
data/                  # Data files (downloaded, processed, masks, etc.), this folder should be created when working with the demo
scripts/               # Processing scripts
```

## 🔧 Usage

Clone the repository and activate a new environment on your local machine
Python version: 3.12.1
   ```bash
   py -3.12 -m venv <name-venv>
   .\<name-env>\Scripts\activate      
   ```


1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Depending on your python setup, you may have to install **jupyer notebook** or other dependencies in the environment to run the notebook.
   
   If you are having trouble installing **argopy**, please visit the documentation for help: https://argopy.readthedocs.io/en/latest/install.html

3. **Run the demo notebook:**
   Open `notebooks/demo.ipynb` and follow the workflow to download, process, and analyze Argo data.

4. **Scripts and modules:**

   * Use functions from `labsea_project/` in your own scripts or notebooks for custom analyses.

## 📌 Notes

* 🌐 Data download may take a while and requires an internet connection.
* 🧩 The project is in some extent modular: you can use individual functions (tools, argo data loading/ processing, ...) or the full workflow as needed.
* ✉️ For questions or contributions, please contact me.
