# Windows Setup Guide

## Quick Start for Windows Users

### Step 1: Install Python Dependencies

Open Command Prompt or PowerShell in your project directory:

```cmd
cd C:\Users\st_ha\OneDrive\Documents\github\automated_trading_test_environment.jl\src\LiquidityAPI
pip install -r requirements.txt
```

### Step 2: Get FRED API Key

1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Copy your API key (looks like: `abc123def456...`)

### Step 3: Configure API Key

Open `global_liquidity_factor.py` in your text editor and find this line (around line 31):

```python
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"
```

Replace it with your actual key:

```python
FRED_API_KEY = "abc123def456ghi789..."  # Your actual key
```

Save the file.

### Step 4: Run the Script

From the same directory:

```cmd
python global_liquidity_factor.py
```

Or use the quick start:

```cmd
python quick_start.py
```

### Where Are My Files?

All output files will be saved in your **current working directory**:

```
C:\Users\st_ha\OneDrive\Documents\github\automated_trading_test_environment.jl\src\LiquidityAPI\
├── global_liquidity_data.csv              ← All indicators (normalized)
├── global_liquidity_factor.csv            ← The liquidity factor
├── global_liquidity_factor_chart.png      ← Visualization
└── liquidity_correlation_matrix.png       ← Correlation heatmap
```

### Common Windows Issues

#### Issue 1: Python Not Found

**Error**: `'python' is not recognized as an internal or external command`

**Solution**: 
```cmd
py global_liquidity_factor.py
```
Or add Python to your PATH environment variable.

#### Issue 2: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'pandas'`

**Solution**:
```cmd
pip install pandas numpy requests matplotlib seaborn scikit-learn
```

Or:
```cmd
python -m pip install -r requirements.txt
```

#### Issue 3: Permission Denied

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**: Run Command Prompt as Administrator or save files to a different directory.

#### Issue 4: Long Path Names

If you get errors related to path length, move the project to a shorter path:

```cmd
# Instead of:
C:\Users\st_ha\OneDrive\Documents\github\automated_trading_test_environment.jl\src\LiquidityAPI\

# Try:
C:\Projects\LiquidityAPI\
```

### Customizing Output Directory (Optional)

If you want to save files to a specific directory, modify the save methods:

```python
# In global_liquidity_factor.py, find save_data() method:
def save_data(self, filename='results/global_liquidity_data.csv'):
    """Save the raw and normalized data"""
    if not self.data.empty:
        output_path = r'C:\Users\st_ha\Documents\outputs'  # Your custom path
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        self.data.to_csv(full_path)
        print(f"\nData saved to {full_path}")
```

Don't forget to add at the top:
```python
import os
```

### Running in Jupyter Notebook (Windows)

1. Install Jupyter:
```cmd
pip install jupyter
```

2. Start Jupyter:
```cmd
jupyter notebook
```

3. Create a new notebook and run:
```python
from global_liquidity_factor import GlobalLiquidityFactor, Config

# Set your API key
Config.FRED_API_KEY = "your_api_key_here"

# Initialize and run
glf = GlobalLiquidityFactor(Config.FRED_API_KEY)
data = glf.fetch_all_data()
factor = glf.construct_factor(method='pca')

# Save outputs
glf.save_data()
glf.save_factor()
glf.plot_factor()
glf.correlation_analysis()
```

### Scheduling Automatic Updates (Windows Task Scheduler)

To update your liquidity factor automatically:

1. Create a batch file `run_liquidity.bat`:
```batch
@echo off
cd C:\Users\st_ha\OneDrive\Documents\github\automated_trading_test_environment.jl\src\LiquidityAPI
python global_liquidity_factor.py
```

2. Open Task Scheduler (`taskschd.msc`)
3. Create Basic Task:
   - Name: "Update Liquidity Factor"
   - Trigger: Weekly (or your preference)
   - Action: Start a program
   - Program: `C:\path\to\run_liquidity.bat`

### Viewing Charts on Windows

Charts are saved as PNG files. Double-click to open with:
- Windows Photos app
- Paint
- Your default image viewer

Or open programmatically:
```python
from PIL import Image
img = Image.open('results/global_liquidity_factor_chart.png')
img.show()
```

### PowerShell vs Command Prompt

Both work fine. PowerShell example:

```powershell
cd "C:\Users\st_ha\OneDrive\Documents\github\automated_trading_test_environment.jl\src\LiquidityAPI"
python .\global_liquidity_factor.py
```

### Next Steps

1. ✅ Verify the script runs successfully
2. ✅ Check the generated CSV and PNG files
3. ✅ Review the factor values
4. ✅ Try `advanced_usage.py` for backtesting examples
5. ✅ Integrate into your trading system

### Need Help?

- Check `README.md` for detailed documentation
- Check `PATCH_NOTES.md` for recent fixes
- Review `DATA_SOURCES.md` for indicator details

---

**All file paths are now compatible with Windows!**
