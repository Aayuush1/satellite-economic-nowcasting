import ee 
import torch
import pandas as pd
import numpy as np

(f'Earth Engine: {ee.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')

# Test GEE authentication
try:
    ee.Initialize()
    print('GEE: Authenticated successfully')
except:
    print('GEE: Run earthengine authenticate in terminal')