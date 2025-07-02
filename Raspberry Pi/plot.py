import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Column_Number = 1

df = pd.read_csv('/home/pi/adc_filtered_data.csv')

row = df.iloc[:,Column_Number]

plt.plot(row)
plt.show()