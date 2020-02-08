import pandas as pd
import matplotlib.pyplot as plt

filename = "src/dataset/training.csv"

# Converting csv to dataframe
df = pd.read_csv(filename, sep=',')
plt.close('all')

# Feature x Feature Plots

df.plot(kind="scatter", x=1, y=2, color='red')
plt.show()
df.plot(kind="scatter", x=2, y=3, color='red')
plt.show()
df.plot(kind="scatter", x=6, y=7, color='red')
plt.show()
df.plot(kind="scatter", x=6, y=9, color='red')
plt.show()
df.plot(kind="scatter", x=14, y=18, color='red')
plt.show()
df.plot(kind="scatter", x=15, y=19, color='red')
plt.show()
df.plot(kind="scatter", x=16, y=20, color='red')
plt.show()
df.plot(kind="scatter", x=20, y=21, color='red')
plt.show()
df.plot(kind="scatter", x=21, y=22, color='red')
plt.show()
df.plot(kind="scatter", x=23, y=24, color='red')
plt.show()
df.plot(kind="scatter", x=25, y=26, color='red')
plt.show()
df.plot(kind="scatter", x=27, y=28, color='red')
plt.show()
df.plot(kind="scatter", x=29, y=30, color='red')
plt.show()

# Missing Values

for col in df:
  print(df[col].value_counts(-999.000))
