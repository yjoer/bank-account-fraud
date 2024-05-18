# %%
import pandas as pd
from ydata_profiling import ProfileReport

# %%
df = pd.read_csv("data/Base.csv")

# %%
if True:
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("report_1.html")

# %%
