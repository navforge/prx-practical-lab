# 📚 Chapter 0: Introduction to polars

**Pre-requisite**  
Follow the instructions of `student_0_polars.ipynb` to learn the basics of `polars`.

**Objective**  
Let us practice coding with `polars`!  
Write a python module named `prx_tools.py` in the folder `src_polars`, which contains the following functions:
- `load_prx_file`
- `apply_iono_corr`
- `extract_columns`
- `plot_nsat_vs_elevation_mask`

📦**Deliverables**
- Updated powerpoint answer doc with plots and analysis

### 1. Function: `load_prx_file(file: str, constellation_filter: dict={"G": ["1C"]}, health_filter: bool=True)`
**Summary:**
- Loads GNSS observations from a `prx` file in a `pl.DataFrame`.
- Filters the data to keep only specific signals per constellation.
- Filters to keep only healthy satellites.

**Args:**
- `file: str`, contains the relative path to the gunzipped `prx` (with extension `.csv.gz`).
- `constellation_filter: dict`, a dictionary with constellation letters ("G","R","E","C") as keys and a list of rinex 3 signal identifier as values. The default value is {"G": ["1C"]}, for GPS L1 C/A.
- `health_filter: bool`, with a default value set to `True`.

**Returns:**
- a `pl.DataFrame` containing only the observations for the considered constellation and frequencies.

> 💡**Hint:**
> - loop over the constellations and filter the dataframe for every considered signal to create one filtered dataframe per constellation.
> - use `pl.concat([df1,...,df9])` to concatenate several `pl.DataFrame`.

**Test:**  
To verify your implementation, test your function using the following line in a terminal:
```
uv run pytest tests_polars/test_chapter0.py::test_load_prx_file
```

**Example Usage:**
```python
import src_polars.prx_tools as prx

# single constellation, single frequency
data_scsf = prx.load_prx_file(
  "data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz",
  constellation_filter={"G": ["1C"]},
  health_filter=True,
)
# dual constellation, single frequency
data_dcsf = prx.load_prx_file(
  "data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz",
  constellation_filter={"G": ["1C"], "E": ["1X"]},
  health_filter=True,
) 
# dual constellation, dual frequency
data_dcdf = prx.load_prx_file(
  "data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz",
  constellation_filter={"G": ["1C","2X"], "E": ["1X","5X"]},
  health_filter=True,
) 
```

### 2. Function: `apply_iono_corr(df_prx: pl.DataFrame)`
**Summary:**
- Compute a iono-corrected code pseudo-range observation, according to the following formula: $C_{corr}=C - I$, where $C$ is the code observation and $I$ is the ionospheric delay.

**Args:**
- `df_prx: pl.DataFrame`, a `pl.DataFrame` loaded from a `prx` file.

**Returns:**
- the same `pl.DataFrame` with an additional column named `"C_obs_corr_m"`.

**Test:**  
To verify your implementation, test your function using the following line in a terminal:
```
uv run pytest tests_polars/test_chapter0.py::test_iono
```

> 💡**Hint:**
> - Check the available information provided in the `prx` file by checking the column names, using `df_prx.columns`

### 3. Function: `extract_columns(df_corr: pl.DataFrame)`
**Summary:**
- return a `pl.DataFrame` with a subset of columns.

**Args:**
- `df_corr: pl.DataFrame`, a `pl.DataFrame` loaded from a `prx` file.

**Returns:**
- the same `pl.DataFrame` with only the following columns:
  - `"epoch"`,
  - `"sat_clock_offset_m"`,
  - `"sat_pos_x_m"`,
  - `"sat_pos_y_m"`,
  - `"sat_pos_z_m"`,
  - `"C_obs_m"`,
  - `"constellation"`,
  - `"prn"`

**Test:**  
To verify your implementation, test your function using the following line in a terminal:
```
uv run pytest tests_polars/test_chapter0.py::test_extract_col
```

### 4. Function: `plot_nsat_vs_elevation_mask(df_prx: pl.DataFrame, elevations_deg: list)`
**Summary:**
- Count the number of observations at each epoch for different elevation mask values.
- Save this in a figure `figures\nbsat_vs_elevation_mask.png`.

**Args:**
- `df_prx: pl.DataFrame`, a `pl.DataFrame` loaded from a `prx` file.
- `elevations_deg: list`, a list of elevation mask values in degrees.

**Returns:**
- None

> 💡**Hints:**
> - For each elevation, create a `DataFrame` by:
>   - filtering the DataFrame on satellite elevation angle (`.filter()`),
>   - grouping by epochs (`.group_by()`),
>   - aggregating each group to the group's length (`.agg()`).
> - Concatenate the resulting DataFrames (`pl.concat()`).
> - Plot using `df.plot.line()`, specifying `x`, `y` and `color`.
> - Save using `df.plot.line().save()`.

**Test:**  
You should test it yourself by plotting the number of GPS satellite broadcasting the C1C signal in the file `data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz` for an elevation mask [0,10,20,30,40,50].

