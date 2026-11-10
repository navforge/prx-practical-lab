# 📚 Chapter 2: Differential GNSS solution
**Objective**  
In this project, we will compute the **Differential GNSS** solution, which consists in:
- using code observations
- using the observations from a nearby station to correct our own observations
- using a Weighted Least Squares algorithm to compute the solution at each epoch
- estimating the receiver position and clock

**Steps Overview**
1. Implement core functions for DGNSS.
2. Run `main_polars_dgnss.py` to compute position estimates.
3. Analyze ENU errors.
4. Compare SPP and DGNSS solutions.

📦**Deliverables**
- Complete python module `src_polars/gnss.py`
- Python script: `main_polars_dgnss.py`
- Updated powerpoint answer doc with plots and analysis

## 1.🧑‍💻Coding Tasks
All functions shall be coded in the `src_polars/gnss.py` module.

### 1.1.Function: `apply_differential_corrections( df_rover: pl.DataFrame, df_base: pl.DataFrame, pos_base: np.ndarray)`
**Summary:**
- Considering a *base* receiver with known position and a *rover* receiver, this function applies differential corrections using the following formula:
$$C_{cor}^s=C_{rov}^s - (C_{bas}^s - geo_{bas}^s)$$
  -  where $geo_{base}^s$ is the geometric distance between satellite $s$ and the base station.
- Ensure differential corrections are applied only to matching observations (same constellation, PRN, signal type, and epoch).

**Args:**
- `df_rover: pl.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `df_base: pl.DataFrame`, a dataframe loaded from a `prx` file for the base receiver.
- `pos_base: np.ndarray`, the base receiver precise position.

**Returns:**
- The same dataframe as `df_rover` with an additional column `C_obs_corr_m` containing the differentially-corrected code observations.

**Tests:**  
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests_polars/test_chapter2.py::test_differential_code_corrections
```

Example of function call:

```python
import src_pandas.prx_tools as prx
import src_pandas.gnss as gnss
from src_pandas.constants import TLSG_2024001_ECEF

df_rover = prx.load_prx_file("data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz")
df_base = prx.load_prx_file("data/TLSG00FRA_R_20240010000_01D_30S_MO.csv.gz")
df_rover = gnss.apply_differential_corrections(df_rover, df_base, TLSG_2024001_ECEF)
```

> 💡**Hints:**
> - Add the geometric distance to `df_base`.
> - Join the 2 dataframes with `.join()` ([doc](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html)), by considering the columns `["epoch", "constellation", "prn", "rnx_obs_identifier"]` as common index.
> - Add the differentially-corrected pseudoranges as a new column.
> - Return the dataframe with only the desired columns.

### 1.2. Function: `obs_model_corrected_code(df: pl.DataFrame, rx_pos: np.ndarray, rx_clk: float=0)`
We have to define a corrected code observation model, which can be simplified to the terms depending only on the unknown states. All other terms (sat clock, satellite hardware biases, atmospheric delays) have been corrected.

**Summary:**
- The corrected code observation model is $C_{cor}^{sat}=\lVert \mathbf{r}_{rx} - \mathbf{r}^{sat}\rVert + {dt}_{rx}$

**Args:**
- `df: pl.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `rx_pos: np.ndarray`,  the receiver position in meters, expressed in the ECEF frame.
- `rx_clk: float=0`, the receiver clock bias in meters. A default value of 0 should be used.

**Returns:**
- an `np.ndarray` containing the predicted corrected code observation values.

**Test:**  
To verify your implementation, test your function using the following line in a terminal:
```bash
uv run pytest tests_polars/test_chapter2.py::test_corrected_code_model
```

### 1.3. Jacobian matrix of the corrected code observation model
If you look at the corrected code observation model, you will realize that it has the same Jacobian matrix as the uncorrected code observation model.

Therefore, we can re-use the existing function `jacobian_code()`!

### 1.4. Covariance matrix of the observations
We will use the same models as the one already coded in you function `obs_covariance_mat()`.

### 1.5. Weighted Least Squares
The same estimation algorithm as for the SPP solution can be used. Just make sure that you correctly define the observation vector, observation model, jacobian matrix and observation covariance matrix.

### 1.6. Script: `main_polars_dgnss.py`
Create a script at the repository root named `main_polars_dgnss.py` and write the code to compute the DGNSS solution, using:
- `data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz` for the rover receiver data,
- `data/TLSG00FRA_R_20240010000_01D_30S_MO.csv.gz` for the base receiver data,.

You may re-use the same script structure as `main_spp.py`, except that you will now use differentially corrected code observations.

The DGNSS solution shall be recorded in a `results` dataframe containing the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`.

## 2.🔎Analysis Tasks
Complete the powerpoint document by pasting figures and writing analyses.

Use the functions `compute_enu_pos_error`, `plot_enu_error` and `plot_enu_error_cdf` from the module `src_polars/helpers.py` to compute, plot and characterize the position error in the East/North/Up frame.
Use the function `helpers.analyze_results` to display various statistics on a `results` dataframe, and to compute the accuracy score for this solution.

❓ **Question to Answer:** Compare the DGNSS and the SPP solutions? How would you characterize the improvement?

## 3.🏅Compete in the positioning leaderboard
Document your best solution in your report. It will be evaluated and ranked against the other student teams' solutions.

## 4. 💰Treasure hunt
Once you have a working DGNSS script, apply it to your file in the folder `data/treasure_hunt`.
If the algorithm is working you should see a clue about the treasure hunt...