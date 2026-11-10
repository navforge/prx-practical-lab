# 📚 Chapter 3: Carrier-Based Positioning (RTK)
## 🎯 Objective
In this project, we will compute a **Real-Time Kinematics** solution, which consists in:
- using code and carrier observations,
- using the observations from a nearby stations to correct our own observations,
- using a Weighted Least Squares algorithm to compute the solution on a set of epochs with continuous phase tracking (batch algorithm),
- estimating the receiver position and clock, as well as the carrier phase ambiguity for each visible satellite over the set of selected epochs.

## ✅ Steps Overview
1. Implement carrier correction and batch processing functions.
2. Run `main_polars_rtk.py` for clean intervals.
3. Analyze RTK accuracy and compare with SPP/DGNSS.

## 1.🧑‍💻Coding Tasks
### 1.1. Function: `apply_differential_correction(df_rover: pl.DataFrame, df_base: pl.DataFrame, pos_base: np.ndarray)`
**Summary:**
- Modify your function `apply_differential_correction` of the `src_polars/gnss.py` module to apply the differential corrections to the rover observations.
- Add a new column `L_obs_corr_m` to the `df_rover` dataframe.

**Test:**
To verify your modification (and make sure you did not break the initial function), run the following lines in a terminal:
```bash
uv run pytest tests_polars/test_chapter3.py::test_differential_carrier_corrections
uv run pytest tests_polars/test_chapter2.py::test_differential_code_corrections
```

> 💡**Hint:**
> - The original carrier observations are expressed in units of cycles. To convert them to meters, you need to multiply by the carrier wavelength. You can use the value of the speed of light stored in `src_polars.constants.C_LIGHT_MPS` and the carrier frequency contained in the column `carrier_frequency_hz` of the dataframe.
> - Make sure that if a carrier observation is absent at one of the 2 receivers, it is dropped from the resulting dataframe, by means of `.filter()` and `.is_nan()`.

### 1.2. Function: `combine_lli(df_rover: pl.DataFrame, df_base: pl.DataFrame)`
One challenge of using carrier phase observations is the occurrence of **cycle slips**, occurring due to challenging tracking conditions.

GNSS receivers implement a cycle slip detection process, whose output is recorded in the `LLI` column (for **Loss of Lock Indicator**).

When combining observations from 2 different receivers, we have to take into account the cycle slips detected in both receivers.

In the module `src_polars/gnss.py`, write a function `combine_lli`.

**Summary:**
- The `LLI` value is `1` if a cycle slip has been detected, `0` if not, and `None` if the carrier observation is absent.
- The combination result of two `LLI` values is therefore:
  - `None` if at least one of the `LLI` is `None`,
  - `LLI_rover OR LLI_base` else.

**Args:**
- `df_rover: pl.DataFrame`, the dataframe loaded from a `prx` file for the base receiver.
- `df_base: pl.DataFrame`, the dataframe loaded from a `prx` file for the rover receiver

**Returns:**
- `df_rover` with the `LLI` column updated with the combination logic.

**tests_polars:**  
```bash
uv run pytest tests_polars/test_chapter1.py::test_combine_lli
```

> 💡**Hints:**
> - Similarly to the `apply_differential_correction`, you should perform the `LLI` combination after joining the 2 dataframes on a common index based on the columns `["epoch", "constellation", "prn", "rnx_obs_identifier"]`.
> - To apply the logical `OR` operation, use `or_()`, and make sure each operands is a boolean, by casting the variable to boolean type with `.cast(pl.Boolean)`.
> - To apply a condition with an `Expression`, you can use `pl.when(condition).then(result_1).otherwise(result_2)`.

### 1.3. Find epochs intervals without loss of lock
We will compute the RTK solution over a batch of epochs where no cycle slips have occurred. To do so, we need to determine sequence of epochs where there are no cycle slips, and choose the longest ones.

This is performed by the function `find_clean_intervals` of the module `src_polars/helpers.py`.

Example of function call:
```python
import src_polars.helpers as helpers

# determine intervals without LLI
clean_intervals = helpers.find_clean_intervals(df_rover, 5)
# select the longest interval
df_clean = df_rover.filter(
  (pl.col("epoch") >= clean_intervals.item(0,"start"))
  & (pl.col("epoch") <= clean_intervals.item(0,"end"))
)
```

### 1.4. Function: `obs_model_corrected_carrier(df, rx_pos, rx_clk=0, amb=None)`
**Summary:**
- The corrected carrier observations is: $L_{cor}^s = \lVert \mathbf{r}_{rx} - \mathbf{r}^{sat}\rVert + {dt}_{rx}+N^s$

**Args:**
- `df: pl.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `rx_pos: np.ndarray`,  the receiver position in meters, expressed in the ECEF frame.
- `rx_clk: float=0`, the receiver clock bias in meters. A default value of `0` should be used.
- `amb: np.array=None`, the vector of carrier ambiguities. A default value of `None` should be used.

**Returns:**
- an `np.ndarray` of shape `(n_row,)` containing the predicted corrected carrier observation values.

**Test:**
```bash
uv run pytest tests_polars/test_chapter2.py::test_corrected_carrier_model
```

> 💡**Hint:**
> - The default value of the ambiguity vector should be a vector of zeros. However, since the length of this vector depends on the length of `df`, we cannot put directly it as a default value. Consequently, the function shall start by `if amb is None: amb=...` 


### 1.4. Compute the Jacobian over a batch of epochs
Remember that we want to estimate the solution over a batch of epochs. The state vector is composed of
- the position of the rover receiver $r_k$ at each epoch $k$,
- the receiver clock bias $dt_k$ at each epoch,
- the carrier phase ambiguity $n^{s}$ for each satellite $s$, assumed constant over the interval.

Assuming $N$ epochs and $M$ visible satellites, let us agree that the states are arranged according to the following convention:
$$\mathbf{x} = [r_1, dt_1, ..., r_N, dt_N, n^{1}, ..., n^{M}]^T$$

Then, let us assume an order for the observation vector:
$$\mathbf{y} = [\mathbf{C}_1, \mathbf{L}_1, ... , \mathbf{C}_N, \mathbf{L}_N]^T$$
, where
- $\mathbf{C}_k$ is the corrected code observation vector at epoch $k$
- $\mathbf{L}_k$ is the corrected carrier observation vector at epoch $k$

In the module `src_polars/gnss.py`, write the functions `jacobian_code_batch` and `jacobian_carrier_batch`.

**Functions:**`jacobian_code_batch(df, rx_pos, idx_epoch, n_epoch, prns)` and `jacobian_carrier_batch(df, rx_pos, idx_epoch, n_epoch, prns)`

**Summary:**
- Each function will compute the Jacobian matrix of the corresponding observation vector at epoch $k$ ($\mathbf{C}_k$ or $\mathbf{L}_k$) wrt to the whole observation vector $\mathbf{x}$.

**Args:**
- `df: pl.DataFrame`, a dataframe loaded from a `prx` file for the rover receiver.
- `rx_pos: np.ndarray`,  the receiver position in meters, expressed in the ECEF frame.
- `idx_epoch: int`, the index of the current epoch.
- `n_epoch: int`, the total number of epochs.
- `prns: list`, the list of visible satellites over the considered interval.

**Returns:**
- an `np.ndarray` of shape `(m, 4 * n_epoch + len(prns))` representing the Jacobian matrix, `m` being the number of observations and `4 * n_epoch + len(prns)` the length of the stacked state vector.

**Test:**  
```bash
uv run pytest tests_polars/test_chapter2.py::test_jacobian_code_batch
uv run pytest tests_polars/test_chapter2.py::test_jacobian_carrier_batch
```
> 💡**Hint:**
> - Create the different blocks of the jacobian wrt to each group of states (xyzt at each epoch and ambiguities) and then concatenate them using `np.concat([arr1, ..., arrN], axis=1)` (horizontal stack). 

**Example Usage:**
```python
import src_polars.gnss as gnss
from src_polars.constants import TLSE_2024001_ECEF

df_clean = ...  # Consider dataframe over a interval of clean epochs
n_epoch = len(df_clean.get_column("epoch").unique())
prns = df_clean.get_column("prn").unique().to_list()
# for the first epoch
idx_epoch = 0
jac_code = gnss.jacobian_code_batch(df_clean, TLSE_2024001_ECEF, idx_epoch, n_epoch, prns)
jac_carrier = gnss.jacobian_carrier_batch(df_clean, TLSE_2024001_ECEF, idx_epoch, n_epoch, prns)
```

### 1.5. Compute the carrier observation covariance matrix
For the carrier observation covariance matrix at an epoch $k$, we can consider the same covariance matrix as for the code observations, with a scaling factor of $(1/100)^2$.

This is due to the much higher accuracy of the carrier observations, compared to code observations.

### 1.6. Script: main_polars_rtk.py
To apply the WLS considering all observations and states contained in the clean interval, we have to compute the different vectors/matrices for each epoch of the interval, and then correctly stack them.

Create a script `main_polars_rtk.py` at the root of the repository to compute the RTK solution. In this script, write a function that will compute the RTK batch solution for a clean interval, using:
- `data/TLSE00FRA_R_20240010000_01D_30S_MO.csv.gz` for the rover receiver data,
- `data/TLSG00FRA_R_20240010000_01D_30S_MO.csv.gz` for the base receiver data,.

The RTK solution shall be recorded in a `results` dataframe containing the columns `"epoch","pos_x","pos_y","pos_z","clk_b"`.

The pseudo-code for such function is:
```
1.  Load rover and base dataframes
2.  Apply differential corrections and combine LLI
3.  Find clean intervals
4.  For each interval in intervals:
5.      For each epoch in interval:
6.          Retrieve observations
7.          Compute predicted observations
8.          Compute jacobian
9.          Compute covariance matrix
10.     Stack obs, predicted obs, jacobian matrices, covariance matrices
11.     Solve WLS
```

> 💡Hint:
> - When stacking (predicted) observation vectors and Jacobian matrices, use `np.concat(..., axis=0)` (vertical stack).
> - When stacking the observation covariance matrices, you can use `scipy.linalg.block_diag(*[arr1, arr2, ...])` (diagonal stack). This requires to import `scipy`.

### 1.7. Compute the RTK batch solution for several clean intervals
Modify your script `main_polars_rtk.py` to process the 5 longest clean intervals, in order to have a significant number of estimation epochs (at least 1,000 epochs).

## 2.🔍Analysis Tasks
Complete the powerpoint document by pasting figures and writing analyses.

Use the functions `compute_enu_pos_error`, `plot_enu_error` and `plot_enu_error_cdf` from the module `src_polars/helpers.py` to compute, plot and characterize the position error in the East/North/Up frame.  
Use the function `helpers.analyze_results` to display various statistics on a `results` dataframe, and to compute the accuracy score for this solution.

❓ **Questions to Answer**: Compare the RTK solutions to the other. How would you characterize the improvement?

## 3.🏅Compete in the positioning leaderboard
Document your best solution in your report. It will be evaluated and ranked against the other student teams' solutions.

## 4. 💰Treasure hunt
Once you have a working RTK script, apply it to your file in the folder `data/treasure_hunt`.
If the algorithm is working you should see a clue about the treasure hunt...
Can you find the treasure stash?