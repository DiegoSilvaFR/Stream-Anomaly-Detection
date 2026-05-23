from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import deque


EPS = 1e-8


def _safe_array(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return x


def _safe_std(x):
    x = _safe_array(x)
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _safe_skew(x):
    x = _safe_array(x)
    if len(x) <= 2:
        return 0.0

    mu = np.mean(x)
    sd = np.std(x) + EPS
    z = (x - mu) / sd
    return float(np.mean(z ** 3))


def _safe_kurtosis(x):
    x = _safe_array(x)
    if len(x) <= 3:
        return 0.0

    mu = np.mean(x)
    sd = np.std(x) + EPS
    z = (x - mu) / sd
    return float(np.mean(z ** 4) - 3.0)


def _autocorr(x, lag=1):
    x = _safe_array(x)
    if len(x) <= lag + 1:
        return 0.0

    x1 = x[:-lag]
    x2 = x[lag:]

    s1 = np.std(x1)
    s2 = np.std(x2)

    if s1 < EPS or s2 < EPS:
        return 0.0

    return float(np.corrcoef(x1, x2)[0, 1])


def _slope_features(x):
    x = _safe_array(x)
    n = len(x)

    if n <= 2:
        return 0.0, 0.0

    t = np.arange(n, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + EPS)

    y = (x - x.mean()) / (x.std() + EPS)

    slope = float(np.polyfit(t, y, deg=1)[0])

    y_hat = slope * t
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + EPS
    r2 = float(1.0 - ss_res / ss_tot)

    return slope, r2


def _sign_change_rate(x):
    x = _safe_array(x)
    if len(x) <= 2:
        return 0.0

    dx = np.diff(x)
    signs = np.sign(dx)
    signs = signs[signs != 0]

    if len(signs) <= 1:
        return 0.0

    return float(np.mean(signs[1:] != signs[:-1]))


def _turning_point_rate(x):
    x = _safe_array(x)
    if len(x) <= 2:
        return 0.0

    dx1 = np.diff(x[:-1])
    dx2 = np.diff(x[1:])

    turning = (dx1 * dx2) < 0
    return float(np.mean(turning))


def _zero_crossing_rate(x):
    x = _safe_array(x)
    if len(x) <= 1:
        return 0.0

    x = x - np.mean(x)
    signs = np.sign(x)
    signs = signs[signs != 0]

    if len(signs) <= 1:
        return 0.0

    return float(np.mean(signs[1:] != signs[:-1]))


class StreamingBreakFeatureExtractor:
    """
    Causal feature extractor for the Crunch structural break task.

    At each online point:
        1. update internal rolling windows
        2. compute features using only historical data + online points seen so far
        3. return one feature dictionary

    This can be used in both training and inference.
    """

    def __init__(
        self,
        x_historical,
        windows=(8, 16, 32, 64, 128),
        cusum_drift=0.25,
        ewma_alpha=0.05,
    ):
        self.x_hist = _safe_array(x_historical)
        self.windows = list(windows)

        self.hist_mean = float(np.mean(self.x_hist))
        self.hist_std = float(np.std(self.x_hist, ddof=1)) if len(self.x_hist) > 1 else 0.0
        self.hist_std = max(self.hist_std, EPS)

        self.hist_median = float(np.median(self.x_hist))
        self.hist_iqr = float(np.quantile(self.x_hist, 0.75) - np.quantile(self.x_hist, 0.25))
        self.hist_iqr = max(self.hist_iqr, EPS)

        self.hist_mad = float(np.median(np.abs(self.x_hist - self.hist_median)))
        self.hist_mad = max(self.hist_mad, EPS)

        self.hist_quantiles = {
            q: float(np.quantile(self.x_hist, q))
            for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        }

        self.buffers = {
            W: deque(self.x_hist[-W:], maxlen=W)
            for W in self.windows
        }

        self.t = -1
        self.n_online_seen = 0

        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.cusum_drift = cusum_drift

        self.ewma_z = 0.0
        self.ewma_abs_z = 0.0
        self.ewma_alpha = ewma_alpha

        self.running_max_abs_z = 0.0
        self.running_mean_abs_z = 0.0

        self.consecutive_abs_z_gt_1 = 0
        self.consecutive_abs_z_gt_2 = 0
        self.consecutive_abs_z_gt_3 = 0

    def update(self, x_t):
        """
        Add one online point and return feature dict for the current time step.
        """

        x_t = float(x_t)
        self.t += 1
        self.n_online_seen += 1

        for W in self.windows:
            self.buffers[W].append(x_t)

        z_t = (x_t - self.hist_mean) / self.hist_std
        abs_z_t = abs(z_t)

        # Cumulative evidence features
        self.cusum_pos = max(0.0, self.cusum_pos + z_t - self.cusum_drift)
        self.cusum_neg = max(0.0, self.cusum_neg - z_t - self.cusum_drift)

        self.ewma_z = (1.0 - self.ewma_alpha) * self.ewma_z + self.ewma_alpha * z_t
        self.ewma_abs_z = (1.0 - self.ewma_alpha) * self.ewma_abs_z + self.ewma_alpha * abs_z_t

        self.running_max_abs_z = max(self.running_max_abs_z, abs_z_t)
        self.running_mean_abs_z += (abs_z_t - self.running_mean_abs_z) / self.n_online_seen

        if abs_z_t > 1:
            self.consecutive_abs_z_gt_1 += 1
        else:
            self.consecutive_abs_z_gt_1 = 0

        if abs_z_t > 2:
            self.consecutive_abs_z_gt_2 += 1
        else:
            self.consecutive_abs_z_gt_2 = 0

        if abs_z_t > 3:
            self.consecutive_abs_z_gt_3 += 1
        else:
            self.consecutive_abs_z_gt_3 = 0

        feats = {
            "t": self.t,
            "n_online_seen": self.n_online_seen,

            "last_value": x_t,
            "last_z": z_t,
            "last_abs_z": abs_z_t,

            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "cusum_max": max(self.cusum_pos, self.cusum_neg),
            "cusum_diff": self.cusum_pos - self.cusum_neg,

            "ewma_z": self.ewma_z,
            "ewma_abs_z": self.ewma_abs_z,
            "running_max_abs_z": self.running_max_abs_z,
            "running_mean_abs_z": self.running_mean_abs_z,

            "consecutive_abs_z_gt_1": self.consecutive_abs_z_gt_1,
            "consecutive_abs_z_gt_2": self.consecutive_abs_z_gt_2,
            "consecutive_abs_z_gt_3": self.consecutive_abs_z_gt_3,
        }

        for W in self.windows:
            w = np.asarray(self.buffers[W], dtype=np.float64)

            feats.update(self._window_features(w, W))

        # Short-vs-long agreement features
        if len(self.windows) >= 2:
            Ws = sorted(self.windows)
            W_short = Ws[0]
            W_long = Ws[-1]

            feats[f"mean_diff_w{W_short}_w{W_long}"] = (
                feats[f"mean_w{W_short}"] - feats[f"mean_w{W_long}"]
            )
            feats[f"std_ratio_w{W_short}_w{W_long}"] = (
                feats[f"std_w{W_short}"] / (feats[f"std_w{W_long}"] + EPS)
            )
            feats[f"abs_z_mean_diff_w{W_short}_w{W_long}"] = (
                feats[f"abs_z_mean_w{W_short}"] - feats[f"abs_z_mean_w{W_long}"]
            )

        return feats

    def _window_features(self, w, W):
        mean_w = float(np.mean(w))
        std_w = _safe_std(w)
        median_w = float(np.median(w))

        q01 = float(np.quantile(w, 0.01))
        q05 = float(np.quantile(w, 0.05))
        q10 = float(np.quantile(w, 0.10))
        q25 = float(np.quantile(w, 0.25))
        q50 = float(np.quantile(w, 0.50))
        q75 = float(np.quantile(w, 0.75))
        q90 = float(np.quantile(w, 0.90))
        q95 = float(np.quantile(w, 0.95))
        q99 = float(np.quantile(w, 0.99))

        iqr_w = q75 - q25
        mad_w = float(np.median(np.abs(w - median_w)))

        z_w = (w - self.hist_mean) / self.hist_std
        abs_z_w = np.abs(z_w)

        slope, slope_r2 = _slope_features(w)

        diffs = np.diff(w)
        if len(diffs) == 0:
            mean_abs_diff = 0.0
            std_abs_diff = 0.0
            max_abs_diff = 0.0
            mean_sq_diff = 0.0
        else:
            abs_diffs = np.abs(diffs)
            mean_abs_diff = float(np.mean(abs_diffs))
            std_abs_diff = float(np.std(abs_diffs))
            max_abs_diff = float(np.max(abs_diffs))
            mean_sq_diff = float(np.mean(diffs ** 2))

        quantile_diffs = np.array([
            q01 - self.hist_quantiles[0.01],
            q05 - self.hist_quantiles[0.05],
            q10 - self.hist_quantiles[0.10],
            q25 - self.hist_quantiles[0.25],
            q50 - self.hist_quantiles[0.50],
            q75 - self.hist_quantiles[0.75],
            q90 - self.hist_quantiles[0.90],
            q95 - self.hist_quantiles[0.95],
            q99 - self.hist_quantiles[0.99],
        ])

        return {
            f"mean_w{W}": mean_w,
            f"std_w{W}": std_w,
            f"median_w{W}": median_w,
            f"min_w{W}": float(np.min(w)),
            f"max_w{W}": float(np.max(w)),
            f"range_w{W}": float(np.max(w) - np.min(w)),
            f"iqr_w{W}": float(iqr_w),
            f"mad_w{W}": float(mad_w),

            f"z_mean_w{W}": (mean_w - self.hist_mean) / self.hist_std,
            f"abs_z_mean_w{W}": abs((mean_w - self.hist_mean) / self.hist_std),
            f"z_median_w{W}": (median_w - self.hist_median) / self.hist_std,
            f"abs_z_median_w{W}": abs((median_w - self.hist_median) / self.hist_std),

            f"log_std_ratio_w{W}": float(np.log((std_w + EPS) / self.hist_std)),
            f"log_iqr_ratio_w{W}": float(np.log((iqr_w + EPS) / self.hist_iqr)),
            f"log_mad_ratio_w{W}": float(np.log((mad_w + EPS) / self.hist_mad)),

            f"max_abs_z_w{W}": float(np.max(abs_z_w)),
            f"mean_abs_z_w{W}": float(np.mean(abs_z_w)),
            f"q95_abs_z_w{W}": float(np.quantile(abs_z_w, 0.95)),
            f"frac_abs_z_gt_1_w{W}": float(np.mean(abs_z_w > 1.0)),
            f"frac_abs_z_gt_2_w{W}": float(np.mean(abs_z_w > 2.0)),
            f"frac_abs_z_gt_3_w{W}": float(np.mean(abs_z_w > 3.0)),

            f"q01_diff_w{W}": q01 - self.hist_quantiles[0.01],
            f"q05_diff_w{W}": q05 - self.hist_quantiles[0.05],
            f"q10_diff_w{W}": q10 - self.hist_quantiles[0.10],
            f"q25_diff_w{W}": q25 - self.hist_quantiles[0.25],
            f"q50_diff_w{W}": q50 - self.hist_quantiles[0.50],
            f"q75_diff_w{W}": q75 - self.hist_quantiles[0.75],
            f"q90_diff_w{W}": q90 - self.hist_quantiles[0.90],
            f"q95_diff_w{W}": q95 - self.hist_quantiles[0.95],
            f"q99_diff_w{W}": q99 - self.hist_quantiles[0.99],
            f"mean_abs_quantile_diff_w{W}": float(np.mean(np.abs(quantile_diffs))),
            f"max_abs_quantile_diff_w{W}": float(np.max(np.abs(quantile_diffs))),

            f"slope_w{W}": slope,
            f"abs_slope_w{W}": abs(slope),
            f"slope_r2_w{W}": slope_r2,

            f"mean_abs_diff_w{W}": mean_abs_diff,
            f"std_abs_diff_w{W}": std_abs_diff,
            f"max_abs_diff_w{W}": max_abs_diff,
            f"mean_sq_diff_w{W}": mean_sq_diff,

            f"autocorr_lag1_w{W}": _autocorr(w, lag=1),
            f"autocorr_lag2_w{W}": _autocorr(w, lag=2),
            f"autocorr_lag5_w{W}": _autocorr(w, lag=5),

            f"sign_change_rate_w{W}": _sign_change_rate(w),
            f"turning_point_rate_w{W}": _turning_point_rate(w),
            f"zero_crossing_rate_w{W}": _zero_crossing_rate(w),

            f"skew_w{W}": _safe_skew(w),
            f"kurtosis_w{W}": _safe_kurtosis(w),
        }


def extract_features_for_one_series(
    series_id,
    x_historical,
    x_online,
    tau=None,
    windows=(8, 16, 32, 64, 128),
):
    """
    Create one feature row per online point for one time series.

    If tau is given:
        target = 1 if t >= tau else 0

    If tau is None:
        target = 0 for all t
    """

    extractor = StreamingBreakFeatureExtractor(
        x_historical=x_historical,
        windows=windows,
    )

    rows = []

    for t, x_t in enumerate(x_online):
        row = extractor.update(x_t)

        row["series_id"] = series_id

        if tau is not None:
            row["target"] = int(t >= tau)
        else:
            row["target"] = 0

        rows.append(row)

    return pd.DataFrame(rows)

def one_time_series_extraction(item: tuple[int, list, list, int], windows: tuple) -> pd.DataFrame:
    
    series_id, x_historical, x_online, tau = item

    return extract_features_for_one_series(
        series_id=series_id,
        x_historical=x_historical,
        x_online=x_online,
        tau=tau,
        windows=windows,
    )

def extract_features_for_dataset(
    data: list[tuple[int, list, list, int]],
    windows=(8, 16, 32, 64, 128)
):
    """
    Build feature dataframe for many series.

    Expected training format:
        data = iterable of (series_id, x_historical, x_online, tau)

    Expected test/inference format:
        data = iterable of (series_id, x_historical, x_online)

    Returns:
        DataFrame with one row per online point.
    """


    dfs = Parallel(n_jobs=-1)(delayed(one_time_series_extraction)(item, windows)
                                for item in tqdm(data, desc="Training Feat. Extraction"))

    return pd.concat(dfs, axis=0, ignore_index=True)
