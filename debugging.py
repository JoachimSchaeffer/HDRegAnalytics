import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import random
from sklearn.metrics import mean_squared_error
import jax.numpy as jnp
from src.featlin import Featlin
from src.featlin import jax_moment
from src.basis import BasicsData

# Initialize the random seed to ensure reproducibility of the results in the paper
random.seed(10)
plt.style.use("./styles/plots.mplstyle")

save_plots = True
save_path = "results/Linearization/"


# Colorblind safe palette from: https://gka.github.io/palettes/#/26|s|00429d,96ffea,ffffe0|ffffe0,ff005e,93003a|1|1
# IBM Colors: https://www.ibm.com/design/language/color/
# https://lospec.com/palette-list/ibm-color-blind-safe

colors = [
    "#332bb3",
    "#4a31b5",
    "#5d37b6",
    "#6d3db7",
    "#7c43b7",
    "#8a49b6",
    "#964fb5",
    "#a256b3",
    "#ad5db1",
    "#b764b0",
    "#c16cae",
    "#ca75ad",
    "#d27eac",
    "#d989ab",
    "#e094aa",
    "#e7a1ab",
    "#ecafac",
    "#f0beae",
    "#f4cfb0",
    "#f6e1b4",
]
colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
cmap_ = clr.LinearSegmentedColormap.from_list("Blue-light cb-safe", colors, N=256)
cmap = clr.LinearSegmentedColormap.from_list("Blue-light cb-IBM", colors_IBM[:-1], N=256)
color_list = [colors_IBM[0], colors_IBM[2], colors_IBM[3], colors_IBM[4], colors_IBM[5]]
marker_list = ["s", "o", "D", "P"]


# Load the LFP Dataset
lfp_df = pd.read_csv("./data/lfp_slim.csv", index_col=0)

X_lfp = np.array(lfp_df.iloc[:, 0:1000])
X_lfp = X_lfp[:, ::-1]
y_lfp_true = np.array(lfp_df.iloc[:, 1000])
x_lfp = np.linspace(2.0, 3.5, 1000)

X_lfp_train = np.array(X_lfp[lfp_df.iloc[:, 1002] == 0, :])
y_lfp_train_true = np.array(y_lfp_true[lfp_df.iloc[:, 1002] == 0])
X_lfp_test = np.array(X_lfp[lfp_df.iloc[:, 1002] == 1, :])
y_lfp_test_true = np.array(y_lfp_true[lfp_df.iloc[:, 1002] == 1])
X_lfp_test2 = np.array(X_lfp[lfp_df.iloc[:, 1002] == 2, :])
y_lfp_test2_true = np.array(y_lfp_true[lfp_df.iloc[:, 1002] == 2])

labels_lfp = {
    "xdata_label": "Voltage (V)",
    "ydata_label": r"$\Delta \mathbf{Q}_{100\mathrm{-}10}$ (Ah)",
    "row_label": "Battery number",
}


# Remove outlier
id_outlier = np.where(
    np.mean(X_lfp_train, axis=1) == np.min(np.mean(X_lfp_train, axis=1))
)
X_lfp_train = np.delete(X_lfp_train, id_outlier, axis=0)
y_lfp_train_true = np.delete(y_lfp_train_true, id_outlier, axis=0)


def sinus_transformation(X: np.ndarray, a=0.07) -> np.ndarray:
    Y = jnp.sin((2 * jnp.pi / a) * X)
    if len(Y.shape) == 1:
        y = jnp.sum(Y)
    else:
        y = jnp.sum(Y, axis=1)
    return y


# True underlying relationship between the measurements and some quantity we ewould like to recover/predict from measurements

# JAX numpy wrapper target function  to allow for autodifferentiation
fun_targetj = [
    lambda x: jnp.mean(x),
    lambda x: sinus_transformation(x, a=0.06),
    lambda x: jnp.sum(x**2),
    lambda x: jnp.var(x),
    lambda x: jax_moment(x, 3) / ((jax_moment(x, 2)) ** (3 / 2)),
    lambda x: jax_moment(x, 4) / (jax_moment(x, 2) ** 2) - 3,
]

fun_target_names = [
    "Sum",
    "Sinus",
    "Sum of Squares",
    "Variance",
    "Skewness",
    "Kurtosis",
]

feat_fun_dict = {fun_target_names[i]: fun_targetj[i] for i in range(len(fun_targetj))}


# Generate BasicsData class objects
# Different examples, only the following are used in this notebook
# 'Sum' : Simple illustration with PLS
# 'Sinus': Using RR
# 'Sum of Squares': Using RR and std
# 'Variance': Using RR and std
# --> 6 Case studies

# Mean
lfp_mean = BasicsData(X=X_lfp_train, x=x_lfp, y=None)
lfp_mean = lfp_mean.construct_y_data(fun_targetj[0]).add_wgn(
    add_noise_X=False, add_noise_y=True, snr_y=50
)

# Sinus
lfp_sin = BasicsData(X=X_lfp_train, x=x_lfp, y=None)
lfp_sin = lfp_sin.construct_y_data(fun_targetj[1]).add_wgn(
    add_noise_X=False, add_noise_y=True, snr_y=50
)

# Sum Squares
lfp_sums = BasicsData(X=X_lfp_train, x=x_lfp, y=None)
lfp_sums = lfp_sums.construct_y_data(fun_targetj[2]).add_wgn(
    add_noise_X=False, add_noise_y=True, snr_y=50
)

# Variance
lfp_var = BasicsData(X=X_lfp_train, x=x_lfp, y=None)
lfp_var = lfp_var.construct_y_data(fun_targetj[3]).add_wgn(
    add_noise_X=False, add_noise_y=True, snr_y=50
)

# Skew
lfp_skew = BasicsData(X=X_lfp_train, x=x_lfp, y=None)
lfp_lfp_skewvar = lfp_skew.construct_y_data(fun_targetj[4]).add_wgn(
    add_noise_X=False, add_noise_y=True, snr_y=50
)

# Kurt
lfp_kurt = BasicsData(X=X_lfp_train, x=x_lfp, y=None)
lfp_kurt = lfp_kurt.construct_y_data(fun_targetj[5]).add_wgn(
    add_noise_X=False, add_noise_y=True, snr_y=50
)

# Log Cycle Life
lfp_lcl = BasicsData(X=X_lfp_train, x=x_lfp, y=np.log(y_lfp_train_true))

# Log Cycle Life
fig_props = {
    "save": save_plots,
    "ax0_xlabel": "Voltage (V)",
    "save_path": save_path + "LCL_RR_",
    "multiple_fig": False,
}

lfp_sums = Featlin(data_obj=lfp_sums, feat_funcs=feat_fun_dict)
fl_lfp_lcl = Featlin(data_obj=lfp_lcl, feat_funcs=feat_fun_dict)

lfp_sums = lfp_sums.analyze_feature(
    "Sum of Squares",
    opt_cv={"active": False, "model": []},
    opt_dist={"active": True, "model": ["RR"]},
    con_thres=1,
    opt_gamma_method="Xv",
)
fig, ax = lfp_sums.linearization_plot("Sum of Squares")
# fig.suptitle('Sinus Feature Nullspace Analysis, RR')
plt.tight_layout()
fig.savefig("figures/sos_nulls_rr.pdf", bbox_inches="tight")


lfp_sums = lfp_sums.analyze_feature(
    "Sum of Squares",
    con_thres=0.2,
    opt_gamma_method="Xv",
    opt_cv={"active": False, "model": []},
    opt_dist={"active": True, "model": ["PLS"]},
)

fig, ax = lfp_sums.linearization_plot("Sum of Squares")
plt.tight_layout()
fig.savefig("figures/sos_nulls_pls.pdf", bbox_inches="tight")


for key in fl_lfp_lcl.nullspace_dict.keys():
    print(key)
    keys_model = [key for key in fl_lfp_lcl.nullspace_dict[key].keys()]

    # Drop 'lfun' entry from keys_model list
    keys_model = [key for key in keys_model if key != "lfun"]

    key_alpha = "w_alpha"
    key_beta = "w_beta"
    for key_model in keys_model:
        nulls_ = fl_lfp_lcl.nullspace_dict[key][key_model]["nulls"]
        X = nulls_.data.X_

        if 0:
            y_ = nulls_.data.y_
            print(
                100
                * mean_squared_error(
                    y_,
                    X @ (nulls_.nullsp["w_alpha"] + nulls_.nullsp["v_"][-1, :]),
                    squared=False,
                )
                / (np.max(y_) - np.min(y_))
            )
            print(
                100
                * mean_squared_error(y_, X @ (nulls_.nullsp["w_beta"]), squared=False)
                / (np.max(y_) - np.min(y_))
            )
            print(
                100
                * mean_squared_error(y_, X @ (nulls_.nullsp["w_alpha"]), squared=False)
                / (np.max(y_) - np.min(y_))
            )

        # constrain NRMSE
        y_ = nulls_.data.y_

        nrmse_reg = (
            100
            * mean_squared_error(y_, X @ (nulls_.nullsp[key_alpha]), squared=False)
            / (np.max(y_) - np.min(y_))
        )
        nrmse_nulls = (
            100
            * mean_squared_error(
                y_,
                X @ (nulls_.nullsp[key_alpha] + nulls_.nullsp["v_"][-1, :].reshape(-1)),
                squared=False,
            )
            / (np.max(y_) - np.min(y_))
        )
        val = np.abs(nrmse_reg - nrmse_nulls)

        print(f"Constraint NRMSE: {val}")

        # Make predicrtions with 'w_beta'
        pred_lin = X @ (nulls_.nullsp["w_beta"])
        pred_model = X @ (nulls_.nullsp["w_alpha"])
        print(
            100
            * mean_squared_error(
                X @ (nulls_.nullsp["w_alpha"]),
                X @ (nulls_.nullsp["w_alpha"] + nulls_.nullsp["v_"][-1, :]),
                squared=False,
            )
            / (np.max(pred_model) - np.min(pred_model))
        )
        print(
            100
            * np.sqrt(np.average((X @ (nulls_.nullsp["v_"][-1, :])) ** 2))
            / (np.max(pred_model) - np.min(pred_model))
        )


# 'Sum of Squares': Using RR and std
lfp_sums_gt = Featlin(data_obj=lfp_sums, feat_funcs=feat_fun_dict)

fig_props = {
    "save": save_plots,
    "ax0_xlabel": "Voltage (V)",
    "save_path": save_path + "sums_gt_RR_",
    "multiple_fig": False,
}

# 'Sum of Squares': Using RR
lfp_sums_gt = Featlin(data_obj=lfp_sums, feat_funcs=feat_fun_dict)

fig_props = {
    "save": save_plots,
    "ax0_xlabel": "Voltage (V)",
    "save_path": save_path + "sums_gt_RR_",
    "multiple_fig": False,
}


# Run the tests
lfp_sums_gt = lfp_sums_gt.analyze_all_features(
    opt_cv={"active": True, "model": []},
    opt_dist={"active": True, "model": ["RR"]},
    fig_props=fig_props,
    max_nrmse=0.1,
    verbose=False,
    std=True,
)


# Run the tests
lfp_sums_gt = lfp_sums_gt.analyze_all_features(
    opt_cv={"active": True, "model": []},
    opt_dist={"active": True, "model": ["RR"]},
    fig_props=fig_props,
    max_nrmse=0.1,
    verbose=False,
)
plt.show()

nulls_ = lfp_sums_gt.nullspace_dict["Sum"]["RR: 3237.45754"]["nulls"]
y2 = nulls_.nullsp["w_alpha"] + nulls_.nullsp["v_"][-1, :]
# x = lfp_meangt.data.x
# fig, ax = plt.subplots(1,1, figsize=(8,6))
# ax.plot(x.reshape(-1), nulls_.nullsp['w_alpha'], color='darkgrey', zorder=-1, alpha=0.8)
# ax.plot(x.reshape(-1), y2, color='darkgrey', zorder=-1, alpha=0.8)
# ax.fill_between(x.reshape(-1), nulls_.nullsp['w_alpha'], y2=y2, color='darkgrey', zorder=-1, alpha=0.8)
