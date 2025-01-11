# pylint: disable=invalid-name

"""Utility functions related to the PED-ANOVA algorithm.

Functions:
    * compute_hpi(study, gamma_prime, ...): Compute importances for
        hyperparameters in an Optuna study.
    * compute_marginal_gamma_set_pdfs(study, gamma, ...): Compute
        marginal gamma-set PDFs from an Optuna study.
    * plot_marginal_pdfs(study, gamma_prime, ...): Plot mult. pairs of
        marginal gamma-set PDFs from an Optuna study.
"""


__all__ = [
    "compute_hpi",
    "compute_marginal_gamma_set_pdfs"
]

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution
)
from optuna.importance._base import (
    _get_distributions,
    _get_filtered_trials
)
from optuna.study import StudyDirection
from pyfonts import load_font
from sklearn.neighbors import KernelDensity

from src.visualization.color import get_color

EPS = 1e-12
INDIE_FLOWER = load_font(
   font_url="https://github.com/google/fonts/blob/main/ofl/indieflower/IndieFlower-Regular.ttf?raw=true"
)
MIN_GAMMA_SET_TRIALS = 2  # see line 3 of Algorithm 1 in Watanabe et al. (2023)


class _KernelDensity:
    """Wrapper class handling log-transformation & normalization.

    Attributes:
        base_kde: The base KDE, potentially fitted in log-space.
        is_log_scale: Whether the base KDE was fitted in log-space.
        normalization_constant: The normalization constant (computed
          using the trapezoidal rule) to ensure that the estimated
          density integrates to 1 over the domain, if ``normalize`` is
          set to True when initializing the instance.  If not, the
          normalization constant is set to 1.0 (i.e., no normalization).

    Methods:
        score_samples(X): Compute the log-likelihood of each sample
          under the model.
    """

    def __init__(
            self,
            base_kde: KernelDensity,
            bounds: np.ndarray,
            is_log_scale: bool = False,
            normalize: bool = False,
            grid_size: int = 1000
    ) -> None:
        """Initialize the ``_KernelDensity`` instance.

        Args:
            base_kde: The base KDE, potentially fitted in log-space.
            bounds: The lower and upper bounds of the domain. Values
              must be in the original domain (i.e., not
              log-transformed), which may differ from the domain used to
              fit the ``base_kde``.
            is_log_scale: Whether the base KDE was fitted in log-space.
            normalize: Whether to normalize the estimated density such
              that it integrates to 1 over its domain specified by
              ``bounds``.
            grid_size: Number of values at which to sample the KDE to
              compute the normalization factor.  Not used if
              ``normalize`` is False.
        """

        self.base_kde = base_kde
        self.is_log_scale = is_log_scale

        if normalize:
            self.normalization_constant = self._compute_normalization_constant(
                bounds=bounds,
                grid_size=grid_size
            )
        else:
            self.normalization_constant = 1.0

    def score_samples(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """Compute the log-likelihood of each sample under the model.

        Args:
            X: A 1-D array of points to query.  Values must be in the
              original domain (i.e., not log-transformed), which may
              differ from the domain used to fit the ``base_kde``.

        Returns:
            Log-likelihood of each sample in ``X``.
        """

        if X.ndim != 1:
            raise ValueError(
                f"'X' must be a 1-D array, but got shape {X.shape}."
            )

        # NOTE: For KDEs fitted to log-transformed data, the change-of-variables formula gives
        #       f_X(x) = f_Y(log(x)) / x, where f_Y(y) is the density of Y = log(X) (i.e., the
        #       ``base_kde`` fitted in log-space).  Since we're dealing with log-probabilities, we
        #       have log(f_X(x)) = log(f_Y(log(x))) - log(x).
        if self.is_log_scale:
            log_dens = self.base_kde.score_samples(np.log(X)[:, np.newaxis]) - np.log(X)
        else:
            log_dens = self.base_kde.score_samples(X[:, np.newaxis])

        return log_dens - np.log(self.normalization_constant)

    def _compute_normalization_constant(
            self,
            bounds: np.ndarray,
            grid_size: int
    ) -> float:
        """Compute the normalization constant.

        Args:
            bounds: The lower and upper bounds of the domain. Values
              must be in the original domain (i.e., not
              log-transformed), which may differ from the domain used to
              fit the ``base_kde``.
            grid_size: Number of values at which to sample the KDE to
              compute the normalization factor.
        """

        if bounds.ndim != 1 or bounds.size != 2:
            raise ValueError(
                f"'bounds' must be a 1-D array of size 2, but got shape {bounds.shape}."
            )

        if bounds[0] >= bounds[1]:
            raise ValueError(
                f"Lower bound ({bounds[0]}) must be strictly less than upper bound ({bounds[1]})."
            )

        # Create grid at which to evaluate the (not yet normalized) KDE
        # NOTE: np.linspace(log(a), log(b), n) and np.log(np.geomspace(a, b, n)) are equivalent
        lower, upper = np.log(bounds) if self.is_log_scale else bounds
        grid = np.linspace(lower, upper, grid_size)

        # Evaluate density, possibly accounting for log-transformation
        log_dens = self.base_kde.score_samples(grid[:, np.newaxis])
        if self.is_log_scale:
            log_dens -= grid  # change-of-variables formula, see ``score_samples()``

        # Convert log-probabilities to probabilities
        dens = np.exp(log_dens)

        # Integrate using the trapezoidal rule (https://en.wikipedia.org/wiki/Trapezoidal_rule)
        dx = np.diff(
            np.exp(grid) if self.is_log_scale else grid
        )
        integral_trapezoid = np.sum(
            dx * 0.5 * (dens[1:] + dens[:-1])  # dx * 1/2 (f(b) + f(a))
        )

        return integral_trapezoid


def compute_hpi(
        study: optuna.study.Study,
        gamma_prime: float,
        gamma: Optional[float] = None,
        params: Optional[List[str]] = None,
        normalize: bool = False,
        grid_size: int = 1000
) -> Dict[str, float]:
    """Compute importances for hyperparameters in an Optuna study.

    Args:
        study: Optuna study to evaluate.
        gamma_prime: Quantile to define the binary function in the
          local space.  Corresponds to gamma' in Eqn. (16) of Watanabe
          et al. (2023).
        gamma: Quantile to define the binary function in the global
          space.  Corresponds to gamma in Eqn. (16) of Watanabe et al.
          (2023).  If None, gamma is set to 1.0.
        params: List of parameter names to compute importances for.  If
          None, importances are computed for all parameters found in the
          study.
        normalize: Whether to normalize the gamma-set PDFs such that
          they integrate to 1 over the domains from which the
          parameters of the study were sampled.
        grid_size: Number of points at which to sample the gamma-set
          PDFs to compute importances.

    Returns:
        A dictionary mapping parameter names to their importances.
    """

    gamma = _check_quantile_inputs(gamma_prime, gamma)

    if grid_size <= 0:
        raise ValueError(f"'grid_size' must be positive, but got {grid_size}.")

    # Compute size of gamma-sets
    dists = _get_distributions(study, params)
    trials = _get_filtered_trials(
        study,
        params=(params if params is not None else list(dists.keys())),
        target=None
    )
    local_num_trials = int(np.ceil(gamma_prime * len(trials)))
    global_num_trials = int(np.ceil(gamma * len(trials)))

    if local_num_trials == global_num_trials:
        warnings.warn(
            f"Gamma-sets contain same {local_num_trials} trials. Unable to determine importances. "
            f"Consider adjusting 'gamma' ({gamma}) and 'gamma_prime' ({gamma_prime})."
        )
        return {}

    local_kdes_dict = compute_marginal_gamma_set_pdfs(
        study=study,
        gamma=gamma_prime,
        params=params,
        normalize=normalize
    )
    global_kdes_dict = compute_marginal_gamma_set_pdfs(
        study=study,
        gamma=gamma,
        params=params,
        normalize=normalize
    )

    # Compute HPIs according to Eqn. (16) of Watanabe et al. (2023)
    hpi_dict = {}
    hpi_total = 0.0

    for param_name, global_kde in global_kdes_dict.items():
        # Create grid in the original domain to evaluate KDEs
        x_min, x_max = _get_bounds(global_kde["dist"])
        if global_kde["kde"].is_log_scale:
            grid = np.geomspace(x_min, x_max, grid_size)
        else:
            grid = np.linspace(x_min, x_max, grid_size)

        # Evaluate KDEs computed according to Eqn. (14) of Watanabe et al. (2023)
        local_pdf = np.exp(local_kdes_dict[param_name]["kde"].score_samples(grid))
        global_pdf = np.exp(global_kde["kde"].score_samples(grid)) + EPS

        # Compute HPI according to Eqn. (16) of Watanabe et al. (2023)
        hpi = global_pdf @ ((local_pdf / global_pdf - 1) ** 2)
        hpi_dict[param_name] = hpi
        hpi_total += hpi

    # Rescale properly (see "normalization constant" Z in Eqn. (16) of Watanabe et al. (2023))
    hpi_dict = {
        param_name: hpi / hpi_total
        for param_name, hpi in hpi_dict.items()
    }

    return hpi_dict


def compute_marginal_gamma_set_pdfs(
        study: optuna.study.Study,
        gamma: float,
        params: Optional[List[str]] = None,
        normalize: bool = False
) -> dict[str, dict[str, Union[BaseDistribution, float, _KernelDensity, np.ndarray]]]:
    """Compute marginal gamma-set PDFs from an Optuna study.

    Args:
        study: Optuna study to evaluate.
        gamma: Quantile for which to compute the marginal gamma-set PDF.
          Corresponds to gamma in Eqn. (14) of Watanabe et al. (2023).
        params: List of parameter names to compute marginal gamma-set
          PDFs for.  If None, PDFs are computed for all parameters found
          in the study.
        normalize: Whether to normalize the gamma-set PDFs such that
          they integrate to 1 over the domains from which the
          parameters of the study were sampled.

    Returns:
        A dictionary, mapping (selected) parameter names to dictionaries
        containing (1) the distribution of the parameter, (2) the
        quantile ``gamma``, (3) the estimated gamma-set PDF, and (4) the
        observed values of the parameter that are in the gamma-set.  The
        gamma-set PDF is represented by a ``_KernelDensity`` instance
        that automatically handles log-transformation and normalization.
    """

    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            f"'gamma' should be a float between 0 (exclusive) and 1 (inclusive), but got {gamma}."
        )

    # Get distributions and trials from study object
    dists = _get_distributions(study, params)  # dictionary of distributions found in the study
    if params is None:
        params = list(dists.keys())

    # Remove parameters for which only a single unique value was observed in the study
    non_single_dists = {
        name: dist
        for name, dist in dists.items()
        if not dist.single()
    }
    if len(non_single_dists) == 0:
        return {}

    # Filter trials and keep only those that ...
    #   - are complete (e.g., not pruned)
    #   - include all hyperparameters specified by ``params``
    #   - have finite target value (w.r.t. the objective function [e.g., validation loss])
    all_trials = _get_filtered_trials(study, params=params, target=None)

    if len(all_trials) == 0:
        warnings.warn("No trials found that match the filtering criteria.")
        return {}

    # Determine size of gamma-set
    is_lower_better = study.directions[0] == StudyDirection.MINIMIZE
    sign = 1.0 if is_lower_better else -1.0
    loss_values = sign * np.array([trial.value for trial in all_trials])
    num_include = int(np.ceil(gamma * len(loss_values)))  # number of observations to include

    # Select trials belonging to gamma-set
    include_trials = np.argsort(np.argsort(loss_values)) + 1 <= num_include
    top_trials = [
        trial
        for trial, include_trial in zip(all_trials, include_trials)
        if include_trial
    ]

    if len(top_trials) < MIN_GAMMA_SET_TRIALS:
        raise RuntimeError(
            f"Gamma-set must contain at least {MIN_GAMMA_SET_TRIALS} trials, but found only "
            f"{len(top_trials)}. Increase the number of trials or consider relaxing 'gamma' "
            f"({gamma})."
        )

    # Extract values observed in ``top_trials`` for each hyperparameter
    param_values_dict = {}
    for param_name in non_single_dists:
        param_values_dict[param_name] = np.array(
            [trial.params[param_name] for trial in top_trials]
        )  # 1-D array of observed values

    # Estimate marginal gamma-set PDFs
    # TODO: Handle categorical distributions properly!
    kdes_dict = {}
    for param_name, dist in non_single_dists.items():
        # Check whether parameter was sampled from a log-scaled domain
        if isinstance(dist, (FloatDistribution, IntDistribution)):
            is_log_scale = dist.log
        else:
            is_log_scale = False

        # Get observed values and bounds for the parameter (both are 1-D arrays)
        values = param_values_dict[param_name]
        bounds = _get_bounds(dist)

        # Fit KDE on observed values
        kde = KernelDensity(
            bandwidth=_scott_bandwidth(
                np.log(values) if is_log_scale else values
            ),
            kernel="gaussian"
        )
        kde.fit(
            (np.log(values) if is_log_scale else values)[:, np.newaxis]
        )

        kdes_dict[param_name] = {
            "dist": dist,
            "gamma": gamma,
            "kde": _KernelDensity(
                base_kde=kde,
                is_log_scale=is_log_scale,
                bounds=bounds,
                normalize=normalize
            ),
            "vals": values  # observed values in gamma-set, not transformed
        }

    return kdes_dict


# noinspection PyTypeChecker
def plot_marginal_pdfs(
        study: optuna.study.Study,
        gamma_prime: float,
        gamma: Optional[float] = None,
        params: Optional[List[str]] = None,
        params_aliases: List[str] = None,
        normalize: bool = False,
        grid_size: int = 1000,
        fig_layout: Optional[Tuple[int, int]] = None,
        fig_size: Optional[Tuple[int, int]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot mult. pairs of marginal gamma-set PDFs from an Optuna study.

    Args:
        study: Optuna study to evaluate.
        gamma_prime: Quantile to define the binary function in the
          local space.  Corresponds to gamma' in Eqn. (16) of Watanabe
          et al. (2023).
        gamma: Quantile to define the binary function in the global
          space.  Corresponds to gamma in Eqn. (16) of Watanabe et al.
          (2023).  If None, gamma is set to 1.0.
        params: List of parameter names for which to plot marginal
          gamma-set PDFs.  If None, PDFs are plotted for all parameters
          found in the study.
        params_aliases: List of parameter names to use as subplot
          titles.  If None, the original parameter names are used.
        normalize: Whether to normalize the gamma-set PDFs such that
          they integrate to 1 over the domains from which the
          parameters of the study were sampled.
        grid_size: Number of points at which to sample the gamma-set
          PDFs when plotting.
        fig_layout: Tuple specifying the number of rows/columns of the
          subplot grid.  If None, all PDFs are plotted in a single row.
        fig_size: Tuple specifying the width and height of the figure.
          If None, the figure size is automatically determined based on
          the number of subplots.

    Returns:
        A tuple containing the figure and axes objects.
    """

    gamma = _check_quantile_inputs(gamma_prime, gamma)

    if grid_size <= 0:
        raise ValueError(f"'grid_size' must be positive, but got {grid_size}.")

    # Compute marginal gamma-set PDFs
    local_kdes_dict = compute_marginal_gamma_set_pdfs(
        study=study,
        gamma=gamma_prime,
        params=params,
        normalize=normalize
    )
    global_kdes_dict = compute_marginal_gamma_set_pdfs(
        study=study,
        gamma=gamma,
        params=params,
        normalize=normalize
    )
    # Compute HPIs
    hpi_dict = compute_hpi(
        study=study,
        gamma_prime=gamma_prime,
        gamma=gamma,
        params=params,
        normalize=normalize,
        grid_size=grid_size
    )

    # Create figure and axes for plotting
    num_plots = len(local_kdes_dict)
    if fig_layout is None:
        fig_layout = (1, num_plots)
    elif fig_layout[0] * fig_layout[1] < num_plots:
        warnings.warn(
            f"Number of subplots ({fig_layout[0] * fig_layout[1]}) is less than the number of "
            f"parameters ({num_plots}). Setting 'fig_layout' to (1, {num_plots})."
        )
        fig_layout = (1, num_plots)

    if fig_size is None:
        fig_size = (4.0 * fig_layout[1], 2.25 * fig_layout[0])  # approx. 16:9 aspect ratio

    with sns.axes_style({
        "axes.edgecolor": get_color("anthracite", tint=0.2),
        "axes.spines.right": False,
        "axes.spines.top": False
    }):
        fig, axes = plt.subplots(fig_layout[0], fig_layout[1], figsize=fig_size)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes_flattened = axes.flatten()

        # Reshape ``axes`` to 2-D array for single-row or single-column plots
        if fig_layout[0] == 1:  # single row
            axes = axes.reshape(1, -1)
        elif fig_layout[1] == 1:  # single column
            axes = axes.reshape(-1, 1)

        # Plot individual PDFs
        for idx, (param_name, local_kde) in enumerate(local_kdes_dict.items()):
            # Create grid in the original domain to evaluate KDEs
            x_min, x_max = _get_bounds(local_kde["dist"])
            is_log_scale = local_kde["kde"].is_log_scale
            grid = (
                np.geomspace(x_min, x_max, grid_size)
                if is_log_scale
                else np.linspace(x_min, x_max, grid_size)
            )

            # Get values of gamma-sets and evaluate KDEs
            local_vals = local_kde["vals"]
            global_vals = global_kdes_dict[param_name]["vals"]
            local_pdf = np.exp(local_kde["kde"].score_samples(grid))
            global_pdf = np.exp(global_kdes_dict[param_name]["kde"].score_samples(grid))

            # Plot PDFs
            ax: plt.Axes = axes_flattened[idx]
            _plot_pdf(
                ax=ax,
                grid=grid,
                local_pdf=local_pdf,
                global_pdf=global_pdf,
                local_vals=local_vals,
                global_vals=global_vals,
                is_log_scale=is_log_scale
            )

            # Set title for subplot
            _alias = params_aliases[idx] if params_aliases is not None else param_name
            subplot_title = f"{_alias} ({hpi_dict[param_name] * 100:.1f}%)"
            ax.set_title(subplot_title, size=10)

        # Hide empty subplots
        for idx in range(len(local_kdes_dict), len(axes_flattened)):
            ax: plt.Axes = axes_flattened[idx]
            ax.set_visible(False)

        # Add labels to outer subplots
        for row in range(fig_layout[0]):
            ax: plt.Axes = axes[row, 0]
            ax.set_ylabel("Density", font=INDIE_FLOWER, size=12)
        for col in range(fig_layout[1]):
            ax: plt.Axes = axes[-1, col]
            ax.set_xlabel("Parameter Values", font=INDIE_FLOWER, size=12)

        plt.tight_layout()

        # Create legend at bottom of figure
        # TODO: Improve legend placement.  Current placement is a quick hack that works reasonably
        #       well for plots with 1 to 3 columns, and 1 to 4 rows.
        h_adjust = 0.06 if fig_layout[1] == 1 else 0.035 / (fig_layout[1] - 1)
        v_adjust = -0.05 if fig_layout[0] == 1 else -0.05 / (fig_layout[0] - 1)
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color=get_color("red")),
                plt.Line2D([0], [0], color="black", linestyle="--")
            ],
            labels=[
                fr"$\gamma' = {gamma_prime}$",
                fr"$\gamma = {gamma}$"
            ],
            loc="center",
            bbox_to_anchor=(0.5 + h_adjust, v_adjust),
            ncol=2
        )

    return fig, axes


def _check_quantile_inputs(
        gamma_prime: float,
        gamma: Optional[float] = None
) -> float:
    """Check that quantile inputs are valid.

    Args:
        gamma_prime: Quantile to define the binary function in the
          local space.
        gamma: Quantile to define the binary function in the global
          space.

    Returns:
        The value of 'gamma'.  Defaults to 1.0 if no value was passed.

    Raises:
        ValueError: If the quantile inputs are invalid.
    """

    if not 0.0 < gamma_prime < 1.0:
        raise ValueError(
            "'gamma_prime' should be a float between 0 and 1 (both exclusive), "
            f"but got {gamma_prime}."
        )

    if gamma is None:
        return 1.0

    if not gamma_prime < gamma <= 1.0:
        raise ValueError(
            "'gamma' should be None or a float strictly larger than 'gamma_prime' and smaller "
            f"than 1, but got 'gamma' = {gamma} and 'gamma_prime' = {gamma_prime}."
        )

    return gamma


def _get_bounds(dist: BaseDistribution) -> np.ndarray:
    """Get bounds of a distribution.

    Args:
        dist: Distribution to extract bounds from.

    Returns:
        Lower and upper bounds of the distribution in a 1-D array with
        two entries.
    """

    if isinstance(dist, CategoricalDistribution):
        return np.array([min(dist.choices), max(dist.choices)])

    if isinstance(dist, (FloatDistribution, IntDistribution)):
        return np.array([dist.low, dist.high])

    raise RuntimeError(f"Unsupported distribution: {dist}")


def _plot_pdf(
        ax: plt.Axes,
        grid: np.ndarray,
        local_pdf: np.ndarray,
        global_pdf: np.ndarray,
        local_vals: np.ndarray,
        global_vals: np.ndarray,
        is_log_scale: bool
) -> None:
    """Plot local and global marginal gamma-set PDFs.

    Args:
        ax: Axes object to plot the PDFs on.
        grid: Grid of points at which the PDFs are evaluated.
        local_pdf: Local marginal gamma-set PDF.
        global_pdf: Global marginal gamma-set PDF.
        local_vals: Values belonging the local gamma-set.
        global_vals: Values belonging the global gamma-set.
        is_log_scale: Whether the parameter was sampled from a
          log-scaled domain.
    """

    # Plot shaded area where local_pdf > global_pdf
    ax.fill_between(
        grid,
        global_pdf,
        local_pdf,
        where=(local_pdf > global_pdf),
        color=get_color("red"),
        alpha=0.2,
        interpolate=True
    )

    # Find boundaries of filled regions and highlight by vertical lines
    diff = local_pdf - global_pdf
    crossings = np.where(np.diff(diff > 0))[0]
    max_density = max(np.max(local_pdf), np.max(global_pdf))

    for idx in crossings:
        # Linear interpolation to find more precise crossing point
        x0, x1 = grid[idx], grid[idx + 1]
        y0, y1 = diff[idx], diff[idx + 1]
        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)

        # Find the y-value at the intersection (can use either PDF since they intersect here)
        y_cross = np.interp(x_cross, grid, local_pdf)

        ax.axvline(
            x=x_cross,
            ymin=0,
            ymax=y_cross / (max_density * 1.1),  # convert to axis coordinates
            color=get_color("red"),
            linestyle=":"
        )

    # Check whether shading extends to the boundaries, and highlight if necessary
    if diff[0] > 0:  # Shading starts at left boundary
        x_left = grid[0]
        y_left = local_pdf[0]
        ax.axvline(
            x=x_left.item(),
            ymin=0,
            ymax=y_left / (max_density * 1.1),
            color=get_color("red"),
            linestyle=":"
        )

    if diff[-1] > 0:  # Shading extends to right boundary
        x_right = grid[-1]
        y_right = local_pdf[-1]
        ax.axvline(
            x=x_right.item(),
            ymin=0,
            ymax=y_right / (max_density * 1.1),
            color=get_color("red"),
            linestyle=":"
        )

    # Plot PDFs and highlight individual values
    sns.lineplot(x=grid, y=global_pdf, ax=ax, color="black", linestyle='--')
    sns.rugplot(x=global_vals, ax=ax, color="black")
    sns.lineplot(x=grid, y=local_pdf, ax=ax, color=get_color("red"))
    sns.rugplot(x=local_vals, ax=ax, color=get_color("red"))

    # Control plot aesthetics
    if is_log_scale:
        ax.set_xscale("log")
    ax.tick_params(axis="both", color=get_color("anthracite", tint=0.2))

    # Set y-axis to start at 0.0 and end slightly above max. density value
    ax.set_ylim(bottom=0.0, top=max_density * 1.1)


def _scott_bandwidth(X: np.ndarray) -> float:
    """Compute bandwidth using Scott's rule.

    This implementation follows the approach used in Optuna's PED-ANOVA
    implementation (see _ScottParzenEstimator class).

    Args:
        X: The data to compute the bandwidth for.  Must be 1-D.

    Returns:
        The bandwidth computed using Scott's rule.
    """

    if X.ndim != 1:
        raise ValueError(
            f"Input must be a 1-D array, but got shape {X.shape}."
        )

    # NOTE: ``h_min = 0.5 / 1.64`` as in Optuna's implementation is not used here, as it tends to
    #       produce overly smooth KDEs for the experimental data we're working with.
    sigma = np.std(X, ddof=1)  # ``ddof=1`` for (corrected) sample standard deviation
    q25, q75 = np.percentile(X, [25, 75])
    h = 1.059 * min(sigma, (q75 - q25) / 1.349) * len(X) ** (-0.2)

    return h
