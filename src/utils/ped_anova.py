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
    "compute_marginal_gamma_set_pdfs",
    "plot_marginal_pdfs"
]

import warnings
from collections import Counter
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


class _DiscreteDensity:
    """Wrapper class for discrete probability mass functions.

    Attributes:
        probabilities: Dictionary mapping discrete values to their
          probabilities.

    Methods:
        score_samples(X): Compute the log-likelihood of each sample
          under the model.
    """

    def __init__(
            self,
            values: np.ndarray
    ) -> None:
        """Initialize the ``_DiscreteDensity`` instance.

        Args:
            values: 1-D array of discrete values to compute the PMF for.
        """

        # Compute probability mass function from observed values
        unique_values = np.unique(values)
        value_counts = Counter(values)
        total_values = len(values)

        # NOTE: All probabilities stored in ``self.probabilities`` are strictly positive, as
        #       ``unique_values`` only contains values that were observed at least once so that
        #       ``value_counts[value]`` >= 1.
        self.probabilities = {
            value: value_counts[value] / total_values for value in unique_values
        }

    def score_samples(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """Compute the log-likelihood of each sample under the model.

        Args:
            X: A 1-D array of points to query.

        Returns:
            Log-likelihood of each sample in ``X``.

        Raises:
            ValueError: If ``X`` is not a 1-D array.
        """

        if X.ndim != 1:
            raise ValueError(
                f"'X' must be a 1-D array, but got shape {X.shape}."
            )

        log_dens = np.array([
            np.log(self.probabilities[x])
            if x in self.probabilities
            else -np.inf
            for x in X
        ])

        return log_dens


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

        Raises:
            ValueError: If ``X`` is not a 1-D array.
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

        Returns:
            The normalization constant computed using the trapezoidal
            rule to ensure that the estimated density integrates to 1
            over the domain specified by ``bounds``.

        Raises:
            ValueError: If invalid bounds are provided.
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
        compute_local_hpi: bool = False,
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
        compute_local_hpi: Whether to compute local HPIs.  If False,
          global HPIs are computed by replacing the marginal gamma-set
          PDFs in the denominator of Eqn. (16) of Watanabe et al. (2023)
          with the uniform distribution over the parameter's domain.
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

    # Compute size of gamma-sets to check whether they differ (local HPI only)
    if compute_local_hpi:
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
                f"Gamma-sets contain same {local_num_trials} trials. Unable to determine "
                f"importances. Consider adjusting 'gamma' ({gamma}) and 'gamma_prime' "
                f"({gamma_prime})."
            )
            return {}

    # Marginal gamma'-set PDFs
    local_kdes_dict = compute_marginal_gamma_set_pdfs(
        study=study,
        gamma=gamma_prime,
        params=params,
        normalize=normalize
    )

    # Marginal gamma-set PDFs (only necessary for local HPIs, else replaced by uniform dist.)
    if compute_local_hpi:
        global_kdes_dict = compute_marginal_gamma_set_pdfs(
            study=study,
            gamma=gamma,
            params=params,
            normalize=normalize
        )

    # Compute HPIs according to Eqn. (16) of Watanabe et al. (2023)
    hpi_dict = {}
    hpi_total = 0.0

    for param_name, local_kde in local_kdes_dict.items():
        # Create grid in the original domain on which to evaluate KDEs
        dist: BaseDistribution = local_kde["dist"]
        kde = local_kde["kde"]

        is_categorical = isinstance(dist, CategoricalDistribution)
        is_log_scale = kde.is_log_scale if isinstance(kde, _KernelDensity) else False

        if is_categorical:
            dist: CategoricalDistribution
            grid = np.array(dist.choices)
        else:
            x_min, x_max = _get_bounds(dist)
            grid = (
                np.geomspace(x_min, x_max, grid_size)
                if is_log_scale
                else np.linspace(x_min, x_max, grid_size)
            )

        # Evaluate KDEs computed according to Eqn. (14) of Watanabe et al. (2023)
        local_pdf = np.exp(local_kde["kde"].score_samples(grid))

        if compute_local_hpi:
            # Evaluate marginal gamma-set PDFs
            global_pdf = np.exp(global_kdes_dict[param_name]["kde"].score_samples(grid)) + EPS
        else:
            # Evaluate uniform distribution over the parameter's domain
            if is_categorical:
                denominator = len(dist.choices)
            else:
                denominator = x_max - x_min
            global_pdf = np.ones_like(grid) / denominator

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
) -> dict[str, dict[str, Union[
    BaseDistribution, float, _DiscreteDensity, _KernelDensity, np.ndarray
]]]:
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
    kdes_dict = {}
    for param_name, dist in non_single_dists.items():
        # Get observed values of parameter
        values = param_values_dict[param_name]

        # Fit KDE to observed values, properly accounting for categorical parameters
        if isinstance(dist, CategoricalDistribution):
            kde = _DiscreteDensity(values)
        else:
            dist: Union[FloatDistribution, IntDistribution]

            # Determine whether parameter was sampled from a log-scaled domain
            is_log_scale = dist.log

            # Initialize and fit KDE on observed values
            base_kde = KernelDensity(
                bandwidth=_scott_bandwidth(
                    np.log(values) if is_log_scale else values
                ),
                kernel="gaussian"
            )
            base_kde.fit(
                (np.log(values) if is_log_scale else values)[:, np.newaxis]
            )

            kde = _KernelDensity(
                base_kde=base_kde,
                is_log_scale=is_log_scale,
                bounds=_get_bounds(dist),
                normalize=normalize
            )

        # Collect results in dictionary
        kdes_dict[param_name] = {
            "dist": dist,
            "gamma": gamma,
            "kde": kde,
            "vals": values
        }

    return kdes_dict


# noinspection PyTypeChecker
def plot_marginal_pdfs(
        study: optuna.study.Study,
        gamma_prime: float,
        gamma: Optional[float] = None,
        plot_against_uniform: bool = False,
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
        plot_against_uniform: Whether to plot the marginal gamma'-set
          PDFs against the uniform distribution over the parameter's
          domain.  If False, the PDFs are plotted against the marginal
          gamma-set PDFs.
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

    # Compute marginal gamma'-set PDFs
    local_kdes_dict = compute_marginal_gamma_set_pdfs(
        study=study,
        gamma=gamma_prime,
        params=params,
        normalize=normalize
    )

    # Compute marginal gamma-set PDFs only if necessary
    if not plot_against_uniform:
        global_kdes_dict = compute_marginal_gamma_set_pdfs(
            study=study,
            gamma=gamma,
            params=params,
            normalize=normalize
        )

    # Compute (local or global) HPIs
    hpi_dict = compute_hpi(
        study=study,
        gamma_prime=gamma_prime,
        gamma=gamma,
        compute_local_hpi=not plot_against_uniform,
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
        fig_size = (4.0 * fig_layout[1], 3.0 * fig_layout[0])  # approx. 4:3 aspect ratio

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
        if params is None:
            params = list(local_kdes_dict.keys())
        for idx, param_name in enumerate(params):
            # Create grid in the original domain on which to evaluate KDEs
            dist: BaseDistribution = local_kdes_dict[param_name]["dist"]
            kde = local_kdes_dict[param_name]["kde"]

            is_categorical = isinstance(dist, CategoricalDistribution)
            is_log_scale = kde.is_log_scale if isinstance(kde, _KernelDensity) else False

            if is_categorical:
                dist: CategoricalDistribution
                grid = np.array(dist.choices)
            else:
                x_min, x_max = _get_bounds(dist)
                grid = (
                    np.geomspace(x_min, x_max, grid_size)
                    if is_log_scale
                    else np.linspace(x_min, x_max, grid_size)
                )

            # Evaluate marginal gamma'-set PDF on grid
            local_pdf = np.exp(kde.score_samples(grid))

            # Evaluate marginal gamma-set PDF / uniform distribution on grid
            if plot_against_uniform:  # uniform distribution
                denominator = x_max - x_min if not is_categorical else len(dist.choices)
                global_pdf = np.ones_like(grid) / denominator
            else:  # marginal gamma-set PDF
                global_pdf = np.exp(global_kdes_dict[param_name]["kde"].score_samples(grid))

            # NOTE: ``local_vals`` and ``global_vals`` are only used for plotting rug plots, which
            #       are redundant for discrete distributions.
            if is_categorical:
                local_vals, global_vals = None, None
            else:
                local_vals = local_kdes_dict[param_name]["vals"]
                global_vals = global_kdes_dict[param_name]["vals"] if not plot_against_uniform \
                    else None

            # Plot PDFs
            ax: plt.Axes = axes_flattened[idx]
            _plot_pdf(
                ax=ax,
                grid=grid,
                local_pdf=local_pdf,
                global_pdf=global_pdf,
                is_categorical=is_categorical,
                is_log_scale=is_log_scale,
                local_vals=local_vals,
                global_vals=global_vals
            )

            # Set title for subplot
            _alias = params_aliases[idx] if params_aliases is not None else param_name
            subplot_title = f"{_alias} ({hpi_dict[param_name] * 100:.1f}%)"
            # TODO: Fontsize of subplot titles should change with number of subplots
            ax.set_title(subplot_title, size=12)

        # Hide empty subplots
        for idx in range(len(local_kdes_dict), len(axes_flattened)):
            ax: plt.Axes = axes_flattened[idx]
            ax.set_visible(False)

        # Add labels to outer subplots
        # TODO: Fontsize of axis labels should change with number of subplots
        for row in range(fig_layout[0]):
            ax: plt.Axes = axes[row, 0]
            ax.set_ylabel("Density", font=INDIE_FLOWER, size=14)
        for col in range(fig_layout[1]):
            ax: plt.Axes = axes[-1, col]
            ax.set_xlabel("Parameter Values", font=INDIE_FLOWER, size=14)

        plt.tight_layout()

        # Create legend at bottom of figure
        # TODO: Improve legend placement.  Current placement is a quick hack that works reasonably
        #       well for plots with 1 to 3 columns, and 1 to 4 rows.
        h_adjust = 0.06 if fig_layout[1] == 1 else 0.035 / (fig_layout[1] - 1)
        v_adjust = -0.05 if fig_layout[0] == 1 else -0.05 / (fig_layout[0] - 1)
        labels = [fr"$\gamma = {gamma_prime}$", fr"$u(\mathcal{{X}}_d)$"] if plot_against_uniform \
            else [fr"$\gamma' = {gamma_prime}$", fr"$\gamma = {gamma}$"]
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color=get_color("red")),
                plt.Line2D([0], [0], color="black", linestyle="--")
            ],
            labels=labels,
            loc="center",
            bbox_to_anchor=(0.5 + h_adjust, v_adjust),
            ncol=2,
            fontsize=12
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
        is_categorical: bool,
        is_log_scale: bool,
        local_vals: Optional[np.ndarray] = None,
        global_vals: Optional[np.ndarray] = None
) -> None:
    """Plot local and global marginal gamma-set PDFs.

    Args:
        ax: Axes object to plot the PDFs on.
        grid: Grid of points at which the PDFs are evaluated.
        local_pdf: Local marginal gamma-set PDF.
        global_pdf: Global marginal gamma-set PDF.
        is_categorical: Whether the parameter is categorical.
        is_log_scale: Whether the parameter was sampled from a
          log-scaled domain.
        local_vals: Values belonging to the local gamma-set.  These will
          be visualized by a rug plot.  Redundant for categorical
          parameters.
        global_vals: Values belonging to the global gamma-set.  These
          will be visualized by a rug plot.  Redundant for categorical
          parameters.
    """

    # Set y-axis to always start at 0.0 and end slightly above max. density value
    y_upper = max(np.max(local_pdf), np.max(global_pdf)) * 1.1
    ax.set_ylim(bottom=0.0, top=y_upper)

    if is_categorical:
        # Settings for bar plots
        x = np.arange(len(grid))
        width = 0.35

        if np.allclose(global_pdf, 1.0 / len(grid)):  # uniform distribution as comparison
            uniform_height = 1.0 / len(grid)

            # Plot shaded area where local_pdf > uniform_height
            excess_heights = np.maximum(local_pdf - uniform_height, 0)
            mask = excess_heights > 0
            if np.any(mask):
                ax.bar(
                    x[mask],
                    excess_heights[mask],
                    1.5 * width,
                    bottom=uniform_height,
                    color=get_color("red", tint=0.8),
                    linewidth=0
                )

            # Plot local PMF and uniform distribution
            ax.bar(x, local_pdf, 1.5 * width, facecolor="none", edgecolor=get_color("red"))
            ax.axhline(y=uniform_height, color="black", linestyle="--")
        else:  # global PMF as comparison

            # Plot shaded area where local_pdf > global_pdf
            excess_heights = np.maximum(local_pdf - global_pdf, 0)
            mask = excess_heights > 0
            if np.any(mask):
                ax.bar(
                    x[mask] - width / 2,
                    excess_heights[mask],
                    width,
                    bottom=global_pdf[mask],
                    color=get_color("red", tint=0.8),
                    linewidth=0
                )

            # Plot local PMF and global PMF
            ax.bar(x - width / 2, local_pdf, width, facecolor="none", edgecolor=get_color("red"))
            ax.bar(
                x + width / 2,
                global_pdf,
                width,
                facecolor="none",
                edgecolor="black",
                linestyle="--"
            )

        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(grid)
    else:  # non-categorical parameters
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
                ymax=y_cross / y_upper,  # convert to axis coordinates
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
                ymax=y_left / y_upper,  # convert to axis coordinates
                color=get_color("red"),
                linestyle=":"
            )

        if diff[-1] > 0:  # Shading extends to right boundary
            x_right = grid[-1]
            y_right = local_pdf[-1]
            ax.axvline(
                x=x_right.item(),
                ymin=0,
                ymax=y_right / y_upper,  # convert to axis coordinates
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
    # TODO: Fontsize of tick labels should change with number of subplots
    ax.tick_params(axis="both", color=get_color("anthracite", tint=0.2), labelsize=12)


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
