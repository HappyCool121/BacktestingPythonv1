# performs monte carlo simulation on returns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Module 8: Monte Carlo Simulations (to be moved to statistics module) ---
class monteCarloSimulation:
    """
        Performs a Monte Carlo simulation on a series of returns from a backtest
        to assess the robustness and risk profile of a trading strategy.
    """

    def __init__(self, returns: pd.Series, num_simulations: int = 1000, initial_capital: float = 100000.0):
        """
        Initializes the Monte Carlo simulation module.

        Args:
            returns (pd.Series): A pandas Series of daily returns, derived from portfolio_management.returns
            num_simulations (int): The number of simulation paths to generate.
            initial_capital (float): The starting capital for each simulation path.
        """
        if returns.empty:
            raise ValueError("Input returns series cannot be empty.")

        self.returns = returns
        self.num_simulations = num_simulations
        self.initial_capital = initial_capital
        self.simulation_results = None

    def run(self):
        """
        Executes the Monte Carlo simulation by shuffling the daily returns to create
        a multitude of possible equity curve paths.
        """
        print(f"\nRunning Monte Carlo simulation with {self.num_simulations} paths...")

        # Create a DataFrame to store all simulation paths
        paths_list = []

        for i in range(self.num_simulations):
            # Shuffle the returns (no changes here)
            shuffled_returns = self.returns.sample(frac=1, replace=True).reset_index(drop=True)

            # Create the equity curve path (no changes here)
            path = (1 + shuffled_returns).cumprod() * self.initial_capital
            path = pd.concat([pd.Series([self.initial_capital]), path], ignore_index=True)

            # 2. Append the completed path Series to the list.
            paths_list.append(path)

            i += 1

            # 3. After the loop, create the DataFrame by concatenating all Series in the list at once.
            #    The `axis=1` argument joins them as columns.
        sim_paths = pd.concat(paths_list, axis=1)
        sim_paths.columns = [f'Path_{i + 1}' for i in range(self.num_simulations)]

        # Store the results and print a summary (no changes here)
        self.simulation_results = {
            "paths": sim_paths,
            "final_equity": sim_paths.iloc[-1],
            "max_drawdown": self._calculate_drawdowns(sim_paths)
        }
        print("Monte Carlo simulation complete.")
        return self.simulation_results

    def _calculate_drawdowns(self, paths: pd.DataFrame) -> pd.Series:
        """Calculates the maximum drawdown for each simulated path."""

        cumulative_max = paths.cummax()
        drawdowns = (paths - cumulative_max) / cumulative_max
        return drawdowns.min()

    def plot_results(self, num_paths_to_plot: int = 100):
        """
        Visualizes the Monte Carlo simulation results.

        Args:
            num_paths_to_plot (int): The number of individual paths to show on the plot.
        """
        if self.simulation_results is None:
            print("Error: Please run the simulation before plotting.")
            return

        paths = self.simulation_results["paths"]
        final_equity = self.simulation_results["final_equity"]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 2]})

        # --- Plot 1: Equity Curve Paths ---
        # Plot a subset of the paths for clarity
        paths.iloc[:, :num_paths_to_plot].plot(ax=ax1, legend=False, color='lightblue', alpha=0.5, lw=1)

        # Plot the median path
        median_path = paths.median(axis=1)
        ax1.plot(median_path, color='royalblue', lw=2, label=f'Median Path (50th Percentile)')

        ax1.set_title(f'Monte Carlo Simulation of Equity Paths ({self.num_simulations} runs)', fontsize=16)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_xlabel('Trading Days', fontsize=12)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.legend("")
        ax1.grid(True)

        # --- Plot 2: Distribution of Final Equity ---
        final_equity.plot(kind='hist', bins=100, ax=ax2, color='skyblue', edgecolor='black')

        # Add percentile lines
        p5 = final_equity.quantile(0.05)
        p95 = final_equity.quantile(0.95)
        median_val = final_equity.median()

        ax2.axvline(median_val, color='royalblue', linestyle='--', lw=2, label=f'Median: ${median_val:,.0f}')
        ax2.axvline(p5, color='red', linestyle='--', lw=2, label=f'5th Percentile: ${p5:,.0f}')
        ax2.axvline(p95, color='green', linestyle='--', lw=2, label=f'95th Percentile: ${p95:,.0f}')

        ax2.set_title('Distribution of Final Portfolio Value', fontsize=16)
        ax2.set_xlabel('Final Equity ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Print final statistics
        print("\n--- Monte Carlo Statistical Summary ---")
        print(f"Average Final Equity    : ${final_equity.mean():,.2f}")
        print(f"Median Final Equity     : ${median_val:,.2f}")
        print(f"Std. Dev. Final Equity  : ${final_equity.std():,.2f}")
        print("-" * 35)
        print(f"5th Percentile (Worst)  : ${p5:,.2f}")
        print(f"95th Percentile (Best)  : ${p95:,.2f}")
        print("-" * 35)
        print(f"Probability of Loss     : {((final_equity < self.initial_capital).sum() / self.num_simulations):.2%}")
        print("---------------------------------------")

        return 0
