
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FinancialModelling():
    """
    Class for performing Monte Carlo simulations under different financial modeling frameworks.

    The class does not allow for model calibration.
    """

    def __init__(self, n_paths, steps, starting_price):

        self.n_paths = n_paths # Number of Monte Carlo paths to be generated
        self.steps = steps # Length of each Monte Carlo path
        self.dt = 1/steps # Time interval between two steps (1/365 = day in a year, 1/24 = hour in a day, ...)
        self.starting_price = starting_price # starting price or rate of the object being simulated

        self.simulations_matrix =  np.zeros(shape=(self.n_paths,self.steps)) # empty matrix to be filled with simulated values

class StockPriceModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for stock price modelling.
    
    """
    
    def gbm(self, vol): 
        """
        Geometric Brownian Motion.
        Stock modelling under the Black - Scholes - Merton model.

        """

        if vol <= 0:
            raise ValueError("Volatility must be a positive Float.")
  
        for i in range(self.n_paths):
            # creates a path at time
            dW = np.sqrt(self.dt) * np.random.normal(0, vol, self.steps) # Change in price
            dW[0] = self.starting_price

            self.simulations_matrix[i, :] = dW.cumsum()

        return self.simulations_matrix
    
    def heston(self, start_vol, rho, mu, rate_of_mean_reversion, long_term_vol, vol_vol):
        """
        Simulate stock prices accoridng to the framework established by the Heston model with stochastic volatility.
        
        ------------

        Parameters:
            - start_vol: volatility observed at time zero.
            - rho: correlation between dWS and dWV
            - mu: 
            - rate_of_mean_reversion: speed at which the CIR process for volatility revert to the long run mean.
            - long_term_vol: long run mean for the volatility as observed by market data. 
            - vol_vol: volatility of volatility (a normally distributed random variable).
        
        """

        # Check parameters before execution

        if start_vol <= 0 or vol_vol <= 0 or long_term_vol <= 0:
            raise ValueError("Volatility must be a positive Float.")
        if rho < -1 or rho > 1:
            raise ValueError("rho (correlation coefficient) must be a value between -1 and 1.")
        if rate_of_mean_reversion <= 0:
            raise ValueError("rate_of_mean_reversion must be positive to guarantee mean reversion.")
        

        vol_matrix = np.zeros(shape=(self.n_paths, self.steps)) 
        vol_matrix[:,0] = start_vol
            
        self.simulations_matrix[:,0] = self.starting_price

        correlation_matrix = np.array([[1, rho] , [rho , 1]])
    
        for i in range(self.n_paths):
            for t in range(self.steps-1):

                multi_norm = np.random.multivariate_normal([0,0], correlation_matrix)

                dWS = np.sqrt(self.dt) * multi_norm[0]
                dWv = np.sqrt(self.dt) * multi_norm[1]

                dS = mu * self.simulations_matrix[i,t] * self.dt + np.sqrt(np.abs(vol_matrix[i,t])) * self.simulations_matrix[i,t] * dWS # simulating stock dynamics with volatility updated by the CIR process
                self.simulations_matrix[i,t+1] = self.simulations_matrix[i,t] + dS
            
                dV = rate_of_mean_reversion * (long_term_vol - vol_matrix[i,t]) * self.dt + vol_vol * np.sqrt(np.abs(vol_matrix[i,t])) * dWv # simulating the (stochastic) volatility process
                vol_matrix[i,t+1] = np.abs(vol_matrix[i,t] + dV)

        # for i in range(self.n_paths):

        #     multi_norm = np.random.multivariate_normal([0,0], correlation_matrix, size = self.steps)
        #     dWS = np.sqrt(self.dt) * multi_norm[:,0]
        #     dWv = np.sqrt(self.dt) * multi_norm[:,1]
            
        #     # Simulating stock price dynamics 
        #     dS = mu * self.simulations_matrix[i, :-1] * self.dt + np.sqrt(np.maximum(0, vol_matrix[i, :-1])) * self.simulations_matrix[i, :-1] * dWS[:-1] 
        #     self.simulations_matrix[i, 1:] = self.starting_price + np.cumsum(dS)

        #     # Simulating the volatility process
        #     dV = rate_of_mean_reversion * (long_term_vol - vol_matrix[i, :-1]) * self.dt + vol_vol * np.sqrt(np.maximum(0, vol_matrix[i, :-1])) * dWv[:-1] 
        #     vol_matrix[i, 1:] = vol_matrix[i, :-1] + dV

        return self.simulations_matrix, vol_matrix
    
    def merton_jumps(self, mu, sigma, lambd, m, s):
        """
        Vectorized simulation of stock prices paths according to the Merton Jump-Diffusion model.

        Parameters:
        - mu: drift of the stock
        - sigma: volatility of the stock
        - lambd: Poisson process intensity (expected number of jumps per time unit)
        - m: mean of the logarithm of the jump size
        - s: standard deviation of the logarithm of the jump size
        
        Returns:
        - A matrix with simulated stock prices
        """

        # Check constraints on input parameters
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be greater than zero.")
        if lambd < 0:
            raise ValueError("Jump intensity (lambd) cannot be negative.")
        if s <= 0:
            raise ValueError("Standard deviation of jump size (s) must be greater than zero.")
        

        # Initialize the starting price
        self.simulations_matrix[:,0] = self.starting_price

        for t in range(self.steps-1):
            
            # Brownian motions for stock prices and jump sizes
            dW = np.sqrt(self.dt) * np.random.normal(0, 1, self.n_paths)
            
            # Poisson processes for jump occurrences across all paths
            jump_occurrences = np.random.poisson(lambd * self.dt, self.n_paths)
            
            # Calculate jump sizes for each path; note that we sum over the jumps for paths where multiple jumps occur
            jump_sizes = np.array([np.sum(np.random.normal(m, s, occ)) if occ > 0 else 0 for occ in jump_occurrences])
            
            # GBM part
            dS_gbm = mu * self.simulations_matrix[:,t] * self.dt + sigma * self.simulations_matrix[:,t] * dW
            
            # Total differential including jumps
            dS = dS_gbm + self.simulations_matrix[:,t] * (np.exp(jump_sizes) - 1)
            
            # Update stock prices
            self.simulations_matrix[:,t+1] = self.simulations_matrix[:,t] + dS

        return self.simulations_matrix


class InterestRatesModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for interest rates modelling.
     
    """

    def vasicek(self, rate_of_mean_reversion, long_run_mean, sigma):
        """
        Simulate interest rate dynamics under the assumption of Vasicek model.

        ----------
        Parameters:
            - rate_of_mean_reversion: speed at which the process for rate reverts to the long run mean.
            - long_run_mean: long run mean of the interest rate as observed by market data.
            - sigma: volatility of the interest rate
        
        Returns:
            - A matrix with simulated interest rate paths
        """
        # Check constraints on input parameters
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be greater than zero.")
        if rate_of_mean_reversion <= 0:
            raise ValueError("Rate of mean reversion must be greater than zero.")


        self.simulations_matrix[:,0] = self.starting_price

        for i in range(self.n_paths):
            for t in range(1, self.steps):

                dr = rate_of_mean_reversion * (long_run_mean - self.simulations_matrix[i, t-1]) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal()

                self.simulations_matrix[i, t] = self.simulations_matrix[i, t-1] + dr

        return self.simulations_matrix
    
    def cir(self, rate_of_mean_reversion, long_run_mean, sigma):
        """
        Simulate interest rate dynamics under the assumption of CIR model.

        ----------
        Parameters:
            - rate_of_mean_reversion: speed at which the CIR process reverts to the long run mean.
            - long_run_mean: long run mean of the interest rate as observed by market data.
            - sigma: volatility of the interest rate
        
        Returns:
            - A matrix with simulated interest rate paths
        """
        # Check constraints on input parameters
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be greater than zero.")
        if rate_of_mean_reversion <= 0:
            raise ValueError("Rate of mean reversion must be greater than zero.")
        if long_run_mean <= 0:
            raise ValueError("Long run mean must be greater than zero.")
        

        self.simulations_matrix[:,0] = self.starting_price

        for i in range(self.n_paths):
            for t in range(1, self.steps):

                dr = rate_of_mean_reversion * (long_run_mean - self.simulations_matrix[i, t-1]) * self.dt + sigma * np.sqrt(self.simulations_matrix[i, t-1]) * np.sqrt(self.dt) * np.random.normal()

                self.simulations_matrix[i, t] = self.simulations_matrix[i, t-1] + dr

        return self.simulations_matrix