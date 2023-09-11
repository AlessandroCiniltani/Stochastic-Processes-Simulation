
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
        self.starting_price = starting_price

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

        if vol == None:
            vol = 1
  
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
        vol_matrix = np.zeros(shape=(self.n_paths, self.steps)) 
        vol_matrix[:,0] = start_vol
            
        self.simulations_matrix[:,0] = self.starting_price

        correlation_matrix = np.array([[1, rho] , [rho , 1]])
    
        # for i in range(self.n_paths):
        #     for t in range(self.steps-1):

        #         multi_norm = np.random.multivariate_normal([0,0], correlation_matrix)

        #         dWS = np.sqrt(self.dt) * multi_norm[0]
        #         dWv = np.sqrt(self.dt) * multi_norm[1]

        #         dS = mu * self.simulations_matrix[i,t] * self.dt + np.sqrt(np.abs(vol_matrix[i,t])) * self.simulations_matrix[i,t] * dWS # simulating stock dynamics with volatility updated by the CIR process
        #         self.simulations_matrix[i,t+1] = self.simulations_matrix[i,t] + dS
            
        #         dV = rate_of_mean_reversion * (long_term_vol - vol_matrix[i,t]) * self.dt + vol_vol * np.sqrt(np.abs(vol_matrix[i,t])) * dWv # simulating the (stochastic) volatility process
        #         vol_matrix[i,t+1] = np.abs(vol_matrix[i,t] + dV)

        for i in range(self.n_paths):

            multi_norm = np.random.multivariate_normal([0,0], correlation_matrix, size = self.steps)
            dWS = np.sqrt(self.dt) * multi_norm[:,0]
            dWv = np.sqrt(self.dt) * multi_norm[:,1]
            
            # Simulating stock price dynamics 
            dS = mu * self.simulations_matrix[i, :-1] * self.dt + np.sqrt(np.maximum(0, vol_matrix[i, :-1])) * self.simulations_matrix[i, :-1] * dWS[:-1] 
            self.simulations_matrix[i, 1:] = np.cumsum(dS)

            # Simulating the volatility process
            dV = rate_of_mean_reversion * (long_term_vol - vol_matrix[i, :-1]) * self.dt + vol_vol * np.sqrt(np.maximum(0, vol_matrix[i, :-1])) * dWv[:-1] 
            vol_matrix[i, 1:] = vol_matrix[i, :-1] + dV

        return self.simulations_matrix, vol_matrix

class InterestRatesModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for interest rates modelling.
    
    """