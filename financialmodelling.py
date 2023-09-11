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
        self.simulations_matrix[:,0] = self.starting_price

class StockPriceModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for stock price modelling.
    
    """
    
    def gbm(self, vol, mu): 
        """
        Geometric Brownian Motion.
        Stock modelling under the Black - Scholes - Merton model.

        """

        if vol == None:
            vol = 1

        if mu == None:
            mu = 0
  
        for i in range(0, self.steps-1):
            # creates a path at time
            dW = np.sqrt(self.dt) * np.random.normal(0, vol, self.n_paths) # Change in price
            drift = (mu - 0.5 * vol**2) * self.dt 
            diffusion = vol * dW
            self.simulations_matrix[:, i+1] = self.simulations_matrix[:, i] * np.exp(drift + diffusion)

        return self.simulations_matrix
    
    #def heston(self,)
    

class InterestRatesModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for interest rates modelling.
    
    """