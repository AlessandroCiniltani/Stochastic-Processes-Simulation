import numpy as np
import pandas as pd

class FinancialModelling():
    """
    Class for performing Monte Carlo simulations under different financial modeling frameworks.

    The class does not allow for model calibration.
    """

    def __init__(self, n_paths, steps):

        self.n_paths = n_paths # Number of Monte Carlo paths to be generated
        self.steps = steps # Length of each Monte Carlo path
        self.dt = 1/steps # Time interval between two steps (1/365 = day in a year, 1/24 = hour in a day, ...)

        self.simulations_matrix =  np.zeros(shape=(self.n_paths,self.steps)) # empty matrix to be filled with simulated values

class StockPriceModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for stock price modelling.
    
    """
    
    def geometric_brownian_motion(self,):

class InterestRatesModelling(FinancialModelling):
    """
    This class allows to simulate the most important financial models for interest rates modelling.
    
    """