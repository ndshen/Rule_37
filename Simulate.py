import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class simulator():
    """For given random variable, the simulator will simulates the m Rule(m from 1% to 99%),
    and record the result"""
    
    def __init__(self, rv, sample_size:int, simulate_freq:int, sample_label=None, m_scale=100):
        self.figWidth = 15
        self.figHeight = 10
        self.rv = rv
        self.sample_size = sample_size
        self.simulate_freq = simulate_freq
        self.sample_label=sample_label
        self.m_scale = m_scale # determines how detail the m value goes, it can be 100 for 1%, 2%, or 1000 for 1.1%, 1.2%...
        
        self.result = np.empty(shape=[self.m_scale, self.simulate_freq]) # a big 2D array that records the results for each simulation
        self.best_values = np.empty([self.simulate_freq]) # a 1D array the records the best value for each simulation's sample
    
    def _m_rule(self, m:int, sample, scale:int):
        """It will follow the strategy as the 37 Rule, except that it can be any number from 1 to 99, not only 37."""
        actual_thresh = int(round(len(sample)*m/scale, 1))
        
        if actual_thresh == 0:
            return(sample[0])
        
        # find the best value within threshhold
        best_in_thresh = sample[0]
        for value in sample[:actual_thresh]:
            best_in_thresh = value if value > best_in_thresh else best_in_thresh
        
        # find the result
        for value in sample[actual_thresh:]:
            if value > best_in_thresh:
                return(value)
        
        # if the best value is in the threshhold unfortunately, return the last value in sample.
        return(sample[-1])

    def generate_sample(self, sample_size:int):
        """generate a sample of the random variable randomly"""
        return(self.rv.rvs(sample_size))

    def simulate(self):
        """Start simulate and record the result of different m rule, 
        with different sample under the same random variable"""
        for sim_count in range(self.simulate_freq):
            sample = self.generate_sample(self.sample_size)
            x = max(sample) # x is the best value in that sample
            self.best_values[sim_count] = x
            
            for m in range(self.m_scale):
                self.result[m, sim_count] = self._m_rule(m, sample, scale=self.m_scale)
    
    def get_m_performance(self):
        """Return the list of probability which m strategy gets the best value,
        also returns the a list of m which achieve highest performance"""
        
        def count＿prob(m_array):
            return(np.sum(m_array == self.best_values)/self.simulate_freq)
        xs_prob = np.apply_along_axis(count_prob, axis=1, arr=self.result)
        best_m = np.argwhere(xs_prob == np.amax(xs_prob)).flatten()
        
        return ((xs_prob, best_m))
    
    def plot_random_variable(self, var_num=4):
        """plot the probability density function of the random variable"""
        mean, std = self.rv.mean(), self.rv.std()
        xs = np.linspace(mean - var_num * std, mean + var_num * std, 100)
        ys = self.rv.pdf(xs)
        
        fig, ax = plt.subplots(figsize=(self.figWidth,self.figHeight))
        ax.plot(xs, ys, label="rv", linewidth=4, color='#fdc086')
        ax.set_title('pdf of the random variable')
        ax.text(0.2 , 0.9, r'$\mu={},\ \sigma={}$'.format(mean, std), ha='center', va='center', transform=ax.transAxes)
        if self.sample_label:
            ax.set_xlabel(self.sample_label)
        
        plt.show()
        
    def plot_result(self):
        fig, ax1 = plt.subplots(figsize=(self.figWidth,self.figHeight))

        xs = np.linspace(0, self.m_scale-1, self.m_scale)
        ys1 = self.result.mean(axis=1)
        
        max_m = np.argmax(ys1)
        bars_color = np.full(self.m_scale, "blue")
        bars_color[max_m] = "cyan"
        xs_ticks = np.linspace(0, self.m_scale, 6)
        xs_ticks = np.insert(xs_ticks, 0, max_m)
        xs_ticks = np.sort(xs_ticks)
        ax1.set_xticks(xs_ticks)
        
        ax1.plot(xs, ys1, 'bo', xs, ys1, 'c')
        ax1.set_xlabel('m')
        ax1.set_ylabel('mean', color='c')
        ax1.tick_params('y', colors='c')
        ax1.set_title('Simulation Result')
        
        ys2 = self.result.std(axis=1)
        ax2 = ax1.twinx()
        ax2.plot(xs, ys2, 'ro', xs, ys2, 'm')
        ax2.set_ylabel('std', color='m')
        ax2.tick_params('y', colors='m')
        
        plt.show()
    
    def plot_error(self):
        
        def error_mean_std(m_array):
            error = np.absolute(m_array - self.best_values)
            return(np.mean(error), np.std(error))
        
        error_matrix = np.apply_along_axis(error_mean_std, axis=1, arr=self.result)
        
        fig, ax1 = plt.subplots(figsize=(self.figWidth,self.figHeight))

        xs = np.linspace(0, self.m_scale-1, self.m_scale)
        ys1 = error_matrix[:, 0]
        ax1.plot(xs, ys1, 'bo', xs, ys1, 'c')
        ax1.set_xlabel('m')
        ax1.set_ylabel('mean', color='c')
        ax1.tick_params('y', colors='c')
        ax1.set_title('Simulation Error')
        
        ys2 = error_matrix[:, 1]
        ax2 = ax1.twinx()
        ax2.plot(xs, ys2, 'ro', xs, ys2, 'm')
        ax2.set_ylabel('std', color='m')
        ax2.tick_params('y', colors='m')
        
        plt.show()
    
    def plot_bingo_prob(self):
        
        xs_prob, best_m = self.get_m_performance()
            
        fig, ax1 = plt.subplots(figsize=(self.figWidth,self.figHeight))
        xs = np.linspace(0, self.m_scale-1, self.m_scale)
        
        bars_color = np.full(self.m_scale, "blue")
        bars_color[best_m] = "cyan" # the best performance m will have different color
        xs_ticks = np.linspace(0, self.m_scale, 6)
        xs_ticks = np.insert(xs_ticks, 0, best_m)
        xs_ticks = np.sort(xs_ticks)
        
        ax1.bar(xs, xs_prob, color=bars_color)
        ax1.set_xticks(xs_ticks)
        ax1.set_title('Percentage of getting the highest value')
        
        plt.show()
        
    
    def report(self):

        fig = plt.subplots(3, 1)
        # grid = GridSpec(3, 2, figure=fig)
        rv_ax = fig.add_subplot()
        result_mean_ax = fig.add_subplot()
        
        # rv_ax ===========================================================================================
        mean, std = self.rv.mean(), self.rv.std()
        var_num = 4
        xs = np.linspace(mean - var_num * std, mean + var_num * std, 100)
        ys = self.rv.pdf(xs)
        
        rv_ax.plot(xs, ys, label="rv", linewidth=4, color='#fdc086')
        rv_ax.set_title('pdf of the random variable')
        rv_ax.text(0.1 , 0.9, r'$\mu={},\ \sigma={}$'.format(mean, std), ha='center', va='center', transform=rv_ax.transAxes)
        if self.sample_label:
            rv_ax.set_xlabel(self.sample_label)
    
        # result_mean_ax ==================================================================================
        x = np.arange(self.m_scale)
        x_result = self.result.mean(axis=1)
        result_mean_ax.bar(x, x_result, align='center')
        result_mean_ax.set_xticks(np.linspace(0, self.m_scale-1, 10))
        
        
        plt.show()
        
    def run(self, rv_plot=False, error_plot=False):
        
        if rv_plot:
            self.plot_random_variable()
        self.simulate()
        self.plot_result()
        if error_plot:
            self.plot_error()
        self.plot_bingo_prob()
        


