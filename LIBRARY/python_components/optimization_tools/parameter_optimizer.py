"""
Parameter Optimizer for EA_SCALPER_XAUUSD Library
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import json
from datetime import datetime
import random

class ParameterOptimizer:
    def __init__(self):
        self.objective_function = None
        self.parameter_bounds = []
        self.parameter_names = []
        self.optimization_results = []
        
    def set_objective_function(self, func):
        """Set the objective function to optimize"""
        self.objective_function = func
        
    def set_parameter_bounds(self, bounds, names):
        """Set parameter bounds and names"""
        self.parameter_bounds = bounds
        self.parameter_names = names
        
    def optimize_de(self, maxiter=100, popsize=15, seed=42):
        """Optimize using Differential Evolution algorithm"""
        if self.objective_function is None:
            raise ValueError("Objective function must be set before optimization")
            
        if not self.parameter_bounds:
            raise ValueError("Parameter bounds must be set before optimization")
            
        # Run differential evolution
        result = differential_evolution(
            self.objective_function,
            self.parameter_bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            disp=True
        )
        
        # Store results
        optimization_result = {
            'algorithm': 'Differential Evolution',
            'best_parameters': dict(zip(self.parameter_names, result.x)),
            'best_score': -result.fun,  # Negate if maximizing
            'success': result.success,
            'message': result.message,
            'nit': result.nit,
            'nfev': result.nfev,
            'optimization_date': datetime.now().isoformat()
        }
        
        self.optimization_results.append(optimization_result)
        
        print("Optimization completed:")
        print(f"Best parameters: {optimization_result['best_parameters']}")
        print(f"Best score: {optimization_result['best_score']}")
        print(f"Success: {optimization_result['success']}")
        
        return optimization_result
        
    def optimize_grid_search(self, param_grids):
        """Optimize using Grid Search"""
        if self.objective_function is None:
            raise ValueError("Objective function must be set before optimization")
            
        best_score = -np.inf
        best_params = None
        total_combinations = 1
        
        # Calculate total combinations
        for grid in param_grids:
            total_combinations *= len(grid)
            
        print(f"Performing grid search over {total_combinations} combinations...")
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grids)
        
        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            score = self.objective_function(params)
            if score > best_score:
                best_score = score
                best_params = params
                
            if (i + 1) % 100 == 0:
                print(f"Evaluated {i + 1}/{total_combinations} combinations...")
                
        # Store results
        optimization_result = {
            'algorithm': 'Grid Search',
            'best_parameters': dict(zip(self.parameter_names, best_params)),
            'best_score': best_score,
            'total_combinations': total_combinations,
            'optimization_date': datetime.now().isoformat()
        }
        
        self.optimization_results.append(optimization_result)
        
        print("Grid search completed:")
        print(f"Best parameters: {optimization_result['best_parameters']}")
        print(f"Best score: {optimization_result['best_score']}")
        
        return optimization_result
        
    def _generate_param_combinations(self, param_grids):
        """Generate all parameter combinations for grid search"""
        import itertools
        return list(itertools.product(*param_grids))
        
    def optimize_random_search(self, n_iter=1000):
        """Optimize using Random Search"""
        if self.objective_function is None:
            raise ValueError("Objective function must be set before optimization")
            
        if not self.parameter_bounds:
            raise ValueError("Parameter bounds must be set before optimization")
            
        best_score = -np.inf
        best_params = None
        
        print(f"Performing random search over {n_iter} iterations...")
        
        # Perform random search
        for i in range(n_iter):
            # Generate random parameters within bounds
            params = []
            for bound in self.parameter_bounds:
                param = random.uniform(bound[0], bound[1])
                params.append(param)
                
            # Evaluate parameters
            score = self.objective_function(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{n_iter} iterations...")
                
        # Store results
        optimization_result = {
            'algorithm': 'Random Search',
            'best_parameters': dict(zip(self.parameter_names, best_params)),
            'best_score': best_score,
            'iterations': n_iter,
            'optimization_date': datetime.now().isoformat()
        }
        
        self.optimization_results.append(optimization_result)
        
        print("Random search completed:")
        print(f"Best parameters: {optimization_result['best_parameters']}")
        print(f"Best score: {optimization_result['best_score']}")
        
        return optimization_result
        
    def optimize_bayesian(self, n_calls=50):
        """Optimize using Bayesian Optimization (simplified implementation)"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            print("scikit-optimize not installed. Please install it with: pip install scikit-optimize")
            return None
            
        if self.objective_function is None:
            raise ValueError("Objective function must be set before optimization")
            
        if not self.parameter_bounds:
            raise ValueError("Parameter bounds must be set before optimization")
            
        # Define search space
        dimensions = [Real(bound[0], bound[1], name=name) 
                     for bound, name in zip(self.parameter_bounds, self.parameter_names)]
        
        # Convert objective function for skopt
        def skopt_objective(params):
            return -self.objective_function(params)  # Negate for minimization
        
        # Run Bayesian optimization
        result = gp_minimize(
            skopt_objective,
            dimensions,
            n_calls=n_calls,
            random_state=42,
            verbose=True
        )
        
        # Store results
        optimization_result = {
            'algorithm': 'Bayesian Optimization',
            'best_parameters': dict(zip(self.parameter_names, result.x)),
            'best_score': -result.fun,  # Negate back to original scale
            'n_calls': n_calls,
            'optimization_date': datetime.now().isoformat()
        }
        
        self.optimization_results.append(optimization_result)
        
        print("Bayesian optimization completed:")
        print(f"Best parameters: {optimization_result['best_parameters']}")
        print(f"Best score: {optimization_result['best_score']}")
        
        return optimization_result
        
    def compare_optimization_results(self):
        """Compare all optimization results"""
        if not self.optimization_results:
            print("No optimization results available")
            return
            
        print("\nComparison of Optimization Results:")
        print("=" * 50)
        
        for i, result in enumerate(self.optimization_results):
            print(f"\n{i+1}. {result['algorithm']}:")
            print(f"   Best Score: {result['best_score']:.6f}")
            print(f"   Best Parameters: {result['best_parameters']}")
            print(f"   Date: {result['optimization_date']}")
            
    def save_results(self, file_path):
        """Save optimization results to file"""
        if not self.optimization_results:
            print("No optimization results to save")
            return
            
        with open(file_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        print(f"Optimization results saved to {file_path}")
        
    def load_results(self, file_path):
        """Load optimization results from file"""
        try:
            with open(file_path, 'r') as f:
                self.optimization_results = json.load(f)
            print(f"Optimization results loaded from {file_path}")
        except Exception as e:
            print(f"Error loading results: {e}")

# Example usage
def example_objective_function(params):
    """Example objective function to maximize"""
    # Simple quadratic function with multiple parameters
    # Maximum at x=1, y=2, z=3 with value = 15
    x, y, z = params
    return -(x - 1)**2 - (y - 2)**2 - (z - 3)**2 + 15

if __name__ == "__main__":
    # Example of how to use the ParameterOptimizer class
    optimizer = ParameterOptimizer()
    optimizer.set_objective_function(example_objective_function)
    optimizer.set_parameter_bounds([(0, 5), (0, 5), (0, 5)], ['x', 'y', 'z'])
    
    # Try different optimization methods
    # result_de = optimizer.optimize_de()
    # result_random = optimizer.optimize_random_search(n_iter=1000)
    # optimizer.compare_optimization_results()
    pass