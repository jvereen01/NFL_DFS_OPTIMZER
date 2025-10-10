"""
Memory optimization module for DFS optimizer
Implements chunking, efficient data structures, and memory monitoring
"""
import pandas as pd
import numpy as np
import psutil
import gc
from typing import Iterator, List, Dict, Any, Optional, Tuple
from functools import wraps
import warnings
from datetime import datetime

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit"""
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory > self.max_memory_mb
    
    def get_memory_report(self) -> Dict[str, float]:
        """Get comprehensive memory report"""
        current = self.get_memory_usage()
        return {
            'current_mb': current,
            'peak_mb': self.peak_memory,
            'initial_mb': self.initial_memory,
            'increase_mb': current - self.initial_memory,
            'limit_mb': self.max_memory_mb,
            'usage_percent': (current / self.max_memory_mb) * 100
        }

def memory_efficient(func):
    """Decorator to monitor and optimize memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        
        try:
            result = func(*args, **kwargs)
            
            # Force garbage collection
            gc.collect()
            
            # Check final memory usage
            if monitor.check_memory_limit():
                warnings.warn(f"Function {func.__name__} exceeded memory limit: {monitor.get_memory_usage():.1f}MB")
            
            return result
            
        except MemoryError:
            gc.collect()  # Emergency cleanup
            raise MemoryError(f"Function {func.__name__} ran out of memory. Consider reducing data size or using chunking.")
    
    return wrapper

class DataFrameOptimizer:
    """Optimize DataFrame memory usage and operations"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # Optimize numeric columns
            if col_type in ['int64', 'int32']:
                # Downcast integers
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            
            elif col_type in ['float64', 'float32']:
                # Downcast floats
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            
            elif col_type == 'object':
                # Convert string columns to categorical if beneficial
                unique_count = optimized_df[col].nunique()
                total_count = len(optimized_df)
                
                if unique_count / total_count < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage information for DataFrame"""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / 1024 / 1024,
            'by_column': {col: mem / 1024 / 1024 for col, mem in memory_usage.items()},
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict()
        }
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        """Split DataFrame into chunks for memory-efficient processing"""
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start:start + chunk_size]

class OptimizedLineupGenerator:
    """Memory-optimized lineup generation"""
    
    def __init__(self, memory_limit_mb: int = 512):
        self.memory_monitor = MemoryMonitor(memory_limit_mb)
        self.chunk_size = 1000
        
    @memory_efficient
    def generate_lineups_chunked(self, df: pd.DataFrame, weighted_pools: Dict, 
                               num_simulations: int, **kwargs) -> List[Tuple]:
        """Generate lineups in chunks to manage memory"""
        
        # Optimize input data
        optimized_df = DataFrameOptimizer.optimize_dtypes(df)
        optimized_pools = {}
        
        for pos, pool in weighted_pools.items():
            optimized_pools[pos] = DataFrameOptimizer.optimize_dtypes(pool)
        
        # Calculate optimal chunk size based on memory
        available_memory = self.memory_monitor.max_memory_mb - self.memory_monitor.get_memory_usage()
        estimated_lineup_memory = self._estimate_lineup_memory(optimized_df)
        optimal_chunk_size = max(100, int(available_memory / estimated_lineup_memory))
        
        # Generate lineups in chunks
        all_lineups = []
        processed = 0
        
        while processed < num_simulations:
            current_chunk_size = min(optimal_chunk_size, num_simulations - processed)
            
            # Generate chunk of lineups
            chunk_lineups = self._generate_lineup_chunk(
                optimized_df, optimized_pools, current_chunk_size, **kwargs
            )
            
            all_lineups.extend(chunk_lineups)
            processed += current_chunk_size
            
            # Memory check and cleanup
            if self.memory_monitor.check_memory_limit():
                gc.collect()
                if self.memory_monitor.check_memory_limit():
                    # Reduce chunk size if still over limit
                    optimal_chunk_size = max(50, optimal_chunk_size // 2)
        
        return all_lineups
    
    def _estimate_lineup_memory(self, df: pd.DataFrame) -> float:
        """Estimate memory usage per lineup in MB"""
        # Rough estimation based on DataFrame size and typical lineup structure
        row_memory = df.memory_usage(deep=True).sum() / len(df) if len(df) > 0 else 1000
        lineup_memory_bytes = row_memory * 9  # 9 players per lineup
        return lineup_memory_bytes / 1024 / 1024  # Convert to MB
    
    def _generate_lineup_chunk(self, df: pd.DataFrame, weighted_pools: Dict, 
                             chunk_size: int, **kwargs) -> List[Tuple]:
        """Generate a chunk of lineups"""
        # Placeholder for actual lineup generation logic
        # This would integrate with your existing generate_lineups function
        chunk_lineups = []
        
        # Simplified example - replace with actual logic
        for _ in range(chunk_size):
            # Generate single lineup (simplified)
            lineup_data = self._generate_single_lineup(df, weighted_pools, **kwargs)
            if lineup_data:
                chunk_lineups.append(lineup_data)
        
        return chunk_lineups
    
    def _generate_single_lineup(self, df: pd.DataFrame, weighted_pools: Dict, **kwargs):
        """Generate a single lineup efficiently"""
        # Placeholder for single lineup generation
        # This would be optimized version of your lineup generation logic
        pass

class EfficientDataLoader:
    """Memory-efficient data loading and processing"""
    
    @staticmethod
    @memory_efficient
    def load_csv_optimized(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """Load large CSV files efficiently"""
        
        # Read in chunks and optimize each chunk
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            optimized_chunk = DataFrameOptimizer.optimize_dtypes(chunk)
            chunks.append(optimized_chunk)
        
        # Combine chunks
        result_df = pd.concat(chunks, ignore_index=True)
        
        # Final optimization
        return DataFrameOptimizer.optimize_dtypes(result_df)
    
    @staticmethod
    @memory_efficient  
    def load_excel_optimized(file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Load Excel files with memory optimization"""
        
        # Use read_excel with optimizations
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        
        # Optimize data types
        return DataFrameOptimizer.optimize_dtypes(df)

class StreamingProcessor:
    """Process data in streams to minimize memory usage"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def process_player_data_streaming(self, df: pd.DataFrame, 
                                    processing_func: callable) -> pd.DataFrame:
        """Process player data in streaming fashion"""
        
        results = []
        
        for chunk in DataFrameOptimizer.chunk_dataframe(df, self.chunk_size):
            processed_chunk = processing_func(chunk)
            results.append(processed_chunk)
            
            # Memory cleanup between chunks
            gc.collect()
        
        return pd.concat(results, ignore_index=True)
    
    def apply_matchup_analysis_streaming(self, df: pd.DataFrame, 
                                       pass_defense: pd.DataFrame, 
                                       rush_defense: pd.DataFrame) -> pd.DataFrame:
        """Apply matchup analysis in streaming chunks"""
        
        def process_chunk(chunk):
            # Simplified matchup analysis for chunk
            chunk['Matchup_Quality'] = 'Good Target'  # Default
            chunk['Adjusted_FPPG'] = chunk['FPPG']
            return chunk
        
        return self.process_player_data_streaming(df, process_chunk)

class MemoryProfiler:
    """Profile memory usage of functions and operations"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_function(self, func_name: str):
        """Decorator to profile function memory usage"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                monitor = MemoryMonitor()
                start_memory = monitor.get_memory_usage()
                start_time = datetime.now()
                
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                end_memory = monitor.get_memory_usage()
                
                self.profiles[func_name] = {
                    'memory_start_mb': start_memory,
                    'memory_end_mb': end_memory,
                    'memory_increase_mb': end_memory - start_memory,
                    'execution_time_seconds': (end_time - start_time).total_seconds(),
                    'peak_memory_mb': monitor.peak_memory
                }
                
                return result
            return wrapper
        return decorator
    
    def get_profile_report(self) -> str:
        """Generate memory profile report"""
        if not self.profiles:
            return "No profiling data available"
        
        report = ["Memory Profiling Report", "=" * 30]
        
        for func_name, profile in self.profiles.items():
            report.append(f"\n{func_name}:")
            report.append(f"  Memory increase: {profile['memory_increase_mb']:.2f} MB")
            report.append(f"  Peak memory: {profile['peak_memory_mb']:.2f} MB") 
            report.append(f"  Execution time: {profile['execution_time_seconds']:.2f} seconds")
        
        return "\n".join(report)

# Utility functions for memory optimization
def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage of DataFrame by optimizing dtypes"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    optimized_df = DataFrameOptimizer.optimize_dtypes(df)
    
    end_mem = optimized_df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return optimized_df

def memory_usage_check(threshold_mb: float = 500) -> callable:
    """Decorator to check memory usage before function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_memory = MemoryMonitor().get_memory_usage()
            
            if current_memory > threshold_mb:
                warnings.warn(f"High memory usage detected ({current_memory:.1f}MB) before executing {func.__name__}")
                gc.collect()  # Try to free some memory
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Configuration for memory optimization
MEMORY_CONFIG = {
    'chunk_size': 1000,
    'max_memory_mb': 1024,
    'enable_optimization': True,
    'enable_profiling': False,
    'gc_frequency': 100  # Run garbage collection every N operations
}