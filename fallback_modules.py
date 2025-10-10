"""
Fallback module loader - provides dummy implementations if enhanced modules aren't available
"""

# Dummy implementations for enhanced features
class DummyLogger:
    def info(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def log_performance(self, *args, **kwargs): pass
    def log_user_activity(self, *args, **kwargs): pass

class DummyConfig:
    def __init__(self):
        self.optimization = type('obj', (object,), {
            'num_simulations': 10000,
            'stack_probability': 0.80,
            'elite_target_boost': 0.45,
            'great_target_boost': 0.25
        })()

class DummyConfigManager:
    def load_config(self): return DummyConfig()
    def save_config(self, config): pass

class DummyConfigUI:
    def __init__(self, manager): pass
    def render_settings_panel(self): return {}

# Dummy functions
def init_logging(): return DummyLogger()
def get_logger(): return DummyLogger()
def log_info(*args): pass
def log_error(*args): pass
def performance_track(*args): 
    def decorator(func): return func
    return decorator

def log_operation(*args):
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    return DummyContext()

def load_config(): return DummyConfig()
def get_config_manager(): return DummyConfigManager()

def cached_load_player_data(): raise ImportError("Enhanced caching not available")
def cached_load_defensive_data(): raise ImportError("Enhanced caching not available")  
def cached_load_fantasy_data(): raise ImportError("Enhanced caching not available")

class DataValidator:
    def validate_player_data(self, df): return df, {'data_quality_score': 100}
    def generate_data_quality_report(self, results): return "No validation available"

def reduce_memory_usage(df, verbose=False): return df

class AdvancedAnalytics:
    def generate_ownership_projections(self, *args): return None
    def generate_lineup_performance_insights(self, *args): return {}
    def create_advanced_visualizations(self, *args): return {}
    def generate_roi_projections(self, *args): return {'avg_roi': 0, 'cash_rate': 0, 'top_1_percent_rate': 0}

class LineupExporter:
    def get_supported_platforms(self): return ['fanduel']
    def export_lineups(self, *args): return ""

class ExportManager:
    def export_to_multiple_platforms(self, *args): return {}

# Make all imports available
globals().update({
    'cached_load_player_data': cached_load_player_data,
    'cached_load_defensive_data': cached_load_defensive_data,
    'cached_load_fantasy_data': cached_load_fantasy_data,
    'DataValidator': DataValidator,
    'AdvancedAnalytics': AdvancedAnalytics,
    'load_config': load_config,
    'ConfigUI': DummyConfigUI,
    'get_config_manager': get_config_manager,
    'reduce_memory_usage': reduce_memory_usage,
    'LineupExporter': LineupExporter,
    'ExportManager': ExportManager,
    'init_logging': init_logging,
    'performance_track': performance_track,
    'log_info': log_info,
    'log_error': log_error,
    'get_logger': get_logger,
    'log_operation': log_operation
})