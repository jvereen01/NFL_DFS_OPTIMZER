"""
Enhanced caching system for DFS optimizer performance improvements
"""
import streamlit as st
import pandas as pd
import hashlib
import pickle
from functools import wraps
from typing import Any, Dict, Optional

class PerformanceCache:
    """Enhanced caching system with granular invalidation and persistence"""
    
    def __init__(self):
        self.cache_keys = {
            'player_data': None,
            'defensive_data': None,
            'fantasy_data': None,
            'matchup_analysis': None,
            'performance_boosts': None,
            'weighted_pools': None
        }
    
    @staticmethod
    def generate_cache_key(data: Any) -> str:
        """Generate a unique cache key for data"""
        if isinstance(data, pd.DataFrame):
            # Use shape, columns, and a sample of data for DataFrame
            key_data = f"{data.shape}_{list(data.columns)}_{str(data.head().values.tobytes() if len(data) > 0 else '')}"
        elif isinstance(data, dict):
            key_data = str(sorted(data.items()))
        else:
            key_data = str(data)
        
        return hashlib.md5(key_data.encode()).hexdigest()[:12]
    
    def invalidate_dependent_caches(self, changed_key: str):
        """Invalidate caches that depend on the changed data"""
        dependencies = {
            'player_data': ['matchup_analysis', 'performance_boosts', 'weighted_pools'],
            'defensive_data': ['matchup_analysis', 'weighted_pools'],
            'fantasy_data': ['performance_boosts', 'weighted_pools'],
            'matchup_analysis': ['weighted_pools']
        }
        
        if changed_key in dependencies:
            for dependent_key in dependencies[changed_key]:
                if dependent_key in st.session_state:
                    del st.session_state[dependent_key]
                self.cache_keys[dependent_key] = None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_load_player_data():
    """Cached version of load_player_data with enhanced error handling"""
    import os
    
    target_file = 'FanDuel-NFL-2025 EDT-10 EDT-19 EDT-121559-players-list.csv'
    
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(current_dir, target_file),
        os.path.join(script_dir, target_file),
        target_file
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"Required CSV file not found: {target_file}")
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        
        # Validate required columns
        required_columns = ['Nickname', 'Position', 'Team', 'Salary', 'FPPG', 'Opponent', 'Injury Indicator', 'Id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Missing columns in CSV: {missing_columns}")
        
        # Apply filters with validation
        injury_exclusions = ['Q', 'IR', 'O', 'D']
        if 'Injury Indicator' in df.columns:
            df = df[~df['Injury Indicator'].isin(injury_exclusions)]
        
        # Salary filters with validation
        if 'Salary' in df.columns and 'Position' in df.columns:
            defense_mask = (df['Position'] == 'D') & (df['Salary'] >= 3000) & (df['Salary'] <= 5000)
            other_positions_mask = (df['Position'] != 'D') & (df['Salary'] >= 5000)
            df = df[defense_mask | other_positions_mask]
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading player data from {csv_path}: {str(e)}")

@st.cache_data(ttl=3600)
def cached_load_defensive_data():
    """Cached version of defensive data loading"""
    excel_path = find_excel_file()
    if excel_path is None:
        return None, None
        
    try:
        # Load defensive data with error handling
        passing_team_names = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                          usecols=[1], skiprows=41, nrows=32, header=None)
        passing_ypg_data = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                        usecols=[15], skiprows=41, nrows=32, header=None)
        
        rushing_team_names = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                          usecols=[1], skiprows=80, nrows=32, header=None)
        rushing_ypg_data = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                        usecols=[7], skiprows=80, nrows=32, header=None)
        
        # Process data with validation
        pass_defense = pd.DataFrame({
            'Team': passing_team_names.iloc[:, 0],
            'Pass_YPG_Allowed': pd.to_numeric(passing_ypg_data.iloc[:, 0], errors='coerce')
        }).dropna()
        
        rush_defense = pd.DataFrame({
            'Team': rushing_team_names.iloc[:, 0],
            'Rush_YPG_Allowed': pd.to_numeric(rushing_ypg_data.iloc[:, 0], errors='coerce')
        }).dropna()
        
        return pass_defense, rush_defense
    except Exception as e:
        st.warning(f"Could not load defensive data: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)
def cached_load_fantasy_data():
    """Cached version of fantasy data loading"""
    excel_path = find_excel_file()
    if excel_path is None:
        return None
        
    try:
        fantasy_data = pd.read_excel(excel_path, sheet_name="Fantasy", header=1)
        
        # Validate and clean numeric columns
        numeric_columns = ['Tgt', 'Rec', 'FDPt', 'Att_1', 'PosRank', 'TD_3', 'Yds_2']
        for col in numeric_columns:
            if col in fantasy_data.columns:
                fantasy_data[col] = pd.to_numeric(fantasy_data[col], errors='coerce')
        
        fantasy_clean = fantasy_data.dropna(subset=['FDPt']).copy()
        return fantasy_clean
    except Exception as e:
        st.warning(f"Could not load fantasy data: {str(e)}")
        return None

def find_excel_file():
    """Find NFL.xlsx file with caching"""
    import os
    possible_paths = [
        'NFL.xlsx',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NFL.xlsx')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Enhanced cache decorator for complex computations
def enhanced_cache(func):
    """Enhanced caching decorator with dependency tracking"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key from function name and arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
        
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        result = func(*args, **kwargs)
        st.session_state[cache_key] = result
        return result
    
    return wrapper

@enhanced_cache
def cached_apply_matchup_analysis(df, pass_defense, rush_defense):
    """Cached version of matchup analysis"""
    # Implementation would be moved from main file
    pass

@enhanced_cache  
def cached_create_performance_boosts(fantasy_data, wr_boost_multiplier=1.0, rb_boost_multiplier=1.0):
    """Cached version of performance boost creation"""
    # Implementation would be moved from main file
    pass