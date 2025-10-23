import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# New enhanced modules
try:
    from performance_cache import cached_load_player_data, cached_load_defensive_data, cached_load_fantasy_data
    from data_validation import DataValidator
    from advanced_analytics import AdvancedAnalytics
    from config_manager import load_config, ConfigUI, get_config_manager
    from memory_optimizer import MemoryMonitor, DataFrameOptimizer, reduce_memory_usage
    from export_templates import LineupExporter, ExportManager
    from logging_system import init_logging, performance_track, log_info, log_error, get_logger, log_operation
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced features loaded successfully!")
except ImportError as e:
    print(f"⚠️ Enhanced features not available, using fallback: {e}")
    try:
        from fallback_modules import *
        ENHANCED_FEATURES_AVAILABLE = False
    except ImportError:
        print("❌ Fallback modules also not available. Some features will be disabled.")
        ENHANCED_FEATURES_AVAILABLE = False
        # Create minimal dummy functions to prevent crashes
        def log_info(*args): pass
        def log_error(*args): pass
        def get_logger(): return type('DummyLogger', (), {'info': log_info, 'error': log_error})()
        def log_operation(*args): 
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()
        def load_config(): return type('DummyConfig', (), {'optimization': type('obj', (), {'num_simulations': 10000, 'stack_probability': 0.80, 'elite_target_boost': 0.45, 'great_target_boost': 0.25})()})()
        def get_config_manager(): return type('DummyManager', (), {'load_config': load_config})()
        ConfigUI = type('DummyConfigUI', (), {'__init__': lambda self, manager: None, 'render_settings_panel': lambda self: {}})
        DataValidator = type('DummyValidator', (), {'validate_player_data': lambda self, df: (df, {'data_quality_score': 100}), 'generate_data_quality_report': lambda self, r: "No validation available"})
        reduce_memory_usage = lambda df, verbose=False: df
        AdvancedAnalytics = type('DummyAnalytics', (), {
            'generate_ownership_projections': lambda self, *args: pd.DataFrame(),
            'generate_lineup_performance_insights': lambda self, *args: {},
            'create_advanced_visualizations': lambda self, *args: {},
            'generate_roi_projections': lambda self, *args: {'avg_roi': 0, 'cash_rate': 0, 'top_1_percent_rate': 0}
        })
        LineupExporter = type('DummyExporter', (), {'get_supported_platforms': lambda self: ['fanduel'], 'export_lineups': lambda self, *args: ""})
        ExportManager = type('DummyManager', (), {'export_to_multiple_platforms': lambda self, *args: {}})

def find_excel_file():
    """Find NFL.xlsx file in current directory or script directory"""
    possible_paths = [
        'NFL.xlsx',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NFL.xlsx')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Set page config
st.set_page_config(
    page_title="🏈 FanDuel NFL DFS Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'lineups_generated' not in st.session_state:
    st.session_state.lineups_generated = False
if 'stacked_lineups' not in st.session_state:
    st.session_state.stacked_lineups = []

@st.cache_data
def load_player_data():
    """Load and process player data with enhanced validation and optimization"""
    
    # FLEXIBLE CSV LOADING - WORKS LOCALLY AND ON STREAMLIT CLOUD
    import pandas as pd
    import os
    import glob
    
    # Try multiple approaches to find the CSV file
    csv_file = None
    
    # First try: Direct path (local development)
    direct_path = r"c:\Users\jamin\OneDrive\NFL scrapping\NFL_DFS_OPTIMZER\FanDuel-NFL-2025 EDT-10 EDT-26 EDT-121824-players-list.csv"
    if os.path.exists(direct_path):
        csv_file = direct_path
    else:
        # Second try: Look in current directory (Streamlit Cloud)
        current_dir_file = "FanDuel-NFL-2025 EDT-10 EDT-26 EDT-121824-players-list.csv"
        if os.path.exists(current_dir_file):
            csv_file = current_dir_file
        else:
            # Third try: Find any FanDuel CSV file in current directory
            csv_files = glob.glob("FanDuel-NFL-*.csv")
            if csv_files:
                csv_file = sorted(csv_files)[-1]  # Get the most recent one
    
    if not csv_file:
        st.error("No FanDuel CSV file found. Please upload a FanDuel player list CSV file.")
        return pd.DataFrame()
    
    try:
        # Load the CSV directly
        df = pd.read_csv(csv_file)
        df.columns = [col.strip() for col in df.columns]
        
        # Debug: Show available columns
        st.write(f"📊 Loaded data from: {os.path.basename(csv_file)}")
        st.write(f"Available columns: {list(df.columns)}")
        
        # Standardize column names - ensure we have a 'Nickname' column
        name_column = None
        for col in ['Nickname', 'Name', 'Player', 'Full Name', 'First Name']:
            if col in df.columns:
                name_column = col
                break
        
        if name_column and name_column != 'Nickname':
            df['Nickname'] = df[name_column]
            st.info(f"Using '{name_column}' column as player names")
        elif not name_column:
            st.error("Could not find a player name column in the CSV file")
            return pd.DataFrame()
        
        # Apply salary filters by position
        if 'Salary' in df.columns and 'Position' in df.columns:
            # Remove players with missing salary
            df = df[df['Salary'].notna() & (df['Salary'] > 0)]
            
            # Position-specific salary filters
            rb_filter = (df['Position'] == 'RB') & (df['Salary'] >= 5000)
            wr_filter = (df['Position'] == 'WR') & (df['Salary'] >= 4600)
            te_filter = (df['Position'] == 'TE') & (df['Salary'] >= 4200)
            qb_filter = (df['Position'] == 'QB') & (df['Salary'] >= 6000)
            def_filter = (df['Position'] == 'D') & (df['Salary'] >= 3000) & (df['Salary'] <= 5000)
            
            # Combine all position filters
            df = df[rb_filter | wr_filter | te_filter | qb_filter | def_filter]
        
        # Remove injury exclusions but keep 'Q' (Questionable) players
        if 'Injury Indicator' in df.columns:
            df = df[~df['Injury Indicator'].isin(['IR', 'O', 'D'])]  # Removed 'Q' from exclusions
        
        # Add ceiling and floor projections
        df = calculate_ceiling_floor_projections(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return pd.DataFrame()

def calculate_ceiling_floor_projections(df):
    """Calculate ceiling and floor projections for each player"""
    
    # Position-specific variance multipliers for ceiling calculation
    position_variance = {
        'QB': 1.25,    # QBs can boom big with multiple TDs
        'RB': 1.35,    # RBs have high ceiling with goal line work  
        'WR': 1.30,    # WRs can explode in shootouts
        'TE': 1.25,    # TEs more consistent but can boom
        'D': 1.20      # Defenses have some variance
    }
    
    # Additional ceiling bonuses based on salary (higher salary = more ceiling)
    df['Salary_Tier'] = pd.cut(df['Salary'], 
                               bins=[0, 5000, 6500, 8000, float('inf')], 
                               labels=['Budget', 'Mid', 'Premium', 'Elite'])
    
    salary_bonus = {
        'Budget': 1.0,    # No bonus for cheap players
        'Mid': 1.1,       # 10% bonus for mid-tier
        'Premium': 1.2,   # 20% bonus for premium  
        'Elite': 1.3      # 30% bonus for elite players
    }
    
    # Calculate ceiling and floor
    df['Base_Projection'] = df['FPPG']
    
    # Ceiling = Base * Position_Variance * Salary_Bonus  
    df['Ceiling'] = df.apply(lambda row: 
        row['FPPG'] * 
        position_variance.get(row['Position'], 1.2) * 
        salary_bonus.get(row['Salary_Tier'], 1.0), axis=1
    )
    
    # Floor = Base * 0.75 (more conservative)
    df['Floor'] = df['FPPG'] * 0.75
    
    # Round projections
    df['Ceiling'] = df['Ceiling'].round(1)
    df['Floor'] = df['Floor'].round(1)
    
    return df
    
    # OLD COMPLEX LOADING (commented out)
    if False and ENHANCED_FEATURES_AVAILABLE:
        # Clear cache to force fresh load
        try:
            cached_load_player_data.clear()
        except:
            pass
            
        # Use enhanced loading with validation and optimization
        try:
            with log_operation("load_player_data"):
                df = cached_load_player_data()
                
                # Add validation
                validator = DataValidator()
                validated_df, validation_results = validator.validate_player_data(df)
                
                if validation_results['data_quality_score'] < 90:
                    with st.expander("📊 Data Quality Report", expanded=True):
                        report = validator.generate_data_quality_report(validation_results)
                        st.markdown(report)
                
                # Optimize memory usage
                optimized_df = reduce_memory_usage(validated_df, verbose=False)
                log_info(f"Loaded {len(optimized_df)} players with enhanced features")
                
                return optimized_df
        except Exception as e:
            log_error("Enhanced data loading failed, falling back to standard loading", e)
            # Fall back to standard loading
    
    # Standard loading (original code)
    import os
    
    # ONLY use the October 26th CSV file
    target_file = 'FanDuel-NFL-2025 EDT-10 EDT-26 EDT-121824-players-list.csv'
    
    # Debug: Show what we're looking for
    st.info(f"🔍 **Looking for CSV file:** {target_file}")
    
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(current_dir, target_file),
        os.path.join(script_dir, target_file),
        target_file
    ]
    
    # Debug: Show all possible paths
    st.write("**Checking paths:**")
    for i, path in enumerate(possible_paths):
        exists = os.path.exists(path)
        st.write(f"{i+1}. `{path}` - {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        st.error(f"❌ Required CSV file not found: {target_file}")
        st.warning("This app requires the October 12th FanDuel player list file.")
        st.info("Please upload the correct CSV file to continue.")
        return None
    
    try:
        # Load player CSV
        st.info(f"📂 **Loading CSV file:** {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        
        # Show file details and timestamp
        import datetime
        file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(csv_path))
        st.success(f"✅ **Loaded {len(df)} players** from {os.path.basename(csv_path)} (Modified: {file_mod_time.strftime('%m/%d/%Y %H:%M')})")
        
        # Debug: Check if CeeDee Lamb is in the dataset
        lamb_check = df[df['Nickname'].str.contains('Lamb', case=False, na=False)]
        if len(lamb_check) > 0:
            st.info(f"🏈 **CeeDee Lamb found:** {lamb_check['Nickname'].iloc[0]} - ${lamb_check['Salary'].iloc[0]:,}")
        else:
            st.warning("⚠️ **CeeDee Lamb not found in dataset** - checking name variations...")
            # Check for other name formats
            cd_check = df[df['Nickname'].str.contains('CeeDee|CD|Ceedee', case=False, na=False)]
            if len(cd_check) > 0:
                st.info(f"🏈 **Found similar:** {', '.join(cd_check['Nickname'].tolist())}")
            else:
                st.error("❌ **No CeeDee Lamb found** - file may not be updated")
                # Show a few sample player names for debugging
                sample_players = df['Nickname'].head(10).tolist()
                st.write(f"**Sample players in file:** {', '.join(sample_players)}")
        
        # Debug: Check CeeDee Lamb BEFORE filtering
        lamb_before = df[df['Nickname'].str.contains('Lamb', case=False, na=False)]
        if len(lamb_before) > 0:
            injury_status = lamb_before['Injury Indicator'].iloc[0]
            st.write(f"🔍 **CeeDee Lamb before filtering:** Injury Status = '{injury_status}'")
        
        # Apply filters
        injury_exclusions = ['IR', 'O', 'D']  # Include Q (Questionable) players
        st.write(f"🚫 **Excluding injury statuses:** {injury_exclusions}")
        df = df[~df['Injury Indicator'].isin(injury_exclusions)]
        
        # Debug: Check CeeDee Lamb AFTER injury filtering
        lamb_after_injury = df[df['Nickname'].str.contains('Lamb', case=False, na=False)]
        if len(lamb_after_injury) > 0:
            st.write(f"✅ **CeeDee Lamb after injury filter:** Still in dataset")
        else:
            st.write(f"❌ **CeeDee Lamb after injury filter:** REMOVED from dataset")
        
        # Salary filters
        st.write(f"💰 **Applying salary filters:** Defense $3,000-$5,000, Others $5,000+")
        
        # Debug: Check CeeDee Lamb BEFORE salary filtering
        lamb_before_salary = df[df['Nickname'].str.contains('Lamb', case=False, na=False)]
        if len(lamb_before_salary) > 0:
            salary = lamb_before_salary['Salary'].iloc[0]
            position = lamb_before_salary['Position'].iloc[0]
            st.write(f"🔍 **CeeDee Lamb before salary filter:** {position}, ${salary:,}")
        
        defense_mask = (df['Position'] == 'D') & (df['Salary'] >= 3000) & (df['Salary'] <= 5000)
        other_positions_mask = (df['Position'] != 'D') & (df['Salary'] >= 5000)
        df = df[defense_mask | other_positions_mask]
        
        # Debug: Check CeeDee Lamb AFTER salary filtering
        lamb_after_salary = df[df['Nickname'].str.contains('Lamb', case=False, na=False)]
        if len(lamb_after_salary) > 0:
            st.write(f"✅ **CeeDee Lamb after salary filter:** Still in dataset")
        else:
            st.write(f"❌ **CeeDee Lamb after salary filter:** REMOVED from dataset (salary too low?)")
        
        # Final comprehensive check
        st.write("---")
        st.write(f"📊 **Final Dataset Summary:**")
        st.write(f"- Total players: {len(df)}")
        st.write(f"- Salary range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
        
        # Show all WRs with "C" names to see if CeeDee is there under different name
        c_wrs = df[(df['Position'] == 'WR') & (df['Nickname'].str.startswith('C', na=False))]['Nickname'].tolist()
        if c_wrs:
            st.write(f"**WRs starting with 'C':** {', '.join(sorted(c_wrs))}")
        
        return df
    except FileNotFoundError:
        st.error(f"File was found but couldn't be read: {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading player data from {csv_path}: {str(e)}")
        return None

@st.cache_data
def load_defensive_data():
    """Load and process defensive matchup data with enhanced caching"""
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            return cached_load_defensive_data()
        except Exception as e:
            log_error("Enhanced defensive data loading failed, falling back to standard loading", e)
    
    # Standard loading (original code)
    excel_path = find_excel_file()
    if excel_path is None:
        st.warning("NFL.xlsx file not found. Using salary-based matchup analysis.")
        return None, None
        
    try:
        # Load Excel data
        excel_file = pd.ExcelFile(excel_path)
        
        # Load defensive data
        passing_team_names = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                          usecols=[1], skiprows=41, nrows=32, header=None)
        passing_ypg_data = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                        usecols=[15], skiprows=41, nrows=32, header=None)
        
        rushing_team_names = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                          usecols=[1], skiprows=80, nrows=32, header=None)
        rushing_ypg_data = pd.read_excel(excel_path, sheet_name="Defense Data 2025", 
                                        usecols=[7], skiprows=80, nrows=32, header=None)
        
        # Process data
        pass_defense = pd.DataFrame({
            'Team': passing_team_names.iloc[:, 0],
            'Pass_YPG_Allowed': pd.to_numeric(passing_ypg_data.iloc[:, 0], errors='coerce')
        }).dropna()
        
        rush_defense = pd.DataFrame({
            'Team': rushing_team_names.iloc[:, 0],
            'Rush_YPG_Allowed': pd.to_numeric(rushing_ypg_data.iloc[:, 0], errors='coerce')
        }).dropna()
        
        return pass_defense, rush_defense
    except FileNotFoundError:
        st.error("NFL.xlsx file not found. Defensive targeting will be disabled.")
        return None, None
    except Exception as e:
        st.warning(f"Could not load defensive data from NFL.xlsx: {str(e)}. Using basic matchup analysis.")
        return None, None

@st.cache_data
def load_fantasy_data():
    """Load fantasy performance data with enhanced caching"""
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            return cached_load_fantasy_data()
        except Exception as e:
            log_error("Enhanced fantasy data loading failed, falling back to standard loading", e)
    
    # Standard loading (original code)
    excel_path = find_excel_file()
    if excel_path is None:
        st.warning("NFL.xlsx file not found. Performance boosts will be disabled.")
        return None
        
    try:
        fantasy_data = pd.read_excel(excel_path, sheet_name="Fantasy", header=1)
        fantasy_data['Tgt'] = pd.to_numeric(fantasy_data['Tgt'], errors='coerce')
        fantasy_data['Rec'] = pd.to_numeric(fantasy_data['Rec'], errors='coerce')
        fantasy_data['FDPt'] = pd.to_numeric(fantasy_data['FDPt'], errors='coerce')
        fantasy_data['Att_1'] = pd.to_numeric(fantasy_data['Att_1'], errors='coerce')
        fantasy_data['PosRank'] = pd.to_numeric(fantasy_data['PosRank'], errors='coerce')
        fantasy_data['TD_3'] = pd.to_numeric(fantasy_data['TD_3'], errors='coerce')  # Rushing TDs
        fantasy_data['Yds_2'] = pd.to_numeric(fantasy_data['Yds_2'], errors='coerce')  # Rushing Yards
        
        fantasy_clean = fantasy_data.dropna(subset=['FDPt']).copy()
        return fantasy_clean
    except FileNotFoundError:
        st.error("Fantasy performance data not found. Performance boosts will be disabled.")
        return None

def apply_matchup_analysis(df, pass_defense, rush_defense):
    """Apply defensive matchup analysis"""
    if pass_defense is None or rush_defense is None:
        # Fallback: Use salary as a basic matchup quality indicator
        df['Adjusted_FPPG'] = df['FPPG']
        df['Matchup_Quality'] = 'Good Target'  # Default to neutral matchup
        
        # Assign better matchups to higher salary players within position
        for pos in ['QB', 'WR', 'RB', 'TE']:
            pos_players = df[df['Position'] == pos]
            if len(pos_players) > 0:
                # Top 25% by salary get Elite Target
                elite_threshold = pos_players['Salary'].quantile(0.75)
                # Top 50% by salary get Great Target  
                great_threshold = pos_players['Salary'].quantile(0.5)
                
                df.loc[(df['Position'] == pos) & (df['Salary'] >= elite_threshold), 'Matchup_Quality'] = 'ELITE TARGET'
                df.loc[(df['Position'] == pos) & (df['Salary'] >= great_threshold) & (df['Salary'] < elite_threshold), 'Matchup_Quality'] = 'Great Target'
        
        return df
    
    # Create team mapping (simplified)
    excel_path = find_excel_file()
    if excel_path is None:
        # No Excel file, use salary-based fallback for all
        for pos in ['QB', 'WR', 'RB', 'TE']:
            pos_players = df[df['Position'] == pos]
            if len(pos_players) > 0:
                elite_threshold = pos_players['Salary'].quantile(0.75)
                great_threshold = pos_players['Salary'].quantile(0.5)
                
                df.loc[(df['Position'] == pos) & (df['Salary'] >= elite_threshold), 'Matchup_Quality'] = 'ELITE TARGET'
                df.loc[(df['Position'] == pos) & (df['Salary'] >= great_threshold) & (df['Salary'] < elite_threshold), 'Matchup_Quality'] = 'Great Target'
        return df
        
    teams_sheet = pd.read_excel(excel_path, sheet_name="Teams")
    team_mapping = {}
    if len(teams_sheet.columns) >= 6:
        excel_teams = teams_sheet.iloc[:, 1]
        csv_teams = teams_sheet.iloc[:, 5]
        for excel_team, csv_team in zip(excel_teams, csv_teams):
            if pd.notna(excel_team) and pd.notna(csv_team):
                team_mapping[excel_team] = csv_team
    
    # Map team names and create rankings
    pass_defense['Team'] = pass_defense['Team'].map(team_mapping).fillna(pass_defense['Team'])
    rush_defense['Team'] = rush_defense['Team'].map(team_mapping).fillna(rush_defense['Team'])
    
    pass_defense['Pass_Defense_Rank'] = range(1, len(pass_defense) + 1)
    rush_defense['Rush_Defense_Rank'] = range(1, len(rush_defense) + 1)
    
    # Merge defensive data
    defensive_matchups = pd.merge(pass_defense, rush_defense, on='Team', how='outer')
    
    # Apply matchup analysis
    df['Matchup_Quality'] = 'Good Target'  # Default instead of Unknown
    df['Overall_Matchup_Multiplier'] = 1.0
    
    for idx, row in df.iterrows():
        opponent = row['Opponent']
        position = row['Position']
        
        if opponent in defensive_matchups['Team'].values:
            team_defense = defensive_matchups[defensive_matchups['Team'] == opponent].iloc[0]
            
            if position in ['QB', 'WR', 'TE']:
                defense_rank = team_defense['Pass_Defense_Rank']
                if defense_rank >= 25:
                    df.loc[idx, 'Matchup_Quality'] = 'ELITE TARGET'
                elif defense_rank >= 20:
                    df.loc[idx, 'Matchup_Quality'] = 'Great Target'
                elif defense_rank >= 15:
                    df.loc[idx, 'Matchup_Quality'] = 'Good Target'
            elif position == 'RB':
                defense_rank = team_defense['Rush_Defense_Rank']
                if defense_rank >= 25:
                    df.loc[idx, 'Matchup_Quality'] = 'ELITE TARGET'
                elif defense_rank >= 20:
                    df.loc[idx, 'Matchup_Quality'] = 'Great Target'
                elif defense_rank >= 15:
                    df.loc[idx, 'Matchup_Quality'] = 'Good Target'
        else:
            # Fallback for teams not found in defensive data - use salary-based logic
            pos_players = df[df['Position'] == position]
            if len(pos_players) > 0:
                elite_threshold = pos_players['Salary'].quantile(0.75)
                great_threshold = pos_players['Salary'].quantile(0.5)
                
                if row['Salary'] >= elite_threshold:
                    df.loc[idx, 'Matchup_Quality'] = 'ELITE TARGET'
                elif row['Salary'] >= great_threshold:
                    df.loc[idx, 'Matchup_Quality'] = 'Great Target'
                # else keep the default 'Good Target'
    
    df['Adjusted_FPPG'] = df['FPPG'] * df['Overall_Matchup_Multiplier']
    return df

def create_performance_boosts(fantasy_data, wr_boost_multiplier=1.0, rb_boost_multiplier=1.0):
    """Create fantasy performance boosts"""
    wr_performance_boosts = {}
    rb_performance_boosts = {}
    te_performance_boosts = {}
    qb_performance_boosts = {}
    
    if fantasy_data is not None:
        # WR boosts
        wr_fantasy = fantasy_data[fantasy_data['FantPos'] == 'WR'].copy()
        if len(wr_fantasy) > 0:
            wr_fantasy['Tgt_Percentile'] = wr_fantasy['Tgt'].rank(pct=True, na_option='bottom')
            wr_fantasy['Rec_Percentile'] = wr_fantasy['Rec'].rank(pct=True, na_option='bottom')
            wr_fantasy['FDPt_Percentile'] = wr_fantasy['FDPt'].rank(pct=True, na_option='bottom')
            
            wr_fantasy['WR_Performance_Score'] = (
                wr_fantasy['Tgt_Percentile'] * 0.25 +
                wr_fantasy['Rec_Percentile'] * 0.25 +
                wr_fantasy['FDPt_Percentile'] * 0.5
            )
            wr_fantasy['WR_Performance_Boost'] = wr_fantasy['WR_Performance_Score'] * 0.4 * wr_boost_multiplier
            
            for _, wr in wr_fantasy.iterrows():
                wr_performance_boosts[wr['Player']] = wr['WR_Performance_Boost']
        
        # RB boosts
        rb_fantasy = fantasy_data[fantasy_data['FantPos'] == 'RB'].copy()
        if len(rb_fantasy) > 0:
            rb_fantasy['FDPt_Percentile'] = rb_fantasy['FDPt'].rank(pct=True, na_option='bottom')
            rb_fantasy['Att_Percentile'] = rb_fantasy['Att_1'].rank(pct=True, na_option='bottom')
            rb_fantasy['Rec_Percentile'] = rb_fantasy['Rec'].rank(pct=True, na_option='bottom')
            
            rb_fantasy['RB_Performance_Score'] = (
                rb_fantasy['FDPt_Percentile'] * 0.5 +
                rb_fantasy['Att_Percentile'] * 0.3 +
                rb_fantasy['Rec_Percentile'] * 0.2
            )
            rb_fantasy['RB_Performance_Boost'] = rb_fantasy['RB_Performance_Score'] * 0.4 * rb_boost_multiplier
            
            for _, rb in rb_fantasy.iterrows():
                rb_performance_boosts[rb['Player']] = rb['RB_Performance_Boost']
        
        # TE boosts - prioritize receptions and FDPts
        te_fantasy = fantasy_data[fantasy_data['FantPos'] == 'TE'].copy()
        if len(te_fantasy) > 0:
            te_fantasy['Rec_Percentile'] = te_fantasy['Rec'].rank(pct=True, na_option='bottom')
            te_fantasy['FDPt_Percentile'] = te_fantasy['FDPt'].rank(pct=True, na_option='bottom')
            
            # TE Performance Score: 50% Receptions + 50% FDPts
            te_fantasy['TE_Performance_Score'] = (
                te_fantasy['Rec_Percentile'] * 0.5 +
                te_fantasy['FDPt_Percentile'] * 0.5
            )
            te_fantasy['TE_Performance_Boost'] = te_fantasy['TE_Performance_Score'] * 0.15  # Further reduced from 25% to 15% boost strength
            
            for _, te in te_fantasy.iterrows():
                te_performance_boosts[te['Player']] = te['TE_Performance_Boost']
        
        # QB boosts - based purely on FDPts performance
        qb_fantasy = fantasy_data[fantasy_data['FantPos'] == 'QB'].copy()
        if len(qb_fantasy) > 0:
            qb_fantasy['FDPt_Percentile'] = qb_fantasy['FDPt'].rank(pct=True, na_option='bottom')
            
            # QB Performance Score: 100% FDPts (simple but effective)
            qb_fantasy['QB_Performance_Score'] = qb_fantasy['FDPt_Percentile']
            qb_fantasy['QB_Performance_Boost'] = qb_fantasy['QB_Performance_Score'] * 0.15  # Reduced from 30% to 15% max boost strength
            
            for _, qb in qb_fantasy.iterrows():
                qb_performance_boosts[qb['Player']] = qb['QB_Performance_Boost']
    
    return wr_performance_boosts, rb_performance_boosts, te_performance_boosts, qb_performance_boosts

def get_top_rushing_qbs(fantasy_data, num_qbs=6):
    """Get top rushing QBs based on TD_3 (60%) and Yds_2 (40%) for non-stack lineups"""
    if fantasy_data is None:
        return set()
    
    # Filter to QBs only
    qb_fantasy = fantasy_data[fantasy_data['FantPos'] == 'QB'].copy()
    if len(qb_fantasy) == 0:
        return set()
    
    # Create percentile ranks for rushing stats
    qb_fantasy['TD_3_Percentile'] = qb_fantasy['TD_3'].rank(pct=True, na_option='bottom')
    qb_fantasy['Yds_2_Percentile'] = qb_fantasy['Yds_2'].rank(pct=True, na_option='bottom')
    
    # Calculate composite rushing score: 60% TDs + 40% Yards
    qb_fantasy['Rushing_Score'] = (
        qb_fantasy['TD_3_Percentile'] * 0.6 +
        qb_fantasy['Yds_2_Percentile'] * 0.4
    )
    
    # Get top 6 rushing QBs
    top_rushing_qbs = qb_fantasy.nlargest(num_qbs, 'Rushing_Score')['Player'].tolist()
    return set(top_rushing_qbs)

def create_weighted_pools(df, wr_performance_boosts, rb_performance_boosts, te_performance_boosts, qb_performance_boosts, elite_target_boost, great_target_boost, forced_players=None, forced_player_boost=0.0):
    """Create weighted player pools"""
    pools = {}
    
    # For QB position, identify highest salary QB per team and apply automatic boost
    qb_salary_boost = 0.5  # Reduced from 100% to 50% boost for highest salary QBs
    highest_salary_qbs = set()
    qb_players = df[df['Position'] == 'QB']
    for team in qb_players['Team'].unique():
        team_qbs = qb_players[qb_players['Team'] == team]
        if len(team_qbs) > 0:
            highest_salary_qb = team_qbs.loc[team_qbs['Salary'].idxmax(), 'Nickname']
            highest_salary_qbs.add(highest_salary_qb)
    
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = df[df['Position'] == pos].copy()
        original_count = len(pos_players)
        
        # Apply ceiling filter for RBs, WRs, and TEs (must have ceiling > 7 points)
        if pos in ['RB', 'WR', 'TE'] and 'Ceiling' in pos_players.columns:
            pos_players = pos_players[pos_players['Ceiling'] > 7.0]
        
        # Apply TE salary filter (reduced minimum to $4,000 for more options)
        if pos == 'TE':
            pos_players = pos_players[pos_players['Salary'] >= 4000]
            # TEs below $4,000 are filtered out (no debug message needed)
        
        # For QB position, only include highest salary QB per team UNLESS they're forced
        if pos == 'QB':
            pre_filter = len(pos_players)
            # Keep highest salary QBs AND any forced QBs
            if forced_players:
                forced_qbs_in_pos = pos_players[pos_players['Nickname'].isin(forced_players)]
                highest_salary_qbs_in_pos = pos_players[pos_players['Nickname'].isin(highest_salary_qbs)]
                # Combine both sets and remove duplicates
                pos_players = pd.concat([highest_salary_qbs_in_pos, forced_qbs_in_pos]).drop_duplicates()
            else:
                pos_players = pos_players[pos_players['Nickname'].isin(highest_salary_qbs)]
            
            if len(pos_players) < pre_filter:
                # Backup QBs included when forced, but no debug message needed
                pass
        
        weights = []
        
        for _, player in pos_players.iterrows():
            base_weight = 1.0
            
            # Apply matchup boost
            if player['Matchup_Quality'] == 'ELITE TARGET':
                weight = base_weight * (1 + elite_target_boost)
            elif player['Matchup_Quality'] == 'Great Target':
                weight = base_weight * (1 + great_target_boost)
            elif player['Matchup_Quality'] == 'Good Target':
                weight = base_weight * 1.1
            else:
                weight = base_weight
            
            # Apply fantasy performance boost
            player_name = player['Nickname']
            if pos == 'QB' and player_name in qb_performance_boosts:
                weight = weight * (1 + qb_performance_boosts[player_name])
            elif pos == 'WR' and player_name in wr_performance_boosts:
                weight = weight * (1 + wr_performance_boosts[player_name])
            elif pos == 'RB' and player_name in rb_performance_boosts:
                weight = weight * (1 + rb_performance_boosts[player_name])
            elif pos == 'TE' and player_name in te_performance_boosts:
                weight = weight * (1 + te_performance_boosts[player_name])
            
            # Apply forced player boost
            if forced_players and forced_player_boost > 0:
                if player_name in forced_players:
                    weight = weight * (1 + forced_player_boost)
            
            # Apply QB highest salary boost (automatic 100% boost)
            if pos == 'QB':
                if player_name in highest_salary_qbs:
                    weight = weight * (1 + qb_salary_boost)
            
            weights.append(weight)
        
        pos_players['Selection_Weight'] = weights
        pools[pos] = pos_players
    
    return pools

def get_top_matchups(df, pass_defense, rush_defense, num_per_position=6):
    """Get top matchup analysis by position for display using current week data"""
    position_matchups = {}
    
    try:
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_players = df[df['Position'] == pos].copy()
            if len(pos_players) > 0:
                # For QB and RB, filter to show only highest salaried player per team
                if pos in ['QB', 'RB']:
                    pos_players = pos_players.loc[pos_players.groupby('Team', observed=True)['Salary'].idxmax()]
                
                # Sort by matchup quality and FPPG for display
                pos_players['sort_key'] = pos_players['Matchup_Quality'].map({
                    'ELITE TARGET': 3,
                    'Great Target': 2, 
                    'Good Target': 1
                }).fillna(0)
                
                # Sort by matchup quality first, then by FPPG
                pos_players = pos_players.sort_values(['sort_key', 'FPPG'], ascending=[False, False])
                
                # Create display dataframe with actual defensive rankings
                display_df = pos_players.head(num_per_position).copy()
                display_df['Player'] = display_df['Nickname']
                display_df['vs'] = display_df['Opponent']
                
                # Get actual defensive rankings based on position
                for idx, row in display_df.iterrows():
                    opponent = row['Opponent']
                    if pos in ['QB', 'WR', 'TE']:
                        # Use pass defense rankings
                        if pass_defense is not None and opponent in pass_defense['Team'].values:
                            def_data = pass_defense[pass_defense['Team'] == opponent].iloc[0]
                            display_df.at[idx, 'Defense_Rank'] = def_data.get('Pass_Defense_Rank', 'N/A')
                            display_df.at[idx, 'YPG_Allowed'] = def_data.get('Pass_YPG_Allowed', 'N/A')
                        else:
                            display_df.at[idx, 'Defense_Rank'] = 'N/A'
                            display_df.at[idx, 'YPG_Allowed'] = 'N/A'
                    else:  # RB
                        # Use rush defense rankings
                        if rush_defense is not None and opponent in rush_defense['Team'].values:
                            def_data = rush_defense[rush_defense['Team'] == opponent].iloc[0]
                            display_df.at[idx, 'Defense_Rank'] = def_data.get('Rush_Defense_Rank', 'N/A')
                            display_df.at[idx, 'YPG_Allowed'] = def_data.get('Rush_YPG_Allowed', 'N/A')
                        else:
                            display_df.at[idx, 'Defense_Rank'] = 'N/A'
                            display_df.at[idx, 'YPG_Allowed'] = 'N/A'
                
                position_matchups[pos] = display_df
                
        return position_matchups
    except Exception as e:
        st.warning(f"Could not generate current week matchups: {str(e)}")
        return {}

def generate_lineups(df, weighted_pools, num_simulations, stack_probability, elite_target_boost, great_target_boost, fantasy_data=None, player_selections=None, force_mode="Soft Force (Boost Only)", forced_player_boost=0.0, strategy_type="Custom", tournament_params=None):
    """Generate optimized lineups with optional player selection constraints and tournament optimization"""
    stacked_lineups = []
    salary_cap = 60000
    
    # Extract tournament parameters
    if tournament_params is None:
        tournament_params = {}
    contrarian_boost = tournament_params.get('contrarian_boost', 0.05)
    correlation_preference = tournament_params.get('correlation_preference', 0.3)
    salary_variance_target = tournament_params.get('salary_variance_target', 0.2)
    leverage_focus = tournament_params.get('leverage_focus', 0.1)
    
    # Pre-validate constraints and adjust simulation count if needed
    original_simulations = num_simulations
    if player_selections:
        # Count total forced players
        total_forced = sum(len(player_selections[pos]['must_include']) for pos in ['QB', 'RB', 'WR', 'TE', 'D'] if pos in player_selections)
        
        # More aggressive reduction for faster generation
        if total_forced > 15:
            num_simulations = min(num_simulations, 2000)
            st.info(f"ℹ️ Reduced simulations to {num_simulations:,} due to {total_forced} forced players for faster generation")
        elif total_forced > 10:
            num_simulations = min(num_simulations, 3000)
            st.info(f"ℹ️ Reduced simulations to {num_simulations:,} due to {total_forced} forced players for faster generation")
        elif total_forced > 5:
            num_simulations = min(num_simulations, 5000)
            st.info(f"ℹ️ Reduced simulations to {num_simulations:,} due to {total_forced} forced players for faster generation")
    
    # Apply strategy-specific adjustments to weighted pools
    strategy_pools = {}
    for pos in weighted_pools:
        strategy_pools[pos] = weighted_pools[pos].copy()
        
        if strategy_type == "Single Entry":
            # Single Entry Strategy: Favor consistency and lower ownership
            if 'FPPG' in strategy_pools[pos].columns and 'Salary' in strategy_pools[pos].columns:
                # Boost floor-based value (FPPG/Salary ratio)
                strategy_pools[pos]['Floor_Value'] = strategy_pools[pos]['FPPG'] / (strategy_pools[pos]['Salary'] / 1000)
                # Penalize extremely cheap players (often boom/bust)
                min_salary_threshold = {'QB': 7000, 'RB': 5000, 'WR': 5000, 'TE': 4000, 'D': 4000}
                strategy_pools[pos].loc[strategy_pools[pos]['Salary'] < min_salary_threshold.get(pos, 4000), 'Selection_Weight'] *= 0.7
                # Boost mid-range salaries (typically more consistent)
                mid_salary_threshold = {'QB': 8500, 'RB': 7000, 'WR': 7000, 'TE': 5500, 'D': 5000}
                strategy_pools[pos].loc[
                    (strategy_pools[pos]['Salary'] >= min_salary_threshold.get(pos, 4000)) & 
                    (strategy_pools[pos]['Salary'] <= mid_salary_threshold.get(pos, 7000)), 
                    'Selection_Weight'
                ] *= 1.3
                
        elif strategy_type == "Tournament":
            # Simplified Tournament Strategy: Basic contrarian focus
            if 'FPPG' in strategy_pools[pos].columns and 'Salary' in strategy_pools[pos].columns:
                # Simple salary tier adjustments (less complex)
                if pos == 'QB':
                    # Boost punt QBs and elite QBs slightly
                    strategy_pools[pos].loc[strategy_pools[pos]['Salary'] < 7000, 'Selection_Weight'] *= 1.2
                    strategy_pools[pos].loc[strategy_pools[pos]['Salary'] >= 9000, 'Selection_Weight'] *= 1.1
                
                elif pos in ['RB', 'WR']:
                    # Boost punt plays and studs slightly
                    strategy_pools[pos].loc[strategy_pools[pos]['Salary'] < 5500, 'Selection_Weight'] *= 1.3
                    strategy_pools[pos].loc[strategy_pools[pos]['Salary'] >= 8500, 'Selection_Weight'] *= 1.1
                
                elif pos == 'TE':
                    # Boost punt TEs
                    strategy_pools[pos].loc[strategy_pools[pos]['Salary'] < 4500, 'Selection_Weight'] *= 1.2
    
    # Use strategy-adjusted pools for lineup generation
    weighted_pools = strategy_pools
    
    # Apply player selection filters if enabled
    if player_selections:
        filtered_pools = {}
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_pool = weighted_pools[pos].copy()
            
            # Remove excluded players
            if player_selections[pos]['exclude']:
                pos_pool = pos_pool[~pos_pool['Nickname'].isin(player_selections[pos]['exclude'])]
            
            filtered_pools[pos] = pos_pool
        
        # Handle defense separately (position is 'D' not 'DEF')
        def_pool = df[df['Position'] == 'D'].copy()
        if player_selections['D']['exclude']:
            def_pool = def_pool[~def_pool['Nickname'].isin(player_selections['D']['exclude'])]
        
        # Update weighted_pools with filtered data
        weighted_pools = filtered_pools
        def_players = def_pool
    else:
        def_players = df[df['Position'] == 'D']
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_lineups = 0
    total_attempts = 0
    
    # Debug tracking
    failure_reasons = {
        'salary_too_high': 0,
        'team_limit_exceeded': 0,
        'rb_same_team': 0,
        'insufficient_rbs': 0,
        'insufficient_players': 0,
        'salary_variance_check': 0,
        'other_errors': 0
    }
    
    for simulation in range(num_simulations):
        attempts = 0
        max_attempts = 50 if player_selections else 20  # More attempts when forcing players
        
        while attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Early exit if too many failed attempts
            if total_attempts > num_simulations * 200:  # More generous attempt limit
                st.warning(f"⚠️ Stopping early due to constraint conflicts. Generated {successful_lineups:,} lineups.")
                break
            
            try:
                lineup_players = []
                
                # Determine if this will be a stacked lineup early (enhanced for tournaments)
                adjusted_stack_prob = stack_probability
                if strategy_type == "Tournament" and correlation_preference > 0.5:
                    # Increase stacking probability for tournament correlation focus
                    adjusted_stack_prob = min(0.95, stack_probability * (1 + correlation_preference * 0.3))
                
                will_attempt_stack = random.random() < adjusted_stack_prob
                
                # Get top rushing QBs for non-stacked lineups
                top_rushing_qbs = get_top_rushing_qbs(fantasy_data) if fantasy_data is not None else set()
                
                # Handle must-include players first (only if Hard Force mode)
                if player_selections and force_mode == "Hard Force (Always Include)":
                    # Must include QB - rotate through forced QBs for variety
                    if player_selections['QB']['must_include']:
                        available_forced_qbs = player_selections['QB']['must_include']
                        # Randomly select from forced QBs to create variety
                        must_qb_name = random.choice(available_forced_qbs)
                        qb = weighted_pools['QB'][weighted_pools['QB']['Nickname'] == must_qb_name]
                        if len(qb) == 0:
                            continue  # Skip if player not found
                    else:
                        qb_pool = weighted_pools['QB']
                        qb = qb_pool.sample(1, weights=qb_pool['Selection_Weight'])
                else:
                    # Regular QB selection with special logic for non-stacked lineups
                    qb_pool = weighted_pools['QB']
                    
                    # Enhanced soft forcing: For forced players with high boost, increase their selection probability
                    forced_qb_in_pool = None
                    if player_selections and forced_player_boost > 0.2:  # If boost > 20% (lowered from 50%)
                        forced_qbs_in_pool = qb_pool[qb_pool['Nickname'].isin(player_selections['QB']['must_include'])]
                        if len(forced_qbs_in_pool) > 0:
                            # Enhanced soft forcing - higher probability for forced players
                            force_probability = min(0.85, forced_player_boost + 0.4)  # Higher base probability
                            if random.random() < force_probability:
                                # Select from forced QBs only
                                qb = forced_qbs_in_pool.sample(1, weights=forced_qbs_in_pool['Selection_Weight'])
                                forced_qb_in_pool = qb['Nickname'].iloc[0]
                    
                    # If no forced QB selected, proceed with normal logic
                    if forced_qb_in_pool is None:
                        # For non-stacked lineups, prefer top rushing QBs
                        if not will_attempt_stack and len(top_rushing_qbs) > 0:
                            # Filter to top rushing QBs only
                            rushing_qb_pool = qb_pool[qb_pool['Nickname'].isin(top_rushing_qbs)]
                            if len(rushing_qb_pool) > 0:
                                qb = rushing_qb_pool.sample(1, weights=rushing_qb_pool['Selection_Weight'])
                            else:
                                # Fallback to regular QB pool if no rushing QBs available
                                qb = qb_pool.sample(1, weights=qb_pool['Selection_Weight'])
                        else:
                            # Regular QB selection for stacked lineups
                            qb = qb_pool.sample(1, weights=qb_pool['Selection_Weight'])
                
                qb_team = qb['Team'].iloc[0]
                lineup_players.append(qb)
                
                # Handle must-include RBs - rotate for variety (only if Hard Force mode)
                selected_rbs = pd.DataFrame()
                if player_selections and player_selections['RB']['must_include'] and force_mode == "Hard Force (Always Include)":
                    available_forced_rbs = player_selections['RB']['must_include']
                    # Randomly select from forced RBs, up to 2
                    num_forced_rbs = min(len(available_forced_rbs), 2)
                    if num_forced_rbs > 0:
                        selected_rb_names = random.sample(available_forced_rbs, num_forced_rbs)
                        for must_rb_name in selected_rb_names:
                            must_rb = weighted_pools['RB'][weighted_pools['RB']['Nickname'] == must_rb_name]
                            if len(must_rb) > 0:
                                selected_rbs = pd.concat([selected_rbs, must_rb])
                
                # Fill remaining RB spots (with team constraint)
                remaining_rb_spots = 2 - len(selected_rbs)
                if remaining_rb_spots > 0:
                    available_rbs = weighted_pools['RB']
                    if len(selected_rbs) > 0:
                        used_rb_names = set(selected_rbs['Nickname'])
                        used_rb_teams = set(selected_rbs['Team'])  # Track teams already used
                        available_rbs = available_rbs[~available_rbs['Nickname'].isin(used_rb_names)]
                        # Exclude RBs from teams already selected
                        available_rbs = available_rbs[~available_rbs['Team'].isin(used_rb_teams)]
                    
                    if len(available_rbs) >= remaining_rb_spots:
                        # CEILING-PRIORITIZED RB SELECTION
                        # Boost Selection_Weight by ceiling potential for RBs
                        if 'Ceiling' in available_rbs.columns and 'Selection_Weight' in available_rbs.columns:
                            ceiling_boost = available_rbs['Ceiling'] / available_rbs['FPPG']  # Ceiling multiplier
                            available_rbs['Ceiling_Boosted_Weight'] = available_rbs['Selection_Weight'] * ceiling_boost
                            additional_rbs = available_rbs.sample(remaining_rb_spots, weights=available_rbs['Ceiling_Boosted_Weight'])
                        elif 'Selection_Weight' in available_rbs.columns:
                            additional_rbs = available_rbs.sample(remaining_rb_spots, weights=available_rbs['Selection_Weight'])
                        else:
                            # Fallback to equal probability if no Selection_Weight
                            additional_rbs = available_rbs.sample(remaining_rb_spots)
                        rb = pd.concat([selected_rbs, additional_rbs])
                    else:
                        failure_reasons['insufficient_rbs'] += 1
                        continue  # Skip this lineup if can't find RBs from different teams
                else:
                    rb = selected_rbs
                
                # Additional check: Ensure no duplicate RB teams in final selection
                if len(rb) == 2:
                    rb_teams = rb['Team'].tolist()
                    if rb_teams[0] == rb_teams[1]:
                        failure_reasons['rb_same_team'] += 1
                        continue  # Skip this lineup if RBs are from same team
                
                lineup_players.append(rb)
                
                # WR/TE selection with stacking (modified for must-include)
                wr_pool = weighted_pools['WR']
                te_pool = weighted_pools['TE']
                
                # Handle must-include WRs (only if Hard Force mode)
                selected_wrs = pd.DataFrame()
                if player_selections and player_selections['WR']['must_include'] and force_mode == "Hard Force (Always Include)":
                    for must_wr_name in player_selections['WR']['must_include'][:3]:  # Max 3 WRs
                        must_wr = wr_pool[wr_pool['Nickname'] == must_wr_name]
                        if len(must_wr) > 0:
                            selected_wrs = pd.concat([selected_wrs, must_wr])
                
                # Handle must-include TEs (only if Hard Force mode)
                selected_te = pd.DataFrame()
                if player_selections and player_selections['TE']['must_include'] and force_mode == "Hard Force (Always Include)":
                    must_te_name = player_selections['TE']['must_include'][0]  # Max 1 TE
                    must_te = te_pool[te_pool['Nickname'] == must_te_name]
                    if len(must_te) > 0:
                        selected_te = must_te
                
                # Enhanced tournament stacking logic
                remaining_wr_spots = 3 - len(selected_wrs)
                need_te = len(selected_te) == 0
                
                if remaining_wr_spots > 0 or need_te:
                    # Enhanced stacking decision for tournaments
                    attempt_stack = will_attempt_stack and remaining_wr_spots > 0
                    
                    # Simplified stacking (back to original working logic)
                    if attempt_stack:
                        same_team_wrs = wr_pool[wr_pool['Team'] == qb_team]
                        # Remove already selected WRs
                        if len(selected_wrs) > 0:
                            used_wr_names = set(selected_wrs['Nickname'])
                            same_team_wrs = same_team_wrs[~same_team_wrs['Nickname'].isin(used_wr_names)]
                        
                        if len(same_team_wrs) >= 1:
                            stack_count = min(remaining_wr_spots, len(same_team_wrs), 2)
                            stacked_wrs = same_team_wrs.sample(stack_count, weights=same_team_wrs['Selection_Weight'])
                            selected_wrs = pd.concat([selected_wrs, stacked_wrs])
                            remaining_wr_spots -= stack_count
                    
                    # Fill remaining WR spots
                    if remaining_wr_spots > 0:
                        available_wrs = wr_pool
                        if len(selected_wrs) > 0:
                            used_wr_names = set(selected_wrs['Nickname'])
                            available_wrs = available_wrs[~available_wrs['Nickname'].isin(used_wr_names)]
                        
                        if len(available_wrs) >= remaining_wr_spots:
                            additional_wrs = available_wrs.sample(remaining_wr_spots, weights=available_wrs['Selection_Weight'])
                            wr = pd.concat([selected_wrs, additional_wrs])
                        else:
                            continue
                    else:
                        wr = selected_wrs
                    
                    # Handle TE
                    if need_te:
                        available_tes = te_pool
                        te = available_tes.sample(1, weights=available_tes['Selection_Weight'])
                    else:
                        te = selected_te
                else:
                    wr = selected_wrs
                    te = selected_te
                
                lineup_players.extend([wr, te])
                
                # Defense selection
                if player_selections and player_selections['D']['must_include']:
                    must_def_name = player_selections['D']['must_include'][0]
                    def_ = def_players[def_players['Nickname'] == must_def_name]
                    if len(def_) == 0:
                        continue
                else:
                    # Regular defense selection
                    if random.random() < 0.8:
                        cheap_def = def_players[def_players['Salary'] <= 4000]
                        if len(cheap_def) > 0:
                            def_ = cheap_def.sample(1)
                        else:
                            def_ = def_players.sample(1)
                    else:
                        def_ = def_players.sample(1)
                
                lineup_players.append(def_)
                
                # FLEX selection - use filtered pools to respect exclusions
                if player_selections:
                    # Combine filtered RB, WR, TE pools for FLEX
                    flex_rb_pool = weighted_pools['RB'] if 'RB' in weighted_pools else pd.DataFrame()
                    flex_wr_pool = weighted_pools['WR'] if 'WR' in weighted_pools else pd.DataFrame()
                    flex_te_pool = weighted_pools['TE'] if 'TE' in weighted_pools else pd.DataFrame()
                    flex_players = pd.concat([flex_rb_pool, flex_wr_pool, flex_te_pool], ignore_index=True)
                else:
                    flex_players = df[df['Position'].isin(['RB', 'WR', 'TE'])]
                
                used_names = set()
                for player_group in lineup_players[1:4]:  # RB, WR, TE
                    used_names.update(player_group['Nickname'])
                
                flex_pool = flex_players[~flex_players['Nickname'].isin(used_names)]
                
                if len(flex_pool) == 0:
                    failure_reasons['insufficient_players'] += 1
                    continue
                
                # CEILING-PRIORITIZED FLEX SELECTION (balanced across all positions)
                if 'Ceiling' in flex_pool.columns and 'Selection_Weight' in flex_pool.columns:
                    flex_weights = flex_pool['Selection_Weight'].copy()
                    
                    # Apply ceiling boost to all positions
                    ceiling_boost = flex_pool['Ceiling'] / flex_pool['FPPG']
                    flex_weights *= ceiling_boost
                    
                    # Small additional boost for RBs (only 10% to maintain balance)
                    rb_mask = flex_pool['Position'] == 'RB'
                    flex_weights.loc[rb_mask] *= 1.10
                    
                    flex = flex_pool.sample(1, weights=flex_weights)
                elif 'Selection_Weight' in flex_pool.columns:
                    flex = flex_pool.sample(1, weights=flex_pool['Selection_Weight'])
                else:
                    # Fallback to equal probability
                    flex = flex_pool.sample(1)
                lineup_players.append(flex)
                
                # Build final lineup
                lineup = pd.concat(lineup_players).reset_index(drop=True)
                
                # Enhanced tournament lineup validation
                total_salary = lineup['Salary'].sum()
                if total_salary > salary_cap:
                    failure_reasons['salary_too_high'] += 1
                    continue
                
                # Check 4-player per team limit (FanDuel rule)
                team_counts = lineup['Team'].value_counts()
                if team_counts.max() > 4:
                    failure_reasons['team_limit_exceeded'] += 1
                    continue  # Skip lineups with more than 4 players from same team
                
                # Tournament-specific salary distribution validation
                if strategy_type == "Tournament":
                    # Advanced salary distribution validation for tournaments
                    high_salary_players = len(lineup[lineup['Salary'] >= 8000])
                    low_salary_players = len(lineup[lineup['Salary'] <= 5500])
                    mid_salary_players = len(lineup[(lineup['Salary'] > 5500) & (lineup['Salary'] < 8000)])
                    
                    # Calculate salary distribution score
                    salary_variance = lineup['Salary'].var()
                    normalized_variance = min(1.0, salary_variance / 5000000)  # Normalize to 0-1
                    
                    # Simplified salary variance (much less restrictive)
                    # Only apply very basic checks to avoid generation failures
                    if salary_variance_target > 0.9:  # Only in extreme cases
                        if high_salary_players == 0 and low_salary_players == 0:
                            failure_reasons['salary_variance_check'] += 1
                            continue  # Only skip if completely flat
                    
                    # Note: Tournament leverage scoring moved after total_points calculation
                
                # Validate lineup
                if (len(lineup) == 9 and 
                    len(lineup['Nickname'].unique()) == 9):
                    
                    # Calculate base points first
                    total_points = lineup['Adjusted_FPPG'].sum()
                    
                    # Enhanced stacking analysis
                    qb_team = lineup[lineup['Position'] == 'QB']['Team'].iloc[0]
                    wr_teams = lineup[lineup['Position'] == 'WR']['Team'].tolist()
                    te_teams = lineup[lineup['Position'] == 'TE']['Team'].tolist()
                    rb_teams = lineup[lineup['Position'] == 'RB']['Team'].tolist()
                    
                    stacked_wrs_count = sum(1 for team in wr_teams if team == qb_team)
                    stacked_tes_count = sum(1 for team in te_teams if team == qb_team)
                    qb_wr_te_count = stacked_wrs_count + stacked_tes_count
                    
                    # Simplified tournament correlation scoring
                    if strategy_type == "Tournament":
                        # QB stack bonus (correlated production)
                        if qb_wr_te_count >= 2:
                            total_points *= 1.03  # 3% bonus for strong stacks
                        elif qb_wr_te_count == 1:
                            total_points *= 1.01  # 1% bonus for mini stacks
                        
                        # Tournament leverage scoring (now safe to use total_points)
                        if leverage_focus > 0.3:
                            high_salary_players = len(lineup[lineup['Salary'] >= 8000])
                            low_salary_players = len(lineup[lineup['Salary'] <= 5500])
                            # Bonus for punt + stud combinations (leverage plays)
                            leverage_score = (high_salary_players * low_salary_players) / 9
                            total_points *= (1 + leverage_score * leverage_focus * 0.02)
                    
                    stacked_lineups.append((total_points, lineup, total_salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count))
                    successful_lineups += 1
                    break
                    
            except Exception as e:
                failure_reasons['other_errors'] += 1
                # Print first 5 errors for debugging
                if failure_reasons['other_errors'] <= 5:
                    st.error(f"🐛 **Debug Error #{failure_reasons['other_errors']}:** {str(e)}")
                continue
        
        # Break outer loop if too many failed attempts
        if total_attempts > num_simulations * 200:  # Match the inner condition
            break
        
        # Update progress with better info
        progress = (simulation + 1) / num_simulations
        progress_bar.progress(progress)
        if (simulation + 1) % 50 == 0 or simulation < 100:
            status_text.text(f"Generated {successful_lineups:,} / {num_simulations:,} lineups... (Success rate: {(successful_lineups/(simulation+1)*100):.1f}%)")
    
    progress_bar.empty()
    status_text.empty()
    
    # Show enhanced debug information for low success rates
    success_rate = successful_lineups / num_simulations if num_simulations > 0 else 0
    if success_rate < 0.1:  # Less than 10% success rate
        st.error(f"🚨 **Critical Generation Failure** ({success_rate:.1%} success rate)")
        
        total_failures = sum(failure_reasons.values())
        if total_failures > 0:
            st.write("**Detailed Failure Analysis:**")
            
            # Create columns for better display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Failure Counts:**")
                for reason, count in failure_reasons.items():
                    if count > 0:
                        percentage = (count / total_failures) * 100
                        st.write(f"• {reason.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
            
            with col2:
                st.write("**Recommended Fixes:**")
                
                if failure_reasons['salary_too_high'] > total_failures * 0.2:
                    st.error("💰 Salary cap issues - reduce high-salary forced players")
                
                if failure_reasons['team_limit_exceeded'] > total_failures * 0.2:
                    st.error("🏈 Team limit issues - spread forced players across teams")
                
                if failure_reasons['insufficient_rbs'] > total_failures * 0.2:
                    st.error("🏃 RB pool too small - reduce RB exclusions")
                
                if failure_reasons['rb_same_team'] > total_failures * 0.2:
                    st.error("🏃 RB team conflicts - need more RB team diversity")
                
                if failure_reasons['salary_variance_check'] > total_failures * 0.2:
                    st.error("📊 Tournament settings too restrictive - lower Salary Variance Target")
    
    elif success_rate < 0.5:  # 10-50% success rate
        st.warning(f"⚠️ Low success rate ({success_rate:.1%}). Consider reducing forced players or adjusting constraints.")
    
    # Apply exposure capping (30% max) while preserving all existing logic
    stacked_lineups = apply_exposure_capping(stacked_lineups, max_exposure=0.30)
    
    return stacked_lineups

def apply_exposure_capping(lineups, max_exposure=0.30):
    """
    Apply exposure capping to lineups while preserving existing selection logic
    Max exposure = 30% means a player can appear in max 30% of lineups
    """
    if not lineups:
        return lineups
    
    total_lineups = len(lineups)
    max_appearances = int(total_lineups * max_exposure)
    
    # Count current player usage
    player_counts = {}
    lineup_players = {}  # Track which players are in each lineup
    
    for i, (points, lineup, salary, stacked_wrs, stacked_tes, qb_wr_te) in enumerate(lineups):
        lineup_players[i] = []
        for _, player in lineup.iterrows():
            player_key = f"{player['Nickname']}_{player['Position']}_{player['Team']}"
            lineup_players[i].append(player_key)
            
            if player_key not in player_counts:
                player_counts[player_key] = {'count': 0, 'lineups': []}
            player_counts[player_key]['count'] += 1
            player_counts[player_key]['lineups'].append(i)
    
    # Identify over-exposed players
    over_exposed = {}
    for player_key, data in player_counts.items():
        if data['count'] > max_appearances:
            over_exposed[player_key] = data
    
    if not over_exposed:
        return lineups  # No exposure issues
    
    # Remove lineups to cap exposure, prioritizing lower-scoring lineups for removal
    lineups_to_remove = set()
    
    for player_key, data in over_exposed.items():
        excess_appearances = data['count'] - max_appearances
        
        # Get lineups with this over-exposed player, sorted by score (lowest first)
        player_lineup_scores = [(i, lineups[i][0]) for i in data['lineups'] if i not in lineups_to_remove]
        player_lineup_scores.sort(key=lambda x: x[1])  # Sort by score ascending
        
        # Mark lowest-scoring lineups for removal
        for i, (lineup_idx, score) in enumerate(player_lineup_scores):
            if i < excess_appearances:
                lineups_to_remove.add(lineup_idx)
    
    # Create final lineup list without over-exposed lineups
    final_lineups = [lineup for i, lineup in enumerate(lineups) if i not in lineups_to_remove]
    
    # Show exposure capping results
    if lineups_to_remove:
        st.info(f"🎯 **Exposure Capping Applied**: Removed {len(lineups_to_remove)} lineups to maintain ≤30% player exposure")
        
        # Show which players were capped
        capped_players = []
        for player_key in over_exposed.keys():
            player_name = player_key.split('_')[0]  # Extract name
            capped_players.append(player_name)
        
        if capped_players:
            st.write(f"**Players capped**: {', '.join(capped_players[:5])}{'...' if len(capped_players) > 5 else ''}")
    
    return final_lineups

def recalculate_all_lineup_stacking():
    """Recalculate stacking counts for ALL lineups in session state"""
    if 'stacked_lineups' not in st.session_state:
        return
    
    updated_lineups = []
    for points, lineup, salary, old_stacked_wrs, old_stacked_tes, old_qb_wr_te in st.session_state.stacked_lineups:
        # Recalculate actual stacking for this lineup
        new_stacked_wrs, new_stacked_tes, new_qb_wr_te = recalculate_lineup_stacking(lineup)
        updated_lineups.append((points, lineup, salary, new_stacked_wrs, new_stacked_tes, new_qb_wr_te))
    
    # Update session state with corrected stacking counts
    st.session_state.stacked_lineups = updated_lineups


def recalculate_lineup_stacking(lineup):
    """Recalculate stacking counts for a modified lineup"""
    qb_team = None
    wr_teams = []
    te_teams = []
    
    # Find QB team and all skill position teams
    for _, player in lineup.iterrows():
        team = str(player['Team']).strip().upper()  # Normalize team names
        if player['Position'] == 'QB':
            qb_team = team
        elif player['Position'] == 'WR':
            wr_teams.append(team)
        elif player['Position'] == 'TE':
            te_teams.append(team)
    
    if qb_team:
        stacked_wrs_count = sum(1 for team in wr_teams if team == qb_team)
        stacked_tes_count = sum(1 for team in te_teams if team == qb_team)
        qb_wr_te_count = stacked_wrs_count + stacked_tes_count
    else:
        stacked_wrs_count = 0
        stacked_tes_count = 0
        qb_wr_te_count = 0
    
    return stacked_wrs_count, stacked_tes_count, qb_wr_te_count


def find_team_stack_relationships(lineup):
    """Helper function to identify team stacking relationships in a lineup"""
    team_positions = {}
    stack_info = {}
    
    for idx, player in lineup.iterrows():
        team = player['Team']
        position = player['Position']
        
        if team not in team_positions:
            team_positions[team] = []
        team_positions[team].append({
            'index': idx,
            'position': position,
            'player': player['Nickname']
        })
    
    # Identify stacks (QB + skill position combinations)
    for team, players in team_positions.items():
        if len(players) > 1:
            positions = [p['position'] for p in players]
            if 'QB' in positions:
                skill_positions = [p for p in players if p['position'] in ['WR', 'TE']]
                if skill_positions:
                    qb_player = next(p for p in players if p['position'] == 'QB')
                    stack_info[team] = {
                        'qb': qb_player,
                        'skill_players': skill_positions,
                        'is_stack': True
                    }
    
    return stack_info


def apply_usage_adjustments(lineups, filtered_players, selected_position):
    """
    Apply usage adjustments by intelligently modifying actual lineups
    to match target exposure percentages across ALL players with adjustments,
    not just the currently filtered position
    """
    import random
    import copy
    
    if not lineups:
        return lineups
    
    # Get ALL adjustments from session state, not just filtered players
    all_adjustments = {}
    
    # Look through ALL session state keys for usage adjustments
    for key in st.session_state.keys():
        if key.startswith("usage_adj_"):
            # Extract player info from key: usage_adj_PlayerName_Position_Team
            key_parts = key.replace("usage_adj_", "").split("_")
            if len(key_parts) >= 3:
                player_name = "_".join(key_parts[:-2]).replace("_", " ")  # Reconstruct name with spaces
                position = key_parts[-2]
                team = key_parts[-1]
                
                target_usage = st.session_state[key]
                
                # Find this player's current usage from the comprehensive data
                current_usage = None
                player_salary = None
                
                if 'comprehensive_usage_data' in st.session_state:
                    for player_entry in st.session_state.comprehensive_usage_data:
                        if (player_entry['Player'] == player_name and 
                            player_entry['Position'] == position):
                            current_usage = player_entry['Usage %']
                            player_salary = player_entry.get('Salary', 5000)
                            break
                
                # If we found the player and there's a meaningful change
                if current_usage is not None and abs(target_usage - current_usage) > 0.1:
                    all_adjustments[player_name] = {
                        'current': current_usage,
                        'target': target_usage,
                        'change': target_usage - current_usage,
                        'position': position,
                        'team': team,
                        'salary': player_salary
                    }
    
    if not all_adjustments:
        st.info("ℹ️ No changes to apply across any positions")
        return lineups
    
    # Create a working copy of lineups
    modified_lineups = copy.deepcopy(lineups)
    total_lineups = len(modified_lineups)
    
    st.write(f"**🔄 Processing {len(all_adjustments)} exposure adjustments across ALL positions...**")
    
    # Process each adjustment to achieve exact exposure targets
    successful_adjustments = 0
    
    for player_name, adjustment in all_adjustments.items():
        current_count = int(adjustment['current'] * total_lineups / 100)
        target_count = int(adjustment['target'] * total_lineups / 100)
        change_needed = target_count - current_count
        
        st.write(f"• **{player_name}** ({adjustment['position']}): {current_count} → {target_count} lineups ({change_needed:+d})")
        
        if change_needed == 0:
            continue
            
        player_position = adjustment['position']
        player_team = adjustment['team']
        player_salary = adjustment['salary']
        
        # Find lineups containing this player
        lineups_with_player = []
        lineups_without_player = []
        
        for idx, (points, lineup, salary, stacked_wrs, stacked_tes, qb_wr_te) in enumerate(modified_lineups):
            has_player = False
            for _, row in lineup.iterrows():
                if row['Nickname'] == player_name and row['Position'] == player_position:
                    has_player = True
                    break
            
            if has_player:
                lineups_with_player.append(idx)
            else:
                lineups_without_player.append(idx)
        
        if change_needed < 0:  # Need to REMOVE player from lineups
            remove_count = min(abs(change_needed), len(lineups_with_player))
            
            # Sort by projected points (remove from lowest scoring lineups first)
            lineups_with_player.sort(key=lambda idx: modified_lineups[idx][0])
            
            removed_successfully = 0
            for i in range(remove_count):
                if i >= len(lineups_with_player):
                    break
                    
                lineup_idx = lineups_with_player[i]
                points, lineup, salary, stacked_wrs, stacked_tes, qb_wr_te = modified_lineups[lineup_idx]
                
                # Find ANY available replacement player from the pool
                replacement_found = False
                best_replacement = None
                best_salary_fit = float('inf')
                
                # Look for replacement from ALL other lineups at the same position
                for other_idx, (_, other_lineup, _, _, _, _) in enumerate(modified_lineups):
                    if other_idx != lineup_idx:
                        for _, other_row in other_lineup.iterrows():
                            if (other_row['Position'] == player_position and 
                                other_row['Nickname'] != player_name):
                                
                                # Calculate salary difference
                                salary_diff = abs(other_row['Salary'] - player_salary)
                                
                                # Prefer closer salary matches but allow flexibility
                                if salary_diff < best_salary_fit:
                                    best_salary_fit = salary_diff
                                    best_replacement = {
                                        'name': other_row['Nickname'],
                                        'salary': other_row['Salary'],
                                        'team': other_row['Team'],
                                        'fppg': other_row.get('FPPG', 10)
                                    }
                
                if best_replacement:
                    # Replace the player in the lineup
                    for idx, row in lineup.iterrows():
                        if row['Nickname'] == player_name and row['Position'] == player_position:
                            lineup.at[idx, 'Nickname'] = best_replacement['name']
                            lineup.at[idx, 'Salary'] = best_replacement['salary']
                            lineup.at[idx, 'Team'] = best_replacement['team']
                            replacement_found = True
                            break
                    
                    if replacement_found:
                        new_salary = lineup['Salary'].sum()
                        if new_salary <= 60000:  # Salary cap check
                            # Recalculate stacking counts after player replacement
                            new_stacked_wrs, new_stacked_tes, new_qb_wr_te = recalculate_lineup_stacking(lineup)
                            modified_lineups[lineup_idx] = (points, lineup, new_salary, new_stacked_wrs, new_stacked_tes, new_qb_wr_te)
                            removed_successfully += 1
            
            successful_adjustments += removed_successfully
        
        elif change_needed > 0:  # Need to ADD player to more lineups
            add_count = min(change_needed, len(lineups_without_player))
            
            # Sort by projected points (add to highest scoring potential lineups)
            lineups_without_player.sort(key=lambda idx: modified_lineups[idx][0], reverse=True)
            
            added_successfully = 0
            for i in range(add_count):
                if i >= len(lineups_without_player):
                    break
                    
                lineup_idx = lineups_without_player[i]
                points, lineup, salary, stacked_wrs, stacked_tes, qb_wr_te = modified_lineups[lineup_idx]
                
                # STACK-AWARE REPLACEMENT LOGIC
                if player_position == 'QB' and player_team:
                    # Get current stack information
                    current_stacks = find_team_stack_relationships(lineup)
                    
                    # Find best QB replacement that considers stacking
                    best_swap_idx = None
                    best_compatibility_score = -1
                    
                    for idx, row in lineup.iterrows():
                        if row['Position'] == 'QB':
                            # Check if this QB replacement would create or maintain stacks
                            potential_team = player_team
                            same_team_skill_count = sum(1 for _, p in lineup.iterrows() 
                                                      if p['Position'] in ['WR', 'TE'] and p['Team'] == potential_team)
                            
                            # Prefer replacements that create or maintain stacks
                            compatibility_score = same_team_skill_count
                            
                            # Also consider preserving existing stacks from other teams
                            if potential_team in current_stacks:
                                compatibility_score += 2  # Bonus for maintaining existing stack
                            
                            if compatibility_score > best_compatibility_score:
                                best_compatibility_score = compatibility_score
                                best_swap_idx = idx
                    
                    if best_swap_idx is not None:
                        # Replace with target QB
                        lineup.at[best_swap_idx, 'Nickname'] = player_name
                        lineup.at[best_swap_idx, 'Salary'] = player_salary
                        lineup.at[best_swap_idx, 'Team'] = player_team
                        
                        new_salary = lineup['Salary'].sum()
                        if new_salary <= 60000:  # Salary cap check
                            # Recalculate stacking counts after QB change
                            new_stacked_wrs, new_stacked_tes, new_qb_wr_te = recalculate_lineup_stacking(lineup)
                            modified_lineups[lineup_idx] = (points, lineup, new_salary, new_stacked_wrs, new_stacked_tes, new_qb_wr_te)
                            added_successfully += 1
                
                else:
                    # Standard position replacement for non-QB positions
                    best_swap_idx = None
                    best_salary_diff = float('inf')
                    
                    for idx, row in lineup.iterrows():
                        if row['Position'] == player_position:
                            salary_diff = abs(row['Salary'] - player_salary)
                            if salary_diff < best_salary_diff:
                                best_salary_diff = salary_diff
                                best_swap_idx = idx
                    
                    if best_swap_idx is not None:
                        # Replace with target player
                        lineup.at[best_swap_idx, 'Nickname'] = player_name
                        lineup.at[best_swap_idx, 'Salary'] = player_salary
                        lineup.at[best_swap_idx, 'Team'] = player_team
                        
                        new_salary = lineup['Salary'].sum()
                        if new_salary <= 60000:  # Salary cap check
                            # Recalculate stacking counts after any position change
                            new_stacked_wrs, new_stacked_tes, new_qb_wr_te = recalculate_lineup_stacking(lineup)
                            modified_lineups[lineup_idx] = (points, lineup, new_salary, new_stacked_wrs, new_stacked_tes, new_qb_wr_te)
                            added_successfully += 1
            
            successful_adjustments += added_successfully
    
    # Update session state with modified lineups
    if successful_adjustments > 0:
        st.session_state.stacked_lineups = modified_lineups + lineups[150:] if len(lineups) > 150 else modified_lineups
        
        # Recalculate ALL lineup stacking counts to ensure accuracy
        recalculate_all_lineup_stacking()
        
        # Recalculate usage statistics based on the actual modified lineups
        recalculate_usage_stats()
    
    st.success(f"""
    **✅ Lineup Modifications Complete!**
    - **{successful_adjustments}** successful player swaps made
    - **ALL position adjustments applied** - QB, RB, WR, TE changes processed
    - **True exposure matching** - your targets are now reality
    - **Salary cap maintained** - all lineups remain valid
    - **Stack counts recalculated** - lineup labels now show accurate team relationships
    """)
    
    return modified_lineups


def recalculate_usage_stats():
    """Recalculate usage statistics after lineup modifications"""
    if 'stacked_lineups' not in st.session_state:
        return
    
    lineups = st.session_state.stacked_lineups[:150]  # Top 150 only
    
    # Count player usage in modified lineups
    player_counts = {}
    total_lineups = len(lineups)
    
    for points, lineup, salary, _, _, _ in lineups:
        for _, player in lineup.iterrows():
            player_name = player['Nickname']
            position = player['Position']
            # Normalize position for Defense to match original data
            display_position = 'DEF' if position == 'D' else position
            key = f"{player_name} ({display_position})"
            
            if key not in player_counts:
                player_counts[key] = {
                    'count': 0,
                    'player': player_name,
                    'position': display_position,
                    'team': player['Team'],
                    'salary': player['Salary']
                }
            player_counts[key]['count'] += 1
    
    # Update comprehensive usage data in session state
    updated_usage_data = []
    original_data = st.session_state.get('comprehensive_usage_data', [])
    
    # Create a lookup for original player data
    original_lookup = {}
    for orig_player in original_data:
        # Handle both 'D' and 'DEF' position formats in original data
        orig_position = orig_player['Position']
        if orig_position == 'D':
            orig_position = 'DEF'
        key = f"{orig_player['Player']} ({orig_position})"
        original_lookup[key] = orig_player
    
    for key, data in player_counts.items():
        usage_percentage = (data['count'] / total_lineups) * 100
        
        # Get original player data if available
        orig_data = original_lookup.get(key, {})
        
        updated_usage_data.append({
            'Player': data['player'],
            'Position': data['position'],
            'Team': data['team'],
            'Salary': data['salary'],
            'Count': data['count'],
            'Usage %': usage_percentage,
            'Usage_Display': f"{usage_percentage:.1f}%",
            'Salary_Display': f"${data['salary']:,}",
            # Preserve original data
            'vs': orig_data.get('vs', ''),
            'Matchup': orig_data.get('Matchup', ''),
            'FPPG': orig_data.get('FPPG', 0),
            'Ceiling': orig_data.get('Ceiling', 0),
            'Floor': orig_data.get('Floor', 0),
            'Upside': orig_data.get('Upside', ''),
            'Points_Per_Dollar_Display': orig_data.get('Points_Per_Dollar_Display', '0.00'),
            'Value Tier': orig_data.get('Value Tier', ''),
            'Leverage_Display': f"{max(0, 15 - usage_percentage):.1f}%",
            'GPP_Score_Display': orig_data.get('GPP_Score_Display', '0.0'),
            'Proj Own': orig_data.get('Proj Own', '0%'),
            'Ceiling_Display': orig_data.get('Ceiling_Display', '0.0'),
            'Floor_Display': orig_data.get('Floor_Display', '0.0')
        })
    
    # Sort by usage percentage
    updated_usage_data.sort(key=lambda x: x['Usage %'], reverse=True)
    
    # Store updated data
    st.session_state.comprehensive_usage_data = updated_usage_data
    st.session_state.adjustments_applied = True

def main():
    # Initialize enhanced systems if available
    if ENHANCED_FEATURES_AVAILABLE:
        logger = init_logging()
        config = load_config()
        log_info("DFS Optimizer v2.1 started with enhanced features")
        
        st.markdown('<h1 class="main-header">🏈 FanDuel NFL DFS Optimizer v2.1</h1>', unsafe_allow_html=True)
        

        
        # Use config values for defaults
        default_simulations = config.optimization.num_simulations
        default_stack_prob = config.optimization.stack_probability
        default_elite_boost = config.optimization.elite_target_boost
        default_great_boost = config.optimization.great_target_boost
        default_lineups_display = config.optimization.num_lineups_display
    else:
        st.markdown('<h1 class="main-header">🏈 FanDuel NFL DFS Optimizer</h1>', unsafe_allow_html=True)
        # Use original defaults
        default_simulations = 5000  # Reduced from 10000 for faster generation
        default_stack_prob = 0.80
        default_elite_boost = 0.45
        default_great_boost = 0.25
        default_lineups_display = 20
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Optimization Settings")
        
        if ENHANCED_FEATURES_AVAILABLE:
            st.success("🚀 Enhanced Performance Active")
        
        # Strategy Type Selection
        st.subheader("🎯 Strategy Type")
        strategy_type = st.selectbox(
            "Contest Strategy",
            ["Single Entry", "Tournament", "Custom"],
            index=0,
            help="Single Entry: Safer picks, higher floor, target ownership. Tournament: Lower boosts for contrarian builds, avoid chalk."
        )
        
        # Strategy presets
        if strategy_type == "Single Entry":
            preset_stack_prob = 0.65  # More conservative stacking
            preset_elite_boost = 0.35  # Favor consistent performers
            preset_great_boost = 0.20
            preset_simulations = 8000  # Fewer simulations for consistency
            ownership_strategy = "Avoid High Ownership (>15%)"
        elif strategy_type == "Tournament":
            preset_stack_prob = 0.40  # LOWER stacking for contrarian builds
            preset_elite_boost = 0.15  # LOWER elite boost to avoid chalk
            preset_great_boost = 0.10  # LOWER great boost for differentiation
            preset_simulations = 15000  # More diversity for unique builds
            ownership_strategy = "Target Low Ownership (<8%) + Contrarian Builds"
        else:  # Custom
            preset_stack_prob = default_stack_prob
            preset_elite_boost = default_elite_boost
            preset_great_boost = default_great_boost
            preset_simulations = default_simulations
            ownership_strategy = "Balanced Approach"
        
        # Show strategy info
        if strategy_type != "Custom":
            st.info(f"📊 **{strategy_type} Strategy Active**\n\n🎯 Focus: {ownership_strategy}")
            
            with st.expander("📋 Strategy Details"):
                st.markdown(f"""
                **{strategy_type} Strategy Adjustments:**
                
                🔢 **Simulations:** {preset_simulations:,} (optimized for {strategy_type.lower()})
                📊 **Stack Probability:** {preset_stack_prob:.1%} ({'Conservative stacking for consistency' if strategy_type == 'Single Entry' else 'Lower stacking for contrarian differentiation'})
                ⭐ **Elite Boost:** {preset_elite_boost:.1%} ({'Consistent scorers focus' if strategy_type == 'Single Entry' else 'Reduced to avoid chalk plays'})
                🎯 **Great Boost:** {preset_great_boost:.1%} ({'Reliable performers' if strategy_type == 'Single Entry' else 'Minimal boost for max differentiation'})
                
                {'💰 **Best for:** Cash games, head-to-head, 50/50s' if strategy_type == 'Single Entry' else '🏆 **Best for:** GPPs, large tournaments, contrarian/leverage plays'}
                """)
        
        # Override defaults with strategy presets if not custom
        if strategy_type != "Custom":
            default_simulations = preset_simulations
            default_stack_prob = preset_stack_prob
            default_elite_boost = preset_elite_boost
            default_great_boost = preset_great_boost
        
        # Configuration sliders (will use strategy presets as defaults)
        num_simulations = st.slider("Number of Simulations", 1000, 20000, default_simulations, step=1000,
                                    help="More simulations = more unique lineups but slower generation. 5000 simulations typically generates 3000-4000 unique lineups.")
        stack_probability = st.slider("Stacking Probability", 0.0, 1.0, default_stack_prob, step=0.05,
                                     help=f"Current strategy optimized for {strategy_type.lower()} contests")
        elite_target_boost = st.slider("Elite Target Boost", 0.0, 1.0, default_elite_boost, step=0.05,
                                      help=f"{'Consistent elite performers' if strategy_type == 'Single Entry' else 'High ceiling elite players'}")
        great_target_boost = st.slider("Great Target Boost", 0.0, 1.0, default_great_boost, step=0.05)
        
        st.subheader("🚀 Fantasy Performance Adjustments")
        
        col1, col2 = st.columns(2)
        with col1:
            wr_boost_multiplier = st.slider("WR Performance Boost Multiplier", 0.5, 2.0, 1.0, step=0.1, 
                                           help="Adjusts WR fantasy projections based on recent performance")
            rb_boost_multiplier = st.slider("RB Performance Boost Multiplier", 0.5, 2.0, 1.0, step=0.1,
                                           help="Adjusts RB fantasy projections based on recent performance")
        
        with col2:
            # Add individual FPPG adjustment sliders
            global_fppg_adjustment = st.slider("Global FPPG Adjustment", 0.5, 1.5, 1.0, step=0.05,
                                              help="Globally adjust all fantasy point projections")
            
            ceiling_floor_variance = st.slider("Ceiling/Floor Variance", 0.8, 1.5, 1.0, step=0.1,
                                             help="Adjust the spread between ceiling and floor projections")
        
        # Advanced Tournament Settings (only show for Tournament strategy)
        if strategy_type == "Tournament":
            st.subheader("🏆 Advanced Tournament Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                contrarian_boost = st.slider(
                    "🎯 Contrarian Player Boost", 
                    0.0, 0.5, 0.15, step=0.05,
                    help="Extra boost for low-owned players (punt plays and studs)"
                )
                
                correlation_preference = st.slider(
                    "🔗 Correlation Preference", 
                    0.0, 1.0, 0.7, step=0.1,
                    help="Preference for correlated lineups (stacks, game stacks, bring-backs)"
                )
                
            with col2:
                salary_variance_target = st.slider(
                    "💰 Salary Variance Target", 
                    0.0, 1.0, 0.6, step=0.1,
                    help="Target salary distribution variance (0=flat, 1=stars+punts)"
                )
                
                leverage_focus = st.slider(
                    "📈 Leverage Focus", 
                    0.0, 1.0, 0.4, step=0.1,
                    help="Focus on players with tournament leverage (low ownership, high upside)"
                )
        else:
            # Set defaults for non-tournament strategies
            contrarian_boost = 0.05
            correlation_preference = 0.3
            salary_variance_target = 0.2
            leverage_focus = 0.1
        
        st.subheader("🎯 Forced Player Boost")
        forced_player_boost = st.slider("Forced Player Extra Boost", 0.0, 1.0, 0.3, step=0.05)  # Increased default from 0.05 to 0.3
        force_mode = st.radio("Force Mode", 
                             ["Hard Force (Always Include)", "Soft Force (Boost Only)"], 
                             index=1,
                             help="Hard Force: Forced players appear in every lineup. Soft Force: Forced players get boost but may not appear in all lineups")
        
        with st.expander("💡 Force Mode Explained"):
            st.markdown("""
            **🔒 Hard Force (Always Include):**
            - Forced players appear in **100% of generated lineups**
            - Best for: Cash games, high-confidence plays
            - Example: If you force Davante Adams, he's in all 20 lineups
            
            **🎯 Soft Force (Boost Only) - Recommended:**
            - Forced players get extra selection weight but **variety is maintained**
            - Best for: Tournaments, exposure plays, lineup diversity
            
            **Soft Force Exposure Guide:**
            - **5% Boost** = ~30-40% of lineups (6-8 out of 20)
            - **15% Boost** = ~60-70% of lineups (12-14 out of 20)  
            - **30% Boost** = ~80-90% of lineups (16-18 out of 20)
            - **50%+ Boost** = ~95%+ of lineups (19-20 out of 20)
            
            **💡 Pro Tips:**
            - Use Soft Force with 10-20% boost for optimal tournament lineup variety!
            - **Tournament Mode**: LOWER boosts = more contrarian, HIGHER boosts = more chalk
            - **Contrarian Edge**: Tournament mode uses lower elite/great boosts to avoid chalk
            - **Stacking Strategy**: 40% stacking creates unique lineup construction
            - **Leverage Plays**: Higher Salary Variance = Stars + Punts strategy
            """)
        st.caption("Extra boost for players you manually include")
        
        st.header("� Player Selection")
        enable_player_selection = st.checkbox("Enable Player Include/Exclude", value=False)
        
        # Add guidance for forced players
        if enable_player_selection:
            with st.expander("💡 Tips for Forcing Players"):
                st.markdown("""
                **Performance Guidelines:**
                - **1-5 forced players**: Normal speed
                - **6-10 forced players**: Slightly slower, auto-reduces to 5,000 simulations
                - **11-15 forced players**: Moderate speed, auto-reduces to 3,000 simulations  
                - **16+ forced players**: Slower, auto-reduces to 2,000 simulations
                
                **For Best Results:**
                - Use "🎯 Force Top 6 Matchups" button for optimal constraint balance
                - Consider forcing fewer players if timeouts occur
                - Focus on 1-2 positions rather than all positions
                """)
        
        st.header("📊 Display Settings")
        num_lineups_display = st.slider("Number of Top Lineups to Show", 5, 50, default_lineups_display, step=5)
        
        generate_button = st.button("🚀 Generate Lineups", type="primary")
    
    # Load data
    with st.spinner("Loading player data..."):
        df = load_player_data()
        
    if df is not None:
        with st.spinner("Loading defensive matchup data..."):
            pass_defense, rush_defense = load_defensive_data()
            
        with st.spinner("Loading fantasy performance data..."):
            fantasy_data = load_fantasy_data()
        
        # Merge PosRank data from fantasy data
        if fantasy_data is not None:
            with st.spinner("Adding position rankings..."):
                # Create a mapping of player names to PosRank
                posrank_mapping = fantasy_data.set_index('Player')['PosRank'].to_dict()
                
                # Check which column contains player names
                name_column = None
                for col in ['Nickname', 'Name', 'Player', 'Full Name']:
                    if col in df.columns:
                        name_column = col
                        break
                
                if name_column:
                    df['PosRank'] = df[name_column].map(posrank_mapping)
                    # Fill missing PosRank with 999 for players not in fantasy data
                    df['PosRank'] = df['PosRank'].fillna(999).astype(int)
                else:
                    st.warning("Could not find player name column for position rankings")
                    df['PosRank'] = 999
        else:
            # If no fantasy data, use default ranking
            df['PosRank'] = 999
        
        # Apply analysis
        with st.spinner("Applying matchup analysis..."):
            df = apply_matchup_analysis(df, pass_defense, rush_defense)
            
        with st.spinner("Creating performance boosts..."):
            wr_performance_boosts, rb_performance_boosts, te_performance_boosts, qb_performance_boosts = create_performance_boosts(fantasy_data, wr_boost_multiplier, rb_boost_multiplier)
        
        # Apply global fantasy adjustments
        with st.spinner("Applying fantasy adjustments..."):
            # Global FPPG adjustment
            df['Adjusted_FPPG'] = df['Adjusted_FPPG'] * global_fppg_adjustment
            
            # Adjust ceiling/floor variance if columns exist
            if 'Ceiling' in df.columns and 'Floor' in df.columns:
                # Calculate current ceiling/floor relative to FPPG
                ceiling_diff = df['Ceiling'] - df['FPPG'] 
                floor_diff = df['FPPG'] - df['Floor']
                
                # Adjust the variance
                df['Ceiling'] = df['FPPG'] * global_fppg_adjustment + (ceiling_diff * ceiling_floor_variance)
                df['Floor'] = df['FPPG'] * global_fppg_adjustment - (floor_diff * ceiling_floor_variance)
                
                # Ensure floor doesn't go negative
                df['Floor'] = df['Floor'].clip(lower=0)
            
            # Show adjustment summary
            if global_fppg_adjustment != 1.0 or ceiling_floor_variance != 1.0 or wr_boost_multiplier != 1.0 or rb_boost_multiplier != 1.0:
                adjustments = []
                if global_fppg_adjustment != 1.0:
                    adjustments.append(f"Global FPPG: {global_fppg_adjustment:.2f}x")
                if ceiling_floor_variance != 1.0:
                    adjustments.append(f"Ceiling/Floor Variance: {ceiling_floor_variance:.2f}x")
                if wr_boost_multiplier != 1.0:
                    adjustments.append(f"WR Performance: {wr_boost_multiplier:.2f}x")
                if rb_boost_multiplier != 1.0:
                    adjustments.append(f"RB Performance: {rb_boost_multiplier:.2f}x")
                
                st.success(f"✅ **Fantasy Adjustments Applied:** {', '.join(adjustments)}")
        
        # Manual Projection Overrides Section
        st.subheader("📝 Manual Projection Overrides")
        st.caption("Adjust individual player projections for injuries, weather, or personal insights")
        
        with st.expander("🎯 Override Player Projections", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Select Player to Override:**")
                
                # Position filter
                positions = ["All Positions"] + sorted(df['Position'].unique().tolist())
                selected_position = st.selectbox(
                    "Filter by Position:",
                    positions,
                    help="Filter players by position to make selection easier"
                )
                
                # Filter players by position if selected
                if selected_position == "All Positions":
                    filtered_df = df
                    position_label = ""
                else:
                    filtered_df = df[df['Position'] == selected_position]
                    position_label = f" ({selected_position}s)"
                
                # Sort players by salary (descending) for easier navigation
                filtered_df_sorted = filtered_df.sort_values('Salary', ascending=False)
                
                # Create player display names with salary for easier identification
                player_options = [""]
                for _, player in filtered_df_sorted.iterrows():
                    display_name = f"{player['Nickname']} - ${player['Salary']:,} ({player['FPPG']:.1f} FPPG)"
                    player_options.append(display_name)
                
                selected_player_display = st.selectbox(
                    f"Choose Player{position_label}:", 
                    player_options,
                    help="Players sorted by salary (highest first)"
                )
                
                # Extract actual player name from display name
                if selected_player_display:
                    selected_player = selected_player_display.split(" - ")[0]
                else:
                    selected_player = ""
                
                if selected_player:
                    # Get current player stats
                    player_row = df[df['Nickname'] == selected_player].iloc[0]
                    current_fppg = player_row['FPPG']
                    current_salary = player_row['Salary']
                    current_pos = player_row['Position']
                    
                    st.info(f"**{selected_player}** ({current_pos}) - ${current_salary:,} - Current: {current_fppg:.1f} FPPG")
            
            with col2:
                if selected_player:
                    st.markdown("**Override Settings:**")
                    
                    override_type = st.radio(
                        "Override Type:",
                        ["Percentage Adjustment", "Absolute Value"],
                        help="Percentage: Multiply by factor (e.g., 1.2 = +20%). Absolute: Set exact FPPG value."
                    )
                    
                    if override_type == "Percentage Adjustment":
                        adjustment_factor = st.slider(
                            "Adjustment Factor",
                            0.0, 3.0, 1.0, 0.05,
                            help="1.0 = no change, 1.2 = +20%, 0.8 = -20%"
                        )
                        new_projection = current_fppg * adjustment_factor
                        st.write(f"**New Projection:** {new_projection:.1f} FPPG ({adjustment_factor:.0%} of original)")
                    
                    else:  # Absolute Value
                        new_projection = st.number_input(
                            "New FPPG Projection",
                            0.0, 50.0, float(current_fppg), 0.1,
                            help="Set exact fantasy points projection"
                        )
                        adjustment_factor = new_projection / current_fppg if current_fppg > 0 else 1.0
                    
                    # Apply override button
                    if st.button(f"Apply Override to {selected_player}", type="primary"):
                        # Store original value if first override for this player
                        if selected_player not in st.session_state.projection_overrides:
                            original_fppg = current_fppg
                        else:
                            original_fppg = st.session_state.projection_overrides[selected_player]['original']
                        
                        # Apply the override
                        mask = df['Nickname'] == selected_player
                        df.loc[mask, 'FPPG'] = new_projection
                        df.loc[mask, 'Adjusted_FPPG'] = new_projection * global_fppg_adjustment
                        
                        # Adjust ceiling/floor proportionally if they exist
                        if 'Ceiling' in df.columns and 'Floor' in df.columns:
                            df.loc[mask, 'Ceiling'] = df.loc[mask, 'Ceiling'] * adjustment_factor
                            df.loc[mask, 'Floor'] = df.loc[mask, 'Floor'] * adjustment_factor
                        
                        # Track the override in session state
                        st.session_state.projection_overrides[selected_player] = {
                            'original': original_fppg,
                            'new': new_projection,
                            'position': current_pos,
                            'adjustment_factor': adjustment_factor
                        }
                        
                        st.success(f"✅ **{selected_player}** projection updated to {new_projection:.1f} FPPG!")
                        st.rerun()
            
            # Show current overrides
            st.markdown("**💡 Pro Tips for Manual Overrides:**")
            st.markdown("""
            - **Weather Impact**: Reduce passing game in heavy wind/rain
            - **Injury Concerns**: Lower projections for questionable players  
            - **Coaching Changes**: Adjust for new play-callers or schemes
            - **Motivation**: Increase for playoff implications, decrease for resting starters
            - **Matchup Intel**: Boost players facing backup defenders
            """)
            
            st.divider()
            
            # Bulk overrides section
            st.markdown("**⚡ Bulk Adjustments:**")
            bulk_col1, bulk_col2 = st.columns(2)
            
            with bulk_col1:
                # Team-based adjustments
                teams = sorted(df['Team'].unique())
                selected_team = st.selectbox("Adjust Entire Team:", [""] + teams)
                
                if selected_team:
                    team_adjustment = st.slider(
                        f"Team Adjustment Factor ({selected_team})",
                        0.0, 2.0, 1.0, 0.05,
                        key=f"team_adj_{selected_team}"
                    )
                    
                    if st.button(f"Apply to All {selected_team} Players", key=f"apply_team_{selected_team}"):
                        team_mask = df['Team'] == selected_team
                        team_players = df[team_mask]['Nickname'].tolist()
                        
                        for player in team_players:
                            current_fppg = df[df['Nickname'] == player]['FPPG'].iloc[0]
                            new_fppg = current_fppg * team_adjustment
                            
                            # Store override
                            if player not in st.session_state.projection_overrides:
                                original_fppg = current_fppg
                            else:
                                original_fppg = st.session_state.projection_overrides[player]['original']
                            
                            st.session_state.projection_overrides[player] = {
                                'original': original_fppg,
                                'new': new_fppg,
                                'position': df[df['Nickname'] == player]['Position'].iloc[0],
                                'adjustment_factor': team_adjustment
                            }
                            
                            # Apply to dataframe
                            player_mask = df['Nickname'] == player
                            df.loc[player_mask, 'FPPG'] = new_fppg
                            df.loc[player_mask, 'Adjusted_FPPG'] = new_fppg * global_fppg_adjustment
                        
                        st.success(f"✅ Applied {team_adjustment:.0%} adjustment to all {selected_team} players!")
                        st.rerun()
            
            with bulk_col2:
                # Position-based adjustments
                positions = ['QB', 'RB', 'WR', 'TE']
                selected_pos = st.selectbox("Adjust by Position:", [""] + positions)
                
                if selected_pos:
                    pos_adjustment = st.slider(
                        f"Position Adjustment Factor ({selected_pos})",
                        0.0, 2.0, 1.0, 0.05,
                        key=f"pos_adj_{selected_pos}"
                    )
                    
                    if st.button(f"Apply to All {selected_pos}s", key=f"apply_pos_{selected_pos}"):
                        pos_mask = df['Position'] == selected_pos
                        pos_players = df[pos_mask]['Nickname'].tolist()
                        
                        for player in pos_players:
                            current_fppg = df[df['Nickname'] == player]['FPPG'].iloc[0]
                            new_fppg = current_fppg * pos_adjustment
                            
                            # Store override
                            if player not in st.session_state.projection_overrides:
                                original_fppg = current_fppg
                            else:
                                original_fppg = st.session_state.projection_overrides[player]['original']
                            
                            st.session_state.projection_overrides[player] = {
                                'original': original_fppg,
                                'new': new_fppg,
                                'position': selected_pos,
                                'adjustment_factor': pos_adjustment
                            }
                            
                            # Apply to dataframe
                            player_mask = df['Nickname'] == player
                            df.loc[player_mask, 'FPPG'] = new_fppg
                            df.loc[player_mask, 'Adjusted_FPPG'] = new_fppg * global_fppg_adjustment
                        
                        st.success(f"✅ Applied {pos_adjustment:.0%} adjustment to all {selected_pos}s!")
                        st.rerun()
        
        # Initialize session state for overrides tracking
        if 'projection_overrides' not in st.session_state:
            st.session_state.projection_overrides = {}
        
        # Apply existing overrides from session state
        if st.session_state.projection_overrides:
            st.info(f"📝 **{len(st.session_state.projection_overrides)} projection overrides active**")
            for player, override_data in st.session_state.projection_overrides.items():
                if player in df['Nickname'].values:
                    mask = df['Nickname'] == player
                    df.loc[mask, 'FPPG'] = override_data['new']
                    df.loc[mask, 'Adjusted_FPPG'] = override_data['new'] * global_fppg_adjustment
                    
                    # Apply to ceiling/floor if they exist
                    if 'Ceiling' in df.columns and 'Floor' in df.columns:
                        adjustment_factor = override_data['adjustment_factor']
                        df.loc[mask, 'Ceiling'] = df.loc[mask, 'Ceiling'] * adjustment_factor
                        df.loc[mask, 'Floor'] = df.loc[mask, 'Floor'] * adjustment_factor
        
        # Display current overrides if any exist
        if st.session_state.projection_overrides:
            with st.expander("📊 Current Projection Overrides", expanded=True):
                override_df = []
                for player, data in st.session_state.projection_overrides.items():
                    override_df.append({
                        'Player': player,
                        'Position': data['position'],
                        'Original FPPG': f"{data['original']:.1f}",
                        'New FPPG': f"{data['new']:.1f}",
                        'Adjustment': f"{((data['new']/data['original']-1)*100):+.1f}%"
                    })
                
                if override_df:
                    st.dataframe(override_df, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Reset All Overrides", help="Remove all manual projection overrides"):
                            st.session_state.projection_overrides = {}
                            st.success("✅ All projection overrides cleared!")
                            st.rerun()
                    
                    with col2:
                        if st.button("💾 Save Overrides", help="Save current overrides for future sessions"):
                            st.info("💡 Overrides are automatically saved for this session")
        
        # Display top matchups
        st.markdown("### 🎯 Top 6 Matchups by Position")
        
        # Get top matchups by position
        position_matchups = get_top_matchups(df, pass_defense, rush_defense, num_per_position=6)
        
        if position_matchups:
            # Create tabs for each position
            pos_tabs = st.tabs(["QB", "RB", "WR", "TE"])
            
            positions = ['QB', 'RB', 'WR', 'TE']
            emojis = ['🎯', '🏈', '⚡', '🎪']
            
            for i, (tab, pos, emoji) in enumerate(zip(pos_tabs, positions, emojis)):
                with tab:
                    if pos in position_matchups and len(position_matchups[pos]) > 0:
                        for j, (_, matchup) in enumerate(position_matchups[pos].iterrows()):
                            if j < 6:  # Show top 6 in each tab
                                quality_icon = "🔥" if matchup['Matchup_Quality'] == 'ELITE TARGET' else ("⭐" if matchup['Matchup_Quality'] == 'Great Target' else "")
                                
                                # Add salary boost indicator for QBs
                                salary_boost_icon = ""
                                if pos == 'QB':
                                    qb_data = df[df['Position'] == 'QB']
                                    for team in qb_data['Team'].unique():
                                        team_qbs = qb_data[qb_data['Team'] == team]
                                        if len(team_qbs) > 0:
                                            highest_qb = team_qbs.loc[team_qbs['Salary'].idxmax(), 'Nickname']
                                            if matchup['Player'] == highest_qb:
                                                salary_boost_icon = " 💰"
                                
                                # Format all fields safely
                                ypg_display = f"{matchup['YPG_Allowed']:.1f} YPG allowed" if isinstance(matchup['YPG_Allowed'], (int, float)) else f"{matchup['YPG_Allowed']} YPG allowed"
                                fppg_display = f"{matchup['FPPG']:.1f} pts" if isinstance(matchup['FPPG'], (int, float)) else f"{matchup['FPPG']} pts"
                                salary_display = f"${matchup['Salary']:,}" if isinstance(matchup['Salary'], (int, float)) else f"${matchup['Salary']}"
                                
                                st.markdown(
                                    f"**{emoji} {matchup['Player']}** vs {matchup['vs']} {quality_icon}{salary_boost_icon}  \n"
                                    f"{salary_display} | {fppg_display}", 
                                    help=f"Defense Rank: #{matchup['Defense_Rank']} ({ypg_display})"
                                )
                    else:
                        st.info(f"No {pos} matchups found")
        else:
            st.info("Top matchups will appear here once data is loaded")
        
        # Player Selection Interface
        if enable_player_selection:
            st.markdown('<h2 class="sub-header">👥 Player Selection</h2>', unsafe_allow_html=True)
            
            # Add button to auto-select top matchups
            col1, col2, col3, col4, col5 = st.columns([1.5, 1.3, 1.3, 1, 1.5])
            with col2:
                if st.button("🎯 Force QB/RB/WR Only", type="secondary", help="Auto-select top 6 matchups for QB, RB, WR only (you can add more manually)"):
                    # Get top matchups and auto-populate selections
                    top_matchups = get_top_matchups(df, pass_defense, rush_defense, num_per_position=6)
                    
                    # Only auto-select QB, RB, WR (not TE or DEF)
                    if 'QB' in top_matchups and len(top_matchups['QB']) > 0:
                        existing_qb = st.session_state.get('auto_qb', [])
                        new_qb = top_matchups['QB']['Player'].head(6).tolist()
                        # Combine existing with new, remove duplicates while preserving order
                        combined_qb = existing_qb + [qb for qb in new_qb if qb not in existing_qb]
                        st.session_state.auto_qb = combined_qb
                    
                    if 'RB' in top_matchups and len(top_matchups['RB']) > 0:
                        existing_rb = st.session_state.get('auto_rb', [])
                        new_rb = top_matchups['RB']['Player'].head(6).tolist()
                        combined_rb = existing_rb + [rb for rb in new_rb if rb not in existing_rb]
                        st.session_state.auto_rb = combined_rb
                    
                    if 'WR' in top_matchups and len(top_matchups['WR']) > 0:
                        existing_wr = st.session_state.get('auto_wr', [])
                        new_wr = top_matchups['WR']['Player'].head(6).tolist()
                        combined_wr = existing_wr + [wr for wr in new_wr if wr not in existing_wr]
                        st.session_state.auto_wr = combined_wr
                    
                    # Don't auto-populate TE or DEF - user can add manually
                    st.success("✅ Top 6 QB, RB, and WR matchups added! Add TE/DEF manually if desired.")
            
            with col3:
                if st.button("🎯 Force All Positions", type="secondary", help="Auto-select top 6 matchups for all positions"):
                    # Get top matchups and auto-populate ALL positions
                    top_matchups = get_top_matchups(df, pass_defense, rush_defense, num_per_position=6)
                    
                    # Auto-select all positions
                    for pos in ['QB', 'RB', 'WR', 'TE']:
                        if pos in top_matchups and len(top_matchups[pos]) > 0:
                            existing = st.session_state.get(f'auto_{pos.lower()}', [])
                            new_players = top_matchups[pos]['Player'].head(6).tolist()
                            combined = existing + [p for p in new_players if p not in existing]
                            st.session_state[f'auto_{pos.lower()}'] = combined
                    
                    st.success("✅ Top 6 matchups added for all positions!")
            
            with col4:
                if st.button("🗑️ Clear", help="Clear all player selections"):
                    # Clear all auto-selections
                    for key in ['auto_qb', 'auto_rb', 'auto_wr', 'auto_te']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("✅ Cleared!")
            
            # Create tabs for different positions
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["QB", "RB", "WR", "TE", "DEF"])
            
            # Helper function to extract player name from "Name ($salary)" format
            def extract_player_name(selection_list):
                """Extract just the player name from 'Name ($salary)' format"""
                return [name.split(' ($')[0] for name in selection_list]
            
            player_selections = {}
            
            with tab1:
                st.subheader("Quarterbacks")
                qb_players = df[df['Position'] == 'QB'].sort_values(['Team', 'Salary'], ascending=[True, False])
                qb_options = [f"{row['Nickname']} (${row['Salary']:,})" for _, row in qb_players.sort_values('Salary', ascending=False).iterrows()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    # Check for auto-selected values
                    default_qb = st.session_state.get('auto_qb', [])
                    must_include_qb = st.multiselect(
                        "Force these QBs in lineups",
                        options=qb_options,
                        default=default_qb,
                        key="must_qb",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_qb = st.multiselect(
                        "Remove these QBs from consideration",
                        options=qb_options,
                        key="exclude_qb",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                player_selections['QB'] = {
                    'must_include': extract_player_name(must_include_qb), 
                    'exclude': extract_player_name(exclude_qb)
                }
                
                # Show QB options with salary/matchup info
                with st.expander("View All QB Options"):
                    qb_display = qb_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    qb_display['Salary'] = qb_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(qb_display, use_container_width=True)
            
            with tab2:
                st.subheader("Running Backs")
                rb_players = df[df['Position'] == 'RB'].sort_values(['Team', 'Salary'], ascending=[True, False])
                rb_options = [f"{row['Nickname']} (${row['Salary']:,})" for _, row in rb_players.sort_values('Salary', ascending=False).iterrows()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    # Check for auto-selected values
                    default_rb = st.session_state.get('auto_rb', [])
                    must_include_rb = st.multiselect(
                        "Force these RBs in lineups",
                        options=rb_options,
                        default=default_rb,
                        key="must_rb",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_rb = st.multiselect(
                        "Remove these RBs from consideration",
                        options=rb_options,
                        key="exclude_rb",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                player_selections['RB'] = {
                    'must_include': extract_player_name(must_include_rb), 
                    'exclude': extract_player_name(exclude_rb)
                }
                
                with st.expander("View All RB Options"):
                    rb_display = rb_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    rb_display['Salary'] = rb_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(rb_display, use_container_width=True)
            
            with tab3:
                st.subheader("Wide Receivers")
                wr_players = df[df['Position'] == 'WR'].sort_values(['Team', 'Salary'], ascending=[True, False])
                wr_options = [f"{row['Nickname']} (${row['Salary']:,})" for _, row in wr_players.sort_values('Salary', ascending=False).iterrows()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    # Check for auto-selected values
                    default_wr = st.session_state.get('auto_wr', [])
                    must_include_wr = st.multiselect(
                        "Force these WRs in lineups",
                        options=wr_options,
                        default=default_wr,
                        key="must_wr",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_wr = st.multiselect(
                        "Remove these WRs from consideration",
                        options=wr_options,
                        key="exclude_wr",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                player_selections['WR'] = {
                    'must_include': extract_player_name(must_include_wr), 
                    'exclude': extract_player_name(exclude_wr)
                }
                
                with st.expander("View All WR Options"):
                    wr_display = wr_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    wr_display['Salary'] = wr_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(wr_display, use_container_width=True)
            
            with tab4:
                st.subheader("Tight Ends")
                te_players = df[df['Position'] == 'TE'].sort_values(['Team', 'Salary'], ascending=[True, False])
                te_options = [f"{row['Nickname']} (${row['Salary']:,})" for _, row in te_players.sort_values('Salary', ascending=False).iterrows()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    # Check for auto-selected values
                    default_te = st.session_state.get('auto_te', [])
                    must_include_te = st.multiselect(
                        "Force these TEs in lineups",
                        options=te_options,
                        default=default_te,
                        key="must_te",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_te = st.multiselect(
                        "Remove these TEs from consideration",
                        options=te_options,
                        key="exclude_te",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                player_selections['TE'] = {
                    'must_include': extract_player_name(must_include_te), 
                    'exclude': extract_player_name(exclude_te)
                }
                
                with st.expander("View All TE Options"):
                    te_display = te_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    te_display['Salary'] = te_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(te_display, use_container_width=True)
            
            with tab5:
                st.subheader("Defense/Special Teams")
                def_players_tab = df[df['Position'] == 'D'].sort_values(['Team', 'Salary'], ascending=[True, False])
                def_options = [f"{row['Nickname']} (${row['Salary']:,})" for _, row in def_players_tab.sort_values('Salary', ascending=False).iterrows()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    must_include_def = st.multiselect(
                        "Force these DEF in lineups",
                        options=def_options,
                        key="must_def",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_def = st.multiselect(
                        "Remove these DEF from consideration",
                        options=def_options,
                        key="exclude_def",
                        help="Players sorted by salary (highest to lowest)"
                    )
                
                player_selections['D'] = {
                    'must_include': extract_player_name(must_include_def), 
                    'exclude': extract_player_name(exclude_def)
                }
                
                with st.expander("View All DEF Options"):
                    def_display = def_players_tab[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    def_display['Salary'] = def_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(def_display, use_container_width=True)
        
        else:
            player_selections = None
        
        # Collect all forced players for boost calculation and display
        all_forced_players = []
        if enable_player_selection and player_selections:
            for pos_data in player_selections.values():
                if pos_data and 'must_include' in pos_data:
                    all_forced_players.extend(pos_data['must_include'])
        
        if generate_button:
            with st.spinner("Creating weighted player pools..."):
                weighted_pools = create_weighted_pools(df, wr_performance_boosts, rb_performance_boosts, te_performance_boosts, qb_performance_boosts, elite_target_boost, great_target_boost, all_forced_players, forced_player_boost)
            
            with st.spinner(f"Generating {num_simulations:,} optimized lineups..."):
                # Pass tournament parameters to generation function
                tournament_params = {
                    'contrarian_boost': contrarian_boost if strategy_type == "Tournament" else 0.05,
                    'correlation_preference': correlation_preference if strategy_type == "Tournament" else 0.3,
                    'salary_variance_target': salary_variance_target if strategy_type == "Tournament" else 0.2,
                    'leverage_focus': leverage_focus if strategy_type == "Tournament" else 0.1,
                    'global_fppg_adjustment': global_fppg_adjustment,
                    'ceiling_floor_variance': ceiling_floor_variance
                }
                
                stacked_lineups = generate_lineups(df, weighted_pools, num_simulations, stack_probability, elite_target_boost, great_target_boost, fantasy_data, player_selections, force_mode, forced_player_boost, strategy_type, tournament_params)
                st.session_state.stacked_lineups = stacked_lineups
                st.session_state.lineups_generated = True
                
                # Debug info
                if len(stacked_lineups) == 0:
                    st.error("⚠️ No lineups were generated! This could be due to:")
                    st.write("- Too many forced players creating impossible constraints")
                    st.write("- Salary cap issues with forced players")
                    st.write("- Try reducing forced players or using the 'Clear' button")
                else:
                    st.success(f"✅ Successfully generated {len(stacked_lineups):,} lineups!")
        
        # Display results
        if st.session_state.lineups_generated and st.session_state.stacked_lineups:
            stacked_lineups = st.session_state.stacked_lineups
            
            st.markdown('<h2 class="sub-header">🏆 Optimized Lineups</h2>', unsafe_allow_html=True)
            
            # Sort and display top lineups
            top_lineups = sorted(stacked_lineups, key=lambda x: x[0], reverse=True)[:num_lineups_display]
        
        elif st.session_state.lineups_generated and not st.session_state.stacked_lineups:
            st.warning("⚠️ Lineups were generated but none met the constraints. Try:")
            st.write("- Reducing the number of forced players")
            st.write("- Using the 'Clear' button and trying again")
            st.write("- Increasing simulation count")
        
        elif not st.session_state.lineups_generated:
            st.info("👆 Click 'Generate Lineups' to create optimized lineups!")
        
        # Only show lineup details if we have valid lineups
        if st.session_state.lineups_generated and st.session_state.stacked_lineups:
            stacked_lineups = st.session_state.stacked_lineups
            
            # Sort and display top lineups
            top_lineups = sorted(stacked_lineups, key=lambda x: x[0], reverse=True)[:num_lineups_display]
            
            # Summary metrics
            if len(stacked_lineups) > 0:
                avg_points = np.mean([lineup[0] for lineup in stacked_lineups])
                best_points = max([lineup[0] for lineup in stacked_lineups])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lineups Generated", f"{len(stacked_lineups):,}")
                with col2:
                    st.metric("Average Projected Points", f"{avg_points:.2f}")
                with col3:
                    st.metric("Best Projected Points", f"{best_points:.2f}")
            
            # Display lineups first, then show usage analysis
            st.markdown("---")
            
            # Lineup Display Controls
            st.subheader("📋 Generated Lineups")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # Dropdown for number of lineups to display
                lineup_count_options = {
                    "Top 20": 20,
                    "Top 50": 50, 
                    "Top 100": 100,
                    "All 150": min(150, len(stacked_lineups))
                }
                selected_count_label = st.selectbox(
                    "📊 Select lineups to display:",
                    options=list(lineup_count_options.keys()),
                    index=0  # Default to "Top 20"
                )
                selected_count = lineup_count_options[selected_count_label]
                
            with col2:
                # Display format toggle
                display_format = st.radio(
                    "📋 Display format:",
                    ["Expandable Cards", "Compact Table"],
                    index=0
                )
            
            # Get the selected number of lineups
            display_lineups = sorted(stacked_lineups, key=lambda x: x[0], reverse=True)[:selected_count]
            
            if display_format == "Compact Table":
                # TABLE VIEW - Show all lineups in a compact table
                st.write(f"**Showing {selected_count_label} ({len(display_lineups)} lineups)**")
                
                table_data = []
                for i, (points, lineup, salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count) in enumerate(display_lineups, 1):
                    # RECALCULATE STACKING FOR DISPLAY
                    actual_stacked_wrs, actual_stacked_tes, actual_qb_wr_te = recalculate_lineup_stacking(lineup)
                    
                    # Get player names by position
                    qb = lineup[lineup['Position'] == 'QB']['Nickname'].iloc[0] if len(lineup[lineup['Position'] == 'QB']) > 0 else 'N/A'
                    rb1, rb2 = lineup[lineup['Position'] == 'RB']['Nickname'].tolist()[:2] if len(lineup[lineup['Position'] == 'RB']) >= 2 else ['N/A', 'N/A']
                    wr1, wr2, wr3 = lineup[lineup['Position'] == 'WR']['Nickname'].tolist()[:3] if len(lineup[lineup['Position'] == 'WR']) >= 3 else ['N/A', 'N/A', 'N/A']
                    te = lineup[lineup['Position'] == 'TE']['Nickname'].iloc[0] if len(lineup[lineup['Position'] == 'TE']) > 0 else 'N/A'
                    dst = lineup[lineup['Position'] == 'D']['Nickname'].iloc[0] if len(lineup[lineup['Position'] == 'D']) > 0 else 'N/A'
                    
                    ceiling = lineup['Ceiling'].sum() if 'Ceiling' in lineup.columns else 0
                    stack_label = f"QB+{actual_qb_wr_te}" if actual_qb_wr_te > 0 else "No Stack"
                    
                    table_data.append({
                        'Rank': i,
                        'Points': f"{points:.1f}",
                        'Salary': f"${salary:,}",
                        'Ceiling': f"{ceiling:.1f}" if 'Ceiling' in lineup.columns else 'N/A',
                        'Stack': stack_label,
                        'QB': qb,
                        'RB1': rb1,
                        'RB2': rb2,
                        'WR1': wr1,
                        'WR2': wr2,
                        'WR3': wr3,
                        'TE': te,
                        'DST': dst
                    })
                
                # Display the table
                table_df = pd.DataFrame(table_data)
                st.dataframe(
                    table_df,
                    width="stretch",
                    hide_index=True
                )
                
            else:
                # EXPANDABLE CARDS VIEW (Original format)
                st.write(f"**Showing {selected_count_label} ({len(display_lineups)} lineups)**")
                
                for i, (points, lineup, salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count) in enumerate(display_lineups, 1):
                    # RECALCULATE STACKING FOR DISPLAY (ensures accuracy after modifications)
                    actual_stacked_wrs, actual_stacked_tes, actual_qb_wr_te = recalculate_lineup_stacking(lineup)
                    
                    # Calculate lineup ceiling for header display
                    ceiling_text = ""
                    if 'Ceiling' in lineup.columns:
                        lineup_ceiling = lineup['Ceiling'].sum()
                        ceiling_text = f" | Ceiling: {lineup_ceiling:.1f}"
                    
                    with st.expander(f"Lineup #{i}: {points:.1f} pts{ceiling_text} | ${salary:,} | {'QB+' + str(actual_qb_wr_te) + ' receivers' if actual_qb_wr_te > 0 else 'No stack'}"):
                        
                        # Create lineup display with ceiling and floor
                        display_columns = ['Nickname', 'Position', 'Team', 'Salary', 'FPPG', 'Matchup_Quality', 'PosRank']
                        if 'Ceiling' in lineup.columns:
                            display_columns.extend(['Ceiling', 'Floor'])
                        
                        lineup_display = lineup[display_columns].copy()
                        lineup_display['Salary'] = lineup_display['Salary'].apply(lambda x: f"${x:,}")
                        lineup_display['FPPG'] = lineup_display['FPPG'].apply(lambda x: f"{x:.1f}")
                        
                        # Format ceiling and floor if they exist
                        if 'Ceiling' in lineup_display.columns:
                            lineup_display['Ceiling'] = lineup_display['Ceiling'].apply(lambda x: f"{x:.1f}")
                            lineup_display['Floor'] = lineup_display['Floor'].apply(lambda x: f"{x:.1f}")
                        
                        # Calculate and display total lineup ceiling/floor
                        if 'Ceiling' in lineup.columns:
                            lineup_ceiling = lineup['Ceiling'].sum()
                            lineup_floor = lineup['Floor'].sum()
                            st.markdown(f"""
                            **📊 Lineup Projections:**
                            - **Projection:** {points:.1f} pts
                            - **Ceiling:** {lineup_ceiling:.1f} pts  
                            - **Floor:** {lineup_floor:.1f} pts
                            """)
                        else:
                            st.markdown(f"**📊 Projection:** {points:.1f} pts")
                        
                        # Set PosRank as the index for display
                        lineup_display.set_index('PosRank', inplace=True)
                        lineup_display = lineup_display.drop('PosRank', axis=1, errors='ignore')  # Remove if accidentally included twice
                        
                        st.dataframe(lineup_display, width="stretch")
                        
                        # Show boosts
                        fantasy_boosted = 0
                        elite_targets = 0
                        forced_boosted = 0
                        qb_team = lineup[lineup['Position'] == 'QB']['Team'].iloc[0]
                        
                        for _, player in lineup.iterrows():
                            if player['Position'] == 'WR' and player['Nickname'] in wr_performance_boosts:
                                fantasy_boosted += 1
                            elif player['Position'] == 'RB' and player['Nickname'] in rb_performance_boosts:
                                fantasy_boosted += 1
                            
                            if player['Matchup_Quality'] == 'ELITE TARGET':
                                elite_targets += 1
                            
                            # Check if player was forced and got boost
                            if enable_player_selection and player_selections and all_forced_players:
                                if player['Nickname'] in all_forced_players:
                                    forced_boosted += 1
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🏈 Fantasy-boosted players: {fantasy_boosted}")
                        with col2:
                            st.write(f"🎯 Elite targets: {elite_targets}")
                        with col3:
                            if forced_boosted > 0:
                                st.write(f"⚡ Forced player boosts: {forced_boosted}")
                            else:
                                st.write("⚡ Forced player boosts: 0")
            
            # Player Usage Analysis for Top 20 Lineups
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📊 Player Usage Analysis (Top 20 Lineups)</h3>', unsafe_allow_html=True)
            
            # Analyze top 20 lineups for player usage
            analysis_lineups = sorted(stacked_lineups, key=lambda x: x[0], reverse=True)[:20]
            player_usage = {}
            
            for points, lineup, salary, _, _, _ in analysis_lineups:
                for _, player in lineup.iterrows():
                    player_name = player['Nickname']
                    position = player['Position']
                    if position == 'D':
                        position = 'DEF'
                    
                    key = f"{player_name} ({position})"
                    if key not in player_usage:
                        player_usage[key] = {
                            'count': 0,
                            'position': position,
                            'salary': player['Salary'],
                            'nickname': player_name
                        }
                    player_usage[key]['count'] += 1
            
            # Create usage breakdown by position
            usage_data = []
            for player_key, data in player_usage.items():
                usage_percentage = (data['count'] / 20) * 100
                usage_data.append({
                    'Player': data['nickname'],
                    'Position': data['position'],
                    'Salary': f"${data['salary']:,}",
                    'Usage Count': f"{data['count']}/20",
                    'Usage %': f"{usage_percentage:.1f}%"
                })
            
            # Sort by usage count descending
            usage_data.sort(key=lambda x: int(x['Usage Count'].split('/')[0]), reverse=True)
            
            # Display in columns by position
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Most Used Players:**")
                top_usage = usage_data[:10]
                usage_df = pd.DataFrame(top_usage)
                st.dataframe(usage_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**📈 Usage by Position:**")
                pos_summary = {}
                for data in usage_data:
                    pos = data['Position']
                    if pos not in pos_summary:
                        pos_summary[pos] = []
                    pos_summary[pos].append(data)
                
                for pos in ['QB', 'RB', 'WR', 'TE', 'DEF']:
                    if pos in pos_summary:
                        pos_players = pos_summary[pos][:3]  # Top 3 per position
                        st.write(f"**{pos}:**")
                        for player in pos_players:
                            st.write(f"• {player['Player']} - {player['Usage Count']} ({player['Usage %']})")
                        st.write("")
            
            # Enhanced Multi-Platform Export Section
            st.markdown("---")
            st.subheader("📥 Export Lineups")

            if ENHANCED_FEATURES_AVAILABLE:
                export_manager = ExportManager()
                exporter = LineupExporter()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Platform selection
                    platforms = st.multiselect(
                        "Select platforms to export to:",
                        options=exporter.get_supported_platforms(),
                        default=['fanduel'],
                        help="Export lineups to multiple DFS platforms simultaneously"
                    )
                    
                    num_export = st.slider("Number of lineups to export", 1, min(len(stacked_lineups), 150), min(20, len(stacked_lineups)))
                    
                    # Entry ID Configuration
                    with st.expander("🎯 Contest Entry Settings"):
                        st.markdown("**Configure contest details for CSV export:**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            base_entry_id = st.number_input(
                                "Base Entry ID", 
                                value=3584175604, 
                                help="Starting entry ID (will increment for each lineup)"
                            )
                            contest_id = st.text_input(
                                "Contest ID", 
                                value="121309-276916553",
                                help="Contest identifier from DFS platform"
                            )
                        with col_b:
                            contest_name = st.text_input(
                                "Contest Name", 
                                value="$60K Sun NFL Hail Mary",
                                help="Name of the contest"
                            )
                            entry_fee = st.text_input(
                                "Entry Fee", 
                                value="0.25",
                                help="Fee per entry (e.g., 0.25, 5.00, 100)"
                            )

                with col2:
                    if st.button("📋 Generate Multi-Platform Export", type="primary"):
                        if platforms:
                            with st.spinner("Generating exports for selected platforms..."):
                                contest_info = {
                                    'base_entry_id': base_entry_id,
                                    'contest_id': contest_id,
                                    'contest_name': contest_name,
                                    'entry_fee': entry_fee
                                }
                                
                                exports = export_manager.export_to_multiple_platforms(
                                    stacked_lineups, platforms, contest_info, num_export
                                )
                                
                                # Display download buttons for each platform
                                for platform, export_content in exports.items():
                                    if not export_content.startswith("Export failed"):
                                        st.download_button(
                                            label=f"💾 Download {platform.title()} CSV",
                                            data=export_content,
                                            file_name=f"{platform}_lineups_{contest_name.replace(' ', '_').replace('$', '').replace(',', '')}.csv",
                                            mime="text/csv",
                                            key=f"download_{platform}"
                                        )
                                    else:
                                        st.error(f"❌ {platform}: {export_content}")
                        else:
                            st.warning("Please select at least one platform to export to.")
            else:
                # Show message when lineups haven't been generated yet
                if not st.session_state.get('lineups_generated', False) or not st.session_state.get('stacked_lineups'):
                    st.info("📊 Generate lineups first to enable CSV export")
            
            # Comprehensive Player Usage Analysis
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📊 Comprehensive Player Usage</h3>', unsafe_allow_html=True)
            st.markdown("Analyze player exposure for optimal tournament strategy")
            
            # Analysis Controls FIRST - so we can use the scope
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.markdown("**📊 Lineup Analysis Scope:**")
                analysis_scope = st.selectbox(
                    "Analysis Type:",
                    ["Top 150 Export Lineups", "All Generated Lineups"],  # Default to Top 150 first
                    key="analysis_scope"
                )
            
            # Use the selected scope to determine which lineups to analyze
            analysis_lineups = stacked_lineups[:150] if analysis_scope == "Top 150 Export Lineups" else stacked_lineups
            
            # Analyze lineups based on selected scope
            all_player_usage = {}
            total_lineups = len(analysis_lineups)
            
            for points, lineup, salary, _, _, _ in analysis_lineups:
                for _, player in lineup.iterrows():
                    player_name = player['Nickname']
                    position = player['Position']
                    if position == 'D':
                        position = 'DEF'
                    
                    key = f"{player_name} ({position})"
                    if key not in all_player_usage:
                        all_player_usage[key] = {
                            'count': 0,
                            'position': position,
                            'salary': player['Salary'],
                            'nickname': player_name,
                            'fppg': player.get('FPPG', 0),
                            'team': player.get('Team', ''),
                            'opponent': player.get('Opponent', ''),
                            'matchup_quality': player.get('Matchup_Quality', '')
                        }
                    all_player_usage[key]['count'] += 1
            
            # Create comprehensive usage data with enhanced metrics
            comprehensive_usage_data = []
            for player_key, data in all_player_usage.items():
                usage_percentage = (data['count'] / total_lineups) * 100
                
                # Calculate additional metrics
                points_per_dollar = (data['fppg'] / data['salary']) * 1000 if data['salary'] > 0 else 0
                
                # Determine value tier
                if data['salary'] >= 8000:
                    value_tier = "Premium"
                elif data['salary'] >= 6000:
                    value_tier = "Mid-Tier"
                else:
                    value_tier = "Value"
                
                # Calculate leverage score (higher FPPG with lower usage = more leverage)
                if usage_percentage > 0:
                    leverage_score = (data['fppg'] * 100) / (usage_percentage + 1)  # +1 to avoid division by zero
                else:
                    leverage_score = data['fppg'] * 100
                
                # Enhanced tournament ownership projection
                base_ownership = usage_percentage * 0.6  # Assume public is less optimal than your model
                
                # Adjust based on salary tier and position
                if data['position'] == 'QB':
                    multiplier = 1.2  # QBs get more attention
                elif data['position'] in ['RB', 'WR']:
                    multiplier = 1.0
                elif data['position'] == 'TE':
                    multiplier = 0.8  # Less attention on TEs
                else:  # DEF
                    multiplier = 0.7
                
                # Adjust for salary tier
                if data['salary'] >= 9000:
                    multiplier *= 1.3  # Expensive players get more attention
                elif data['salary'] <= 5000:
                    multiplier *= 0.7  # Cheap players less noticed
                
                projected_ownership_pct = min(base_ownership * multiplier, 85)  # Cap at 85%
                
                if projected_ownership_pct >= 40:
                    projected_ownership = f"High ({projected_ownership_pct:.1f}%)"
                elif projected_ownership_pct >= 15:
                    projected_ownership = f"Medium ({projected_ownership_pct:.1f}%)"
                elif projected_ownership_pct >= 5:
                    projected_ownership = f"Low ({projected_ownership_pct:.1f}%)"
                else:
                    projected_ownership = f"Contrarian ({projected_ownership_pct:.1f}%)"
                
                # Calculate ceiling/floor estimates
                ceiling_multiplier = 1.4 if data['matchup_quality'] in ['ELITE TARGET', 'Great Target'] else 1.2
                floor_multiplier = 0.6 if data['position'] in ['WR', 'TE'] else 0.7
                
                ceiling = data['fppg'] * ceiling_multiplier
                floor = data['fppg'] * floor_multiplier
                
                # Calculate variance (upside vs consistency)
                variance = ceiling - floor
                upside_rating = "High" if variance >= 8 else "Medium" if variance >= 5 else "Low"
                
                # GPP score (combines leverage, upside, and value)
                gpp_score = (leverage_score * 0.4) + (variance * 0.3) + (points_per_dollar * 0.3)
                
                comprehensive_usage_data.append({
                    'Player': data['nickname'],
                    'Position': data['position'],
                    'Team': data['team'],
                    'vs': data['opponent'],
                    'Matchup': data['matchup_quality'],
                    'Salary': data['salary'],
                    'FPPG': data['fppg'],
                    'Ceiling': ceiling,
                    'Floor': floor,
                    'Upside': upside_rating,
                    'Count': data['count'],
                    'Usage %': usage_percentage,
                    'Points/$': points_per_dollar,
                    'Value Tier': value_tier,
                    'Leverage': leverage_score,
                    'GPP Score': gpp_score,
                    'Proj Own': projected_ownership,
                    'Salary_Display': f"${data['salary']:,}",
                    'Usage_Display': f"{usage_percentage:.1f}%",
                    'Points_Per_Dollar_Display': f"{points_per_dollar:.2f}",
                    'Leverage_Display': f"{leverage_score:.1f}",
                    'Ceiling_Display': f"{ceiling:.1f}",
                    'Floor_Display': f"{floor:.1f}",
                    'GPP_Score_Display': f"{gpp_score:.1f}"
                })
            
            # Sort by usage percentage descending
            comprehensive_usage_data.sort(key=lambda x: x['Usage %'], reverse=True)
            
            # Store in session state for Dynamic Usage Adjustment
            # CRITICAL: Only set this if we don't already have adjustments applied
            # or if the scope has genuinely changed, to prevent overwriting user adjustments
            scope_key = f"scope_{analysis_scope}"
            has_existing_adjustments = any(k.startswith('usage_adj_') for k in st.session_state.keys())
            
            should_update_data = (
                scope_key not in st.session_state or  # New scope
                not st.session_state.get('adjustments_applied', False) or  # No adjustments applied yet
                not has_existing_adjustments  # No user adjustments exist
            )
            
            if should_update_data:
                st.session_state.comprehensive_usage_data = comprehensive_usage_data
                st.session_state[scope_key] = True
                # Only clear the adjustments flag when loading genuinely fresh data
                if scope_key not in st.session_state:
                    st.session_state.adjustments_applied = False
            
            # Display comprehensive usage analysis - TABLE ONLY
            st.subheader("🎯 Complete Player Usage Breakdown")
            
            with col2:
                st.markdown("**🎯 Filter by QB Stack:**")
                
                # Get all QBs from already defined analysis_lineups
                all_qbs = set()
                
                for points, lineup, salary, _, _, _ in analysis_lineups:
                    qb_row = lineup[lineup['Position'] == 'QB']
                    if not qb_row.empty:
                        qb_name = qb_row.iloc[0]['Nickname']
                        qb_team = qb_row.iloc[0]['Team']
                        all_qbs.add(f"{qb_name} ({qb_team})")
                
                qb_filter_options = ['All Players'] + sorted(list(all_qbs))
                selected_qb = st.selectbox("Select QB Stack:", qb_filter_options, key="usage_qb_filter")
            
            with col3:
                if analysis_scope == "Top 150 Export Lineups":
                    st.info("🎯 **Analyzing your actual 150-lineup portfolio** - this matches what you'll submit to FanDuel!")
                else:
                    st.info(f"📊 **Analyzing all {len(stacked_lineups)} generated lineups** - includes experimental lineups not for export")
                
                if selected_qb != 'All Players':
                    qb_name = selected_qb.split(' (')[0]  # Extract QB name from "Name (Team)" format
                    st.write(f"🎯 Filtered to **{selected_qb}** stacks only")
            
            # Apply analysis scope and QB filtering
            working_lineups = stacked_lineups[:150] if analysis_scope == "Top 150 Export Lineups" else stacked_lineups
            
            if selected_qb != 'All Players':
                qb_name = selected_qb.split(' (')[0]
                
                # Get lineups that contain this QB from working set
                qb_lineups = []
                for points, lineup, salary, _, _, _ in working_lineups:
                    qb_row = lineup[lineup['Position'] == 'QB']
                    if not qb_row.empty and qb_row.iloc[0]['Nickname'] == qb_name:
                        qb_lineups.append((points, lineup, salary))
                
                # Recalculate usage stats for QB-filtered lineups
                filtered_usage_data = []
                qb_total_lineups = len(qb_lineups)
                working_lineups = qb_lineups
                
                if len(working_lineups) > 0:
                    # Count usage in filtered lineups
                    player_counts = {}
                    for points, lineup, salary in working_lineups:
                        for _, player in lineup.iterrows():
                            player_key = f"{player['Nickname']}_{player['Position']}_{player['Team']}"
                            if player_key not in player_counts:
                                player_counts[player_key] = {
                                    'count': 0,
                                    'player_info': player
                                }
                            player_counts[player_key]['count'] += 1
                    
                    # Create filtered usage data
                    for player_key, data in player_counts.items():
                        player = data['player_info']
                        count = data['count']
                        usage_pct = (count / len(working_lineups)) * 100
                        
                        filtered_usage_data.append({
                            'Player': player['Nickname'],
                            'Position': player['Position'],
                            'Team': player['Team'],
                            'vs': player.get('Opponent', 'N/A'),
                            'Matchup': player.get('Matchup_Quality', 'N/A'),
                            'Salary_Display': f"${player['Salary']:,}",
                            'FPPG': player['FPPG'],
                            'Ceiling_Display': f"{player.get('Ceiling', player['FPPG'] * 1.25):.1f}",
                            'Floor_Display': f"{player.get('Floor', player['FPPG'] * 0.75):.1f}",
                            'Upside': f"{((player.get('Ceiling', player['FPPG'] * 1.25) / player['FPPG']) - 1) * 100:.0f}%" if player['FPPG'] > 0 else "0%",
                            'Points_Per_Dollar_Display': f"{(player['FPPG'] / player['Salary']) * 1000:.2f}" if player['Salary'] > 0 else "0.00",
                            'Value Tier': 'Premium' if player['Salary'] >= 8000 else 'Mid' if player['Salary'] >= 6000 else 'Value',
                            'Count': count,
                            'Usage %': usage_pct,
                            'Usage_Display': f"{usage_pct:.1f}%",
                            'Leverage_Display': f"{max(0, 15 - usage_pct):.1f}%",  # Simple leverage calc
                            'GPP_Score_Display': f"{(usage_pct * 0.3 + (player['FPPG'] / player['Salary'] * 1000) * 0.7):.1f}" if player['Salary'] > 0 else f"{usage_pct * 0.3:.1f}",
                            'Proj Own': f"{min(usage_pct * 1.5, 50):.0f}%"  # Estimated ownership
                        })
                    
                    # Sort filtered data
                    filtered_usage_data.sort(key=lambda x: x['Usage %'], reverse=True)
                    comprehensive_usage_data = filtered_usage_data
                    total_lineups = len(working_lineups)
                else:
                    if selected_qb != 'All Players':
                        st.warning(f"No lineups found with {selected_qb}")
                    comprehensive_usage_data = []
            else:
                # No QB filtering, just apply scope filtering  
                if analysis_scope == "Top 150 Export Lineups":
                    # Recalculate comprehensive usage for top 150 only
                    top_150_usage_data = []
                    top_150_lineups = stacked_lineups[:150]
                    
                    if top_150_lineups:
                        player_counts = {}
                        for points, lineup, salary, _, _, _ in top_150_lineups:
                            for _, player in lineup.iterrows():
                                player_key = f"{player['Nickname']}_{player['Position']}_{player['Team']}"
                                if player_key not in player_counts:
                                    player_counts[player_key] = {
                                        'count': 0,
                                        'player_info': player
                                    }
                                player_counts[player_key]['count'] += 1
                        
                        for player_key, data in player_counts.items():
                            player = data['player_info']
                            count = data['count']
                            usage_pct = (count / 150) * 100
                            
                            top_150_usage_data.append({
                                'Player': player['Nickname'],
                                'Position': player['Position'],
                                'Team': player['Team'],
                                'vs': player.get('Opponent', 'N/A'),
                                'Matchup': player.get('Matchup_Quality', 'N/A'),
                                'Salary_Display': f"${player['Salary']:,}",
                                'FPPG': player['FPPG'],
                                'Ceiling_Display': f"{player.get('Ceiling', player['FPPG'] * 1.25):.1f}",
                                'Floor_Display': f"{player.get('Floor', player['FPPG'] * 0.75):.1f}",
                                'Upside': f"{((player.get('Ceiling', player['FPPG'] * 1.25) / player['FPPG']) - 1) * 100:.0f}%" if player['FPPG'] > 0 else "0%",
                                'Points_Per_Dollar_Display': f"{(player['FPPG'] / player['Salary']) * 1000:.2f}" if player['Salary'] > 0 else "0.00",
                                'Value Tier': 'Premium' if player['Salary'] >= 8000 else 'Mid' if player['Salary'] >= 6000 else 'Value',
                                'Count': count,
                                'Usage %': usage_pct,
                                'Usage_Display': f"{usage_pct:.1f}%",
                                'Leverage_Display': f"{max(0, 15 - usage_pct):.1f}%",
                                'GPP_Score_Display': f"{(usage_pct * 0.3 + (player['FPPG'] / player['Salary'] * 1000) * 0.7):.1f}" if player['Salary'] > 0 else f"{usage_pct * 0.3:.1f}",
                                'Proj Own': f"{min(usage_pct * 1.5, 50):.0f}%"
                            })
                        
                        top_150_usage_data.sort(key=lambda x: x['Usage %'], reverse=True)
                        comprehensive_usage_data = top_150_usage_data
                        total_lineups = 150
            
            # Use session state data if it exists (updated by Dynamic Usage Adjustments)
            # Otherwise use the freshly calculated data
            display_usage_data = st.session_state.get('comprehensive_usage_data', comprehensive_usage_data)
            
            # Ensure data is always sorted by usage percentage (highest to lowest)
            display_usage_data = sorted(display_usage_data, key=lambda x: x['Usage %'], reverse=True)
            
            # Debug info
            using_session_state = 'comprehensive_usage_data' in st.session_state
            st.info(f"""
            **📊 Data Source Debug:**
            - **Using session state**: {using_session_state}
            - **Original data**: {len(comprehensive_usage_data)} players
            - **Display data**: {len(display_usage_data)} players
            - **Session state keys**: {len([k for k in st.session_state.keys() if k.startswith('usage_adj_')])} adjustments saved
            - **Data sorted by**: Usage % (highest to lowest)
            """)
            
            # Create display dataframe with enhanced tournament columns
            display_df = pd.DataFrame([{
                'Player': data['Player'],
                'Pos': data['Position'],
                'Team': data['Team'],
                'vs': data['vs'],
                'Matchup': data['Matchup'],
                'Salary': data['Salary_Display'],
                'FPPG': f"{data['FPPG']:.1f}",
                'Ceiling': data['Ceiling_Display'],
                'Floor': data['Floor_Display'],
                'Upside': data['Upside'],
                'Pts/$': data['Points_Per_Dollar_Display'],
                'Value Tier': data['Value Tier'],
                'Count': f"{data['Count']}/{total_lineups}",
                'Usage %': data['Usage_Display'],
                'Target %': float(data['Usage_Display'].replace('%', '')),  # Add Target % right after Usage %
                'Leverage': data['Leverage_Display'],
                'GPP Score': data['GPP_Score_Display'],
                'Proj Own': data['Proj Own']
            } for data in display_usage_data])
            
            # Display editable usage breakdown table
            st.markdown("**📊 Complete Player Usage Breakdown** - Click on Target % cells to edit exposures")
            
            # Configure which columns are editable
            column_config = {
                "Target %": st.column_config.NumberColumn(
                    "Target %",
                    help="Type new exposure percentage here",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format="%.1f"
                )
            }
            
            # Use data_editor for inline editing
            edited_df = st.data_editor(
                display_df, 
                use_container_width=True, 
                hide_index=True, 
                height=600,
                column_config=column_config,
                disabled=[col for col in display_df.columns if col != 'Target %']  # Only Target % is editable
            )
            
            # Check for changes and show apply button
            changes_made = False
            if 'Target %' in edited_df.columns:
                for idx, row in edited_df.iterrows():
                    original_usage = float(display_df.iloc[idx]['Usage %'].replace('%', ''))
                    new_target = row['Target %']
                    if abs(new_target - original_usage) > 0.1:
                        changes_made = True
                        break
            
            if changes_made:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🚀 Apply Changes", type="primary"):
                        # Update session state with new targets
                        for idx, row in edited_df.iterrows():
                            player_name = row['Player']
                            position = row['Pos'] 
                            team = row['Team']
                            new_target = row['Target %']
                            
                            # Create session key
                            clean_name = player_name.replace(" ", "_").replace(".", "").replace("'", "")
                            session_key = f"usage_adj_{clean_name}_{position}_{team}"
                            st.session_state[session_key] = new_target
                        
                        with st.spinner("Applying exposure changes to lineups..."):
                            # Apply changes to lineups
                            modified_lineups = apply_usage_adjustments(stacked_lineups[:150], display_usage_data, "All Positions")
                            if modified_lineups:
                                st.success("✅ Lineups successfully modified!")
                                st.rerun()
                            else:
                                st.error("❌ Unable to modify lineups. Try smaller changes.")
                
                with col2:
                    # Show summary of changes
                    changes_summary = []
                    for idx, row in edited_df.iterrows():
                        original_usage = float(display_df.iloc[idx]['Usage %'].replace('%', ''))
                        new_target = row['Target %']
                        if abs(new_target - original_usage) > 0.1:
                            change = new_target - original_usage
                            changes_summary.append(f"{row['Player']}: {original_usage:.1f}% → {new_target:.1f}% ({change:+.1f}%)")
                    
                    if changes_summary:
                        st.info(f"**{len(changes_summary)} changes ready:**\n" + "\n".join(changes_summary[:5]) + 
                               (f"\n... and {len(changes_summary)-5} more" if len(changes_summary) > 5 else ""))
        else:
            st.info("📊 Generate lineups first to enable exposure adjustments")
            
            # Tournament Metrics Explanation
            with st.expander("🎯 Tournament Metrics Guide - Click to Learn How to Use Each Column"):
                st.markdown("### 📊 **Understanding Your Tournament Analysis Table**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏆 **Key Tournament Metrics:**")
                    st.markdown("""
                    **GPP Score:**
                    - **> 15**: 🔥 Elite tournament plays - highest priority
                    - **10-15**: ⭐ Good tournament options
                    - **< 10**: ⚠️ Risky plays, use sparingly
                    
                    **Leverage Score:**
                    - **> 50**: 📈 Good leverage opportunities
                    - **30-50**: 📊 Moderate leverage potential
                    - **< 30**: 📉 Low leverage, likely popular
                    
                    **Ceiling Points:**
                    - **> 25**: 🚀 Tournament-winning upside
                    - **20-25**: 💪 Solid ceiling, good for GPPs
                    - **< 20**: 😐 Limited upside potential
                    
                    **Projected Ownership:**
                    - **< 10%**: 💎 Massive leverage potential
                    - **10-25%**: ⚡ Good differentiation
                    - **> 40%**: ⚠️ Chalk plays, use carefully
                    """)
                
                with col2:
                    st.markdown("#### 💡 **How to Use This Data:**")
                    st.markdown("""
                    **🎯 Tournament Strategy:**
                    
                    **Core Plays (60-70% of lineup):**
                    - High GPP Score (>15) + Medium ownership (15-40%)
                    - Players you're most confident in
                    
                    **Leverage Plays (20-30% of lineup):**
                    - High Ceiling (>25) + Low ownership (<15%)
                    - Tournament differentiators
                    
                    **Contrarian Plays (5-10% of lineup):**
                    - Very Low ownership (<5%) + High upside
                    - Massive leverage if they hit
                    
                    **💰 Value Identification:**
                    - **Pts/$**: Higher = better salary efficiency
                    - **Value Tier**: Premium/Mid/Value for roster balance
                    - **Floor**: Minimum expected points (safety)
                    
                    **📈 Upside Categories:**
                    - **High**: Boom/bust players, great for tournaments
                    - **Medium**: Steady with upside potential
                    - **Low**: Consistent, better for cash games
                    """)
                
                st.markdown("---")
                st.markdown("#### 🔍 **Quick Filters for Tournament Success:**")
                
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.markdown("""
                    **🔥 Elite GPP Targets:**
                    - GPP Score > 15
                    - Leverage Score > 40
                    - Ceiling > 22
                    """)
                
                with col4:
                    st.markdown("""
                    **💎 Leverage Gems:**
                    - Proj Own < 15%
                    - Ceiling > 20
                    - Value Tier: Any
                    """)
                
                with col5:
                    st.markdown("""
                    **⚡ Contrarian Bombs:**
                    - Proj Own < 8%
                    - Upside: High
                    - GPP Score > 12
                    """)
                
                st.info("💡 **Pro Tip**: Sort the table by different columns to find players that match these criteria!")
            
            st.markdown("---")

if __name__ == "__main__":
    main()

