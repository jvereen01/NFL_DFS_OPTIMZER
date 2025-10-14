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

# @st.cache_data(ttl=60)  # Cache disabled to force fresh data loading
def load_player_data(_cache_key=None):
    """Load and process player data with enhanced validation and optimization"""
    if ENHANCED_FEATURES_AVAILABLE:
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
    
    # ONLY use the October 19th CSV file - Updated to latest
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
        st.error(f"❌ Required CSV file not found: {target_file}")
        st.warning("This app requires the October 12th FanDuel player list file.")
        st.info("Please upload the correct CSV file to continue.")
        return None
    
    try:
        # Load player CSV
        st.info(f"🔍 Loading CSV file: {os.path.basename(csv_path)}")  # Debug info
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        
        # Apply filters - Include Q (Questionable) players, exclude only serious injuries
        injury_exclusions = ['IR', 'O', 'D']  # Removed 'Q' to include Questionable players
        df = df[~df['Injury Indicator'].isin(injury_exclusions)]
        
        # Salary filters
        defense_mask = (df['Position'] == 'D') & (df['Salary'] >= 3000) & (df['Salary'] <= 5000)
        other_positions_mask = (df['Position'] != 'D') & (df['Salary'] >= 5000)
        df = df[defense_mask | other_positions_mask]
        
        st.success(f"✅ Loaded {len(df)} players from {os.path.basename(csv_path)}")  # Debug info
        return df
    except FileNotFoundError:
        st.error(f"File was found but couldn't be read: {csv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading player data from {csv_path}: {str(e)}")
        return None

# @st.cache_data  # Cache disabled to force fresh data loading
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
        st.info("NFL.xlsx file not found. Using salary-based analysis instead.")
        return None, None
    except PermissionError:
        st.info("NFL.xlsx file is currently open in another program. Close Excel to enable advanced features, or continue with basic analysis.")
        return None, None
    except Exception as e:
        st.info(f"Could not load NFL.xlsx data: {str(e)}. Using basic analysis instead.")
        return None, None

# @st.cache_data  # Cache disabled to force fresh data loading
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
        st.info("NFL.xlsx file not found. Performance analysis will use basic projections.")
        return None
    except PermissionError:
        st.info("NFL.xlsx file is currently open in another program. Close Excel to enable performance analysis, or continue with basic projections.")
        return None
    except Exception as e:
        st.info(f"Could not load fantasy data: {str(e)}. Using basic projections instead.")
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
                    pos_players = pos_players.loc[pos_players.groupby('Team')['Salary'].idxmax()]
                
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

def generate_lineups(df, weighted_pools, num_simulations, stack_probability, elite_target_boost, great_target_boost, fantasy_data=None, player_selections=None, force_mode="Soft Force (Boost Only)", forced_player_boost=0.0):
    """Generate optimized lineups with optional player selection constraints"""
    stacked_lineups = []
    salary_cap = 60000
    
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
                
                # Determine if this will be a stacked lineup early
                will_attempt_stack = random.random() < stack_probability
                
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
                        additional_rbs = available_rbs.sample(remaining_rb_spots, weights=available_rbs['Selection_Weight'])
                        rb = pd.concat([selected_rbs, additional_rbs])
                    else:
                        continue  # Skip this lineup if can't find RBs from different teams
                else:
                    rb = selected_rbs
                
                # Additional check: Ensure no duplicate RB teams in final selection
                if len(rb) == 2:
                    rb_teams = rb['Team'].tolist()
                    if rb_teams[0] == rb_teams[1]:
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
                
                # Fill remaining WR/TE spots with stacking logic
                remaining_wr_spots = 3 - len(selected_wrs)
                need_te = len(selected_te) == 0
                
                if remaining_wr_spots > 0 or need_te:
                    # Use the pre-determined stacking decision
                    attempt_stack = will_attempt_stack and remaining_wr_spots > 0
                    
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
                    continue
                
                flex = flex_pool.sample(1)
                lineup_players.append(flex)
                
                # Build final lineup
                lineup = pd.concat(lineup_players).reset_index(drop=True)
                
                # Validate lineup with early salary check
                total_salary = lineup['Salary'].sum()
                if total_salary > salary_cap:
                    continue
                
                # Check 4-player per team limit (FanDuel rule)
                team_counts = lineup['Team'].value_counts()
                if team_counts.max() > 4:
                    continue  # Skip lineups with more than 4 players from same team
                
                # Validate lineup
                if (len(lineup) == 9 and 
                    len(lineup['Nickname'].unique()) == 9):
                    
                    total_points = lineup['Adjusted_FPPG'].sum()
                    
                    # Check stacking
                    qb_team = lineup[lineup['Position'] == 'QB']['Team'].iloc[0]
                    wr_teams = lineup[lineup['Position'] == 'WR']['Team'].tolist()
                    te_teams = lineup[lineup['Position'] == 'TE']['Team'].tolist()
                    
                    stacked_wrs_count = sum(1 for team in wr_teams if team == qb_team)
                    stacked_tes_count = sum(1 for team in te_teams if team == qb_team)
                    qb_wr_te_count = stacked_wrs_count + stacked_tes_count
                    
                    stacked_lineups.append((total_points, lineup, total_salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count))
                    successful_lineups += 1
                    break
                    
            except Exception as e:
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
    
    # Show final stats
    if player_selections and successful_lineups < num_simulations * 0.5:
        st.warning(f"⚠️ Low success rate ({(successful_lineups/num_simulations*100):.1f}%). Consider reducing forced players or adjusting salary constraints.")
    
    return stacked_lineups

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
        
        # Quick Strategy Presets
        st.subheader("🎯 Quick Strategy Presets")
        
        # Strategy selection with radio buttons (like force mode)
        strategy_preset = st.radio("Strategy Type", 
                                 ["Custom Settings", "💰 Single Entry", "🏆 Tournament"], 
                                 index=0,
                                 help="Single Entry: Cash games, head-to-head, 50/50s. Tournament: GPPs, large tournaments")
        
        with st.expander("💡 Strategy Types Explained"):
            st.markdown("""
            **💰 Single Entry (Cash Games):**
            - Conservative stacking (65%)
            - Consistent elite performers (35% boost)
            - 8,000 simulations for stability
            - Best for: Head-to-head, 50/50s, cash games
            
            **🏆 Tournament (GPPs):**
            - Aggressive stacking (85%)
            - High ceiling players (55% boost)
            - 12,000 simulations for diversity
            - Best for: Large tournaments, contrarian plays
            
            **⚙️ Custom Settings:**
            - Use your manual slider configurations
            - Full control over all parameters
            """)
        
        # Apply presets based on selection
        if strategy_preset == "💰 Single Entry":
            current_simulations = 8000
            current_stack_prob = 0.65
            current_elite_boost = 0.35
            current_great_boost = 0.20
            st.success("💰 Single Entry preset active!")
        elif strategy_preset == "🏆 Tournament":
            current_simulations = 12000
            current_stack_prob = 0.85
            current_elite_boost = 0.55
            current_great_boost = 0.30
            st.success("🏆 Tournament preset active!")
        else:  # Custom Settings
            current_simulations = default_simulations
            current_stack_prob = default_stack_prob
            current_elite_boost = default_elite_boost
            current_great_boost = default_great_boost
        
        # Configuration sliders (will use preset values if buttons were clicked)
        num_simulations = st.slider("Number of Simulations", 1000, 20000, current_simulations, step=1000,
                                    help="More simulations = more unique lineups but slower generation. 5000 simulations typically generates 3000-4000 unique lineups.")
        stack_probability = st.slider("Stacking Probability", 0.0, 1.0, current_stack_prob, step=0.05)
        elite_target_boost = st.slider("Elite Target Boost", 0.0, 1.0, current_elite_boost, step=0.05)
        great_target_boost = st.slider("Great Target Boost", 0.0, 1.0, current_great_boost, step=0.05)
        
        st.subheader("🚀 Performance Boost Multipliers")
        wr_boost_multiplier = st.slider("WR Performance Boost Multiplier", 0.5, 2.0, 1.0, step=0.1)
        rb_boost_multiplier = st.slider("RB Performance Boost Multiplier", 0.5, 2.0, 1.0, step=0.1)
        
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
            
            **💡 Pro Tip:** Use Soft Force with 10-20% boost for optimal tournament lineup variety!
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
        df = load_player_data("FanDuel-NFL-2025 EDT-10 EDT-19 EDT-121559-players-list.csv")
        
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
                df['PosRank'] = df['Nickname'].map(posrank_mapping)
                # Fill missing PosRank with 999 for players not in fantasy data
                df['PosRank'] = df['PosRank'].fillna(999).astype(int)
        else:
            # If no fantasy data, use default ranking
            df['PosRank'] = 999
        
        # Apply analysis
        with st.spinner("Applying matchup analysis..."):
            df = apply_matchup_analysis(df, pass_defense, rush_defense)
            
        with st.spinner("Creating performance boosts..."):
            wr_performance_boosts, rb_performance_boosts, te_performance_boosts, qb_performance_boosts = create_performance_boosts(fantasy_data, wr_boost_multiplier, rb_boost_multiplier)
        
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
        
        # Manual Projection Override Section (Always Available)
        st.markdown("---")
        st.markdown("### 📝 Manual Projection Overrides")
        st.markdown("*Adjust player projections for injuries, weather, or personal insights*")
        
        # Create editable dataframe for projection overrides
        if 'edited_projections' not in st.session_state:
            st.session_state.edited_projections = {}
        
        # Create a subset of data for editing - show key players
        # Use the actual column names from the CSV
        edit_cols = ['Nickname', 'Position', 'Team', 'Salary', 'FPPG', 'Adjusted_FPPG']
        
        # Show top players by position for easy editing
        edit_df = df[edit_cols].copy()
        edit_df = edit_df.sort_values(['Position', 'Salary'], ascending=[True, False])
        edit_df = edit_df.groupby('Position').head(10).reset_index(drop=True)
        
        # Add editable column for new projections
        edit_df['New_FPPG'] = edit_df['FPPG']
        
        # Create data editor
        edited_data = st.data_editor(
            edit_df,
            column_config={
                "Nickname": st.column_config.TextColumn("Player", disabled=True),
                "Position": st.column_config.TextColumn("Pos", disabled=True, width="small"),
                "Team": st.column_config.TextColumn("Team", disabled=True, width="small"),
                "Salary": st.column_config.NumberColumn("Salary", disabled=True, format="$%d"),
                "FPPG": st.column_config.NumberColumn("Original", disabled=True, format="%.1f"),
                "Adjusted_FPPG": st.column_config.NumberColumn("Current", disabled=True, format="%.1f"),
                "New_FPPG": st.column_config.NumberColumn("📝 Edit FPPG", format="%.1f", min_value=0.0, max_value=50.0)
            },
            key="projection_editor",
            hide_index=True,
            width='stretch'
        )
        
        # Apply manual overrides to main dataframe
        if edited_data is not None:
            for idx, row in edited_data.iterrows():
                player_name = row['Nickname']
                new_projection = row['New_FPPG']
                original_projection = row['FPPG']
                
                # Only update if there's a significant change (avoid floating point precision issues)
                if abs(new_projection - original_projection) > 0.1:
                    # Update the main dataframe
                    player_mask = df['Nickname'] == player_name
                    if player_mask.any():
                        df.loc[player_mask, 'Adjusted_FPPG'] = new_projection
                        st.session_state.edited_projections[player_name] = new_projection
            
            # Show summary of overrides
            if st.session_state.edited_projections:
                st.info(f"🎯 {len(st.session_state.edited_projections)} player projections manually adjusted")
                if st.button("🔄 Reset All Manual Overrides"):
                    st.session_state.edited_projections = {}
                    st.rerun()
        
        # Player Selection Interface
        if enable_player_selection:
            
            # Manual Projection Override Section
            st.markdown("---")
            st.markdown("### 📝 Manual Projection Overrides")
            st.markdown("*Adjust player projections for injuries, weather, or personal insights*")
            
            # Create editable dataframe for projection overrides
            if 'edited_projections' not in st.session_state:
                st.session_state.edited_projections = {}
            
            # Create a subset of data for editing - show key players
            edit_cols = ['Player', 'Position', 'Team', 'Salary', 'FPPG', 'Adjusted_FPPG']
            if all(col in df.columns for col in edit_cols):
                # Show top players by position for easy editing
                edit_df = df[edit_cols].copy()
                edit_df = edit_df.sort_values(['Position', 'Salary'], ascending=[True, False])
                edit_df = edit_df.groupby('Position').head(10).reset_index(drop=True)
                
                # Add editable column for new projections
                edit_df['New_FPPG'] = edit_df['FPPG']
                
                # Create data editor
                edited_data = st.data_editor(
                    edit_df,
                    column_config={
                        "Player": st.column_config.TextColumn("Player", disabled=True),
                        "Position": st.column_config.TextColumn("Pos", disabled=True, width="small"),
                        "Team": st.column_config.TextColumn("Team", disabled=True, width="small"),
                        "Salary": st.column_config.NumberColumn("Salary", disabled=True, format="$%d"),
                        "FPPG": st.column_config.NumberColumn("Original", disabled=True, format="%.1f"),
                        "Adjusted_FPPG": st.column_config.NumberColumn("Current", disabled=True, format="%.1f"),
                        "New_FPPG": st.column_config.NumberColumn("📝 Edit FPPG", format="%.1f", min_value=0.0, max_value=50.0)
                    },
                    key="projection_editor",
                    hide_index=True,
                    width='stretch'
                )
                
                # Apply manual overrides to main dataframe
                if edited_data is not None:
                    for idx, row in edited_data.iterrows():
                        player_name = row['Player']
                        new_projection = row['New_FPPG']
                        original_projection = row['FPPG']
                        
                        # Only update if there's a significant change (avoid floating point precision issues)
                        if abs(new_projection - original_projection) > 0.1:
                            # Update the main dataframe
                            player_mask = df['Player'] == player_name
                            if player_mask.any():
                                df.loc[player_mask, 'Adjusted_FPPG'] = new_projection
                                st.session_state.edited_projections[player_name] = new_projection
                
                # Show summary of overrides
                if st.session_state.edited_projections:
                    st.info(f"🎯 {len(st.session_state.edited_projections)} player projections manually adjusted")
                    if st.button("🔄 Reset All Manual Overrides"):
                        st.session_state.edited_projections = {}
                        st.rerun()
            
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
                stacked_lineups = generate_lineups(df, weighted_pools, num_simulations, stack_probability, elite_target_boost, great_target_boost, fantasy_data, player_selections, force_mode, forced_player_boost)
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
            
            # Display lineups
            st.subheader("📋 Generated Lineups")
            for i, (points, lineup, salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count) in enumerate(top_lineups, 1):
                with st.expander(f"Lineup #{i}: {points:.2f} points | ${salary:,} | {'QB+' + str(qb_wr_te_count) + ' receivers' if qb_wr_te_count > 0 else 'No stack'}"):
                    
                    # Create lineup display
                    lineup_display = lineup[['Nickname', 'Position', 'Team', 'Salary', 'FPPG', 'Matchup_Quality', 'PosRank']].copy()
                    lineup_display['Salary'] = lineup_display['Salary'].apply(lambda x: f"${x:,}")
                    lineup_display['FPPG'] = lineup_display['FPPG'].apply(lambda x: f"{x:.1f}")
                    
                    # Set PosRank as the index for display
                    lineup_display.set_index('PosRank', inplace=True)
                    lineup_display = lineup_display.drop('PosRank', axis=1, errors='ignore')  # Remove if accidentally included twice
                    
                    st.dataframe(lineup_display, use_container_width=True)
                    
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
            
            # Comprehensive Player Usage Analysis - ALL Lineups
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📊 Comprehensive Player Usage - ALL Lineups</h3>', unsafe_allow_html=True)
            st.markdown("*Analyze player exposure across all generated lineups for optimal tournament strategy*")
            
            # Analyze ALL lineups for comprehensive usage
            all_player_usage = {}
            total_lineups = len(stacked_lineups)
            
            for points, lineup, salary, _, _, _ in stacked_lineups:
                for _, player in lineup.iterrows():
                    player_name = player['Nickname']
                    position = player['Position']
                    if position == 'D':
                        position = 'DEF'
                    
                    key = f"{player_name}"
                    if key not in all_player_usage:
                        all_player_usage[key] = {
                            'count': 0,
                            'position': position,
                            'salary': player['Salary'],
                            'team': player['Team'],
                            'fppg': player.get('Adjusted_FPPG', player.get('FPPG', 0))
                        }
                    all_player_usage[key]['count'] += 1
            
            # Create comprehensive usage data
            all_usage_data = []
            for player_name, data in all_player_usage.items():
                usage_percentage = (data['count'] / total_lineups) * 100
                all_usage_data.append({
                    'Player': player_name,
                    'Position': data['position'],
                    'Team': data['team'],
                    'Salary': data['salary'],
                    'FPPG': data['fppg'],
                    'Usage Count': data['count'],
                    'Total Lineups': total_lineups,
                    'Usage %': usage_percentage
                })
            
            # Sort by usage percentage descending
            all_usage_data.sort(key=lambda x: x['Usage %'], reverse=True)
            
            # Convert to DataFrame for display
            usage_display_df = pd.DataFrame([{
                'Player': d['Player'],
                'Pos': d['Position'],
                'Team': d['Team'],
                'Salary': f"${d['Salary']:,}",
                'FPPG': f"{d['FPPG']:.1f}",
                'Count': d['Usage Count'],
                'Total': d['Total Lineups'],
                'Usage %': f"{d['Usage %']:.1f}%"
            } for d in all_usage_data])
            
            # Add filtering options
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                position_filter = st.selectbox(
                    "Filter by Position:",
                    options=['All'] + sorted(usage_display_df['Pos'].unique().tolist()),
                    key="usage_pos_filter"
                )
            
            with col_filter2:
                min_usage = st.slider(
                    "Minimum Usage %:",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    key="usage_min_filter"
                )
            
            with col_filter3:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=['Usage %', 'Count', 'Salary', 'FPPG'],
                    key="usage_sort"
                )
            
            # Apply filters
            filtered_df = usage_display_df.copy()
            
            if position_filter != 'All':
                filtered_df = filtered_df[filtered_df['Pos'] == position_filter]
            
            if min_usage > 0:
                filtered_df = filtered_df[filtered_df['Usage %'].str.rstrip('%').astype(float) >= min_usage]
            
            # Sort
            if sort_by == 'Usage %':
                filtered_df = filtered_df.sort_values('Usage %', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
            elif sort_by == 'Count':
                filtered_df = filtered_df.sort_values('Count', ascending=False)
            elif sort_by == 'Salary':
                filtered_df = filtered_df.sort_values('Salary', key=lambda x: x.str.replace('$', '').str.replace(',', '').astype(int), ascending=False)
            elif sort_by == 'FPPG':
                filtered_df = filtered_df.sort_values('FPPG', key=lambda x: x.astype(float), ascending=False)
            
            # Display the table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                avg_usage = filtered_df['Usage %'].str.rstrip('%').astype(float).mean()
                st.metric("Average Usage", f"{avg_usage:.1f}%")
            
            with col_stat2:
                max_usage = filtered_df['Usage %'].str.rstrip('%').astype(float).max()
                st.metric("Max Usage", f"{max_usage:.1f}%")
            
            with col_stat3:
                total_unique_players = len(filtered_df)
                st.metric("Unique Players", total_unique_players)
            
            with col_stat4:
                high_usage_players = len(filtered_df[filtered_df['Usage %'].str.rstrip('%').astype(float) >= 50])
                st.metric("High Usage (≥50%)", high_usage_players)
            
            # Enhanced Multi-Platform Export Section
            st.markdown("---")
            
            if ENHANCED_FEATURES_AVAILABLE:
                st.subheader("📥 Multi-Platform Export")
                
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
                                            file_name=f"{platform}_lineups_{num_export}lineups.csv",
                                            mime="text/csv"
                                        )
                                        st.success(f"✅ {platform.title()} export ready!")
                                    else:
                                        st.error(f"❌ {platform.title()}: {export_content}")
                        else:
                            st.warning("Please select at least one platform to export to")
            else:
                # Fallback to original export (FanDuel only)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("📥 Export Lineups")
                    num_export = st.slider("Number of lineups to export", 1, min(len(stacked_lineups), 150), min(20, len(stacked_lineups)))
                    st.caption(f"Export top {num_export} lineups for FanDuel upload")
                
                with col2:
                    if st.button("📋 Prepare CSV Download", type="primary"):
                        # Original CSV export code (shortened for space)
                        export_lineups = sorted(stacked_lineups, key=lambda x: x[0], reverse=True)[:num_export]
                        csv_data = []
                        
                        for i, (points, lineup, salary, _, _, _) in enumerate(export_lineups, 1):
                            # Create FanDuel format without contest entry columns
                            positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DEF': []}
                            
                            for _, player in lineup.iterrows():
                                pos = player['Position']
                                player_id = player['Id']
                                
                                if pos == 'D':  # Handle defense position mapping
                                    positions['DEF'].append(player_id)
                                elif pos in positions:
                                    positions[pos].append(player_id)
                            
                            # Validate that we have all required positions filled
                            if (len(positions['QB']) < 1 or len(positions['RB']) < 2 or 
                                len(positions['WR']) < 3 or len(positions['TE']) < 1 or 
                                len(positions['DEF']) < 1):
                                continue  # Skip incomplete lineups
                            
                            # Fill FanDuel roster format: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF
                            row = [
                                positions['QB'][0],    # QB
                                positions['RB'][0],    # RB
                                positions['RB'][1],    # RB
                                positions['WR'][0],    # WR
                                positions['WR'][1],    # WR
                                positions['WR'][2],    # WR
                                positions['TE'][0],    # TE
                                '',                    # FLEX - will be determined below
                                positions['DEF'][0]    # DEF
                            ]
                            
                            # Determine FLEX (extra RB, WR, or TE)
                            if len(positions['RB']) > 2:
                                row[7] = positions['RB'][2]
                            elif len(positions['WR']) > 3:
                                row[7] = positions['WR'][3]
                            elif len(positions['TE']) > 1:
                                row[7] = positions['TE'][1]
                            else:
                                continue  # Skip lineups without valid FLEX player
                            
                            # Add completed lineup
                            csv_data.append(row)
                        
                        # Create CSV string with contest entry columns
                        csv_lines = ['entry_id,contest_id,contest_name,entry_fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF']
                        
                        # Use user-configured entry settings
                        for i, row in enumerate(csv_data):
                            entry_id = fallback_base_entry_id + i
                            lineup_data = ','.join(map(str, row))
                            csv_line = f"{entry_id},{fallback_contest_id},{fallback_contest_name},{fallback_entry_fee},{lineup_data}"
                            csv_lines.append(csv_line)
                        
                        csv_string = '\n'.join(csv_lines)
                        
                        # Show a preview of the first few lines to verify format
                        st.write("**CSV Format Preview:**")
                        st.code('\n'.join(csv_lines[:3]))
                        
                        if len(csv_data) == 0:
                            st.error(f"❌ No valid lineups found!")
                        else:
                            st.success(f"✅ {len(csv_data)} lineups prepared for download!")
                            st.download_button(
                                label="💾 Download CSV for FanDuel",
                                data=csv_string,
                                file_name=f"fanduel_lineups_{len(csv_data)}lineups.csv",
                                mime="text/csv",
                                type="secondary"
                            )
            
            st.markdown("---")

if __name__ == "__main__":
    main()
