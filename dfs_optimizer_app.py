import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

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
    """Load and process player data"""
    try:
        # Load player CSV
        df = pd.read_csv('FanDuel-NFL-2025 EDT-10 EDT-05 EDT-121036-players-list (1).csv')
        df.columns = [col.strip() for col in df.columns]
        
        # Apply filters
        injury_exclusions = ['Q', 'IR', 'O', 'D']
        df = df[~df['Injury Indicator'].isin(injury_exclusions)]
        
        # Salary filters
        defense_mask = (df['Position'] == 'D') & (df['Salary'] >= 3000) & (df['Salary'] <= 5000)
        other_positions_mask = (df['Position'] != 'D') & (df['Salary'] >= 5000)
        df = df[defense_mask | other_positions_mask]
        
        return df
    except FileNotFoundError:
        st.error("Player CSV file not found. Please upload the FanDuel player list.")
        return None

@st.cache_data
def load_defensive_data():
    """Load and process defensive matchup data"""
    try:
        # Load Excel data
        excel_file = pd.ExcelFile("NFL.xlsx")
        
        # Load defensive data
        passing_team_names = pd.read_excel("NFL.xlsx", sheet_name="Defense Data 2025", 
                                          usecols=[1], skiprows=41, nrows=32, header=None)
        passing_ypg_data = pd.read_excel("NFL.xlsx", sheet_name="Defense Data 2025", 
                                        usecols=[15], skiprows=41, nrows=32, header=None)
        
        rushing_team_names = pd.read_excel("NFL.xlsx", sheet_name="Defense Data 2025", 
                                          usecols=[1], skiprows=80, nrows=32, header=None)
        rushing_ypg_data = pd.read_excel("NFL.xlsx", sheet_name="Defense Data 2025", 
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

@st.cache_data
def load_fantasy_data():
    """Load fantasy performance data"""
    try:
        fantasy_data = pd.read_excel("NFL.xlsx", sheet_name="Fantasy", header=1)
        fantasy_data['Tgt'] = pd.to_numeric(fantasy_data['Tgt'], errors='coerce')
        fantasy_data['Rec'] = pd.to_numeric(fantasy_data['Rec'], errors='coerce')
        fantasy_data['FDPt'] = pd.to_numeric(fantasy_data['FDPt'], errors='coerce')
        fantasy_data['Att_1'] = pd.to_numeric(fantasy_data['Att_1'], errors='coerce')
        
        fantasy_clean = fantasy_data.dropna(subset=['FDPt']).copy()
        return fantasy_clean
    except FileNotFoundError:
        st.error("Fantasy performance data not found. Performance boosts will be disabled.")
        return None

def apply_matchup_analysis(df, pass_defense, rush_defense):
    """Apply defensive matchup analysis"""
    if pass_defense is None or rush_defense is None:
        df['Adjusted_FPPG'] = df['FPPG']
        df['Matchup_Quality'] = 'Unknown'
        return df
    
    # Create team mapping (simplified)
    teams_sheet = pd.read_excel("NFL.xlsx", sheet_name="Teams")
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
    df['Matchup_Quality'] = 'Unknown'
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
    
    df['Adjusted_FPPG'] = df['FPPG'] * df['Overall_Matchup_Multiplier']
    return df

def create_performance_boosts(fantasy_data):
    """Create fantasy performance boosts"""
    wr_performance_boosts = {}
    rb_performance_boosts = {}
    
    if fantasy_data is not None:
        # WR boosts
        wr_fantasy = fantasy_data[fantasy_data['FantPos'] == 'WR'].copy()
        if len(wr_fantasy) > 0:
            wr_fantasy['Tgt_Percentile'] = wr_fantasy['Tgt'].rank(pct=True, na_option='bottom')
            wr_fantasy['Rec_Percentile'] = wr_fantasy['Rec'].rank(pct=True, na_option='bottom')
            wr_fantasy['FDPt_Percentile'] = wr_fantasy['FDPt'].rank(pct=True, na_option='bottom')
            
            wr_fantasy['WR_Performance_Score'] = (
                wr_fantasy['Tgt_Percentile'] * 0.33 +
                wr_fantasy['Rec_Percentile'] * 0.33 +
                wr_fantasy['FDPt_Percentile'] * 0.34
            )
            wr_fantasy['WR_Performance_Boost'] = wr_fantasy['WR_Performance_Score'] * 0.4
            
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
            rb_fantasy['RB_Performance_Boost'] = rb_fantasy['RB_Performance_Score'] * 0.4
            
            for _, rb in rb_fantasy.iterrows():
                rb_performance_boosts[rb['Player']] = rb['RB_Performance_Boost']
    
    return wr_performance_boosts, rb_performance_boosts

def create_weighted_pools(df, wr_performance_boosts, rb_performance_boosts, elite_target_boost, great_target_boost):
    """Create weighted player pools"""
    pools = {}
    
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = df[df['Position'] == pos].copy()
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
            if pos == 'WR' and player_name in wr_performance_boosts:
                weight = weight * (1 + wr_performance_boosts[player_name])
            elif pos == 'RB' and player_name in rb_performance_boosts:
                weight = weight * (1 + rb_performance_boosts[player_name])
            
            weights.append(weight)
        
        pos_players['Selection_Weight'] = weights
        pools[pos] = pos_players
    
    return pools

def get_top_matchups(df, pass_defense, rush_defense, num_per_position=5):
    """Get top matchup analysis by position for display"""
    if pass_defense is None or rush_defense is None:
        return {}
    
    try:
        # Create team mapping (simplified)
        teams_sheet = pd.read_excel("NFL.xlsx", sheet_name="Teams")
        team_mapping = {}
        if len(teams_sheet.columns) >= 6:
            excel_teams = teams_sheet.iloc[:, 1]
            csv_teams = teams_sheet.iloc[:, 5]
            for excel_team, csv_team in zip(excel_teams, csv_teams):
                if pd.notna(excel_team) and pd.notna(csv_team):
                    team_mapping[excel_team] = csv_team
        
        # Map team names and create rankings
        pass_def_mapped = pass_defense.copy()
        rush_def_mapped = rush_defense.copy()
        pass_def_mapped['Team'] = pass_def_mapped['Team'].map(team_mapping).fillna(pass_def_mapped['Team'])
        rush_def_mapped['Team'] = rush_def_mapped['Team'].map(team_mapping).fillna(rush_def_mapped['Team'])
        
        pass_def_mapped['Pass_Defense_Rank'] = range(1, len(pass_def_mapped) + 1)
        rush_def_mapped['Rush_Defense_Rank'] = range(1, len(rush_def_mapped) + 1)
        
        position_matchups = {}
        
        # QB Matchups (vs worst pass defenses)
        qb_matchups = []
        worst_pass_def = pass_def_mapped.tail(10)  # Bottom 10 pass defenses
        for _, defense in worst_pass_def.iterrows():
            players_vs_def = df[(df['Opponent'] == defense['Team']) & (df['Position'] == 'QB')]
            for _, player in players_vs_def.iterrows():
                qb_matchups.append({
                    'Player': player['Nickname'],
                    'Team': player['Team'],
                    'vs': defense['Team'],
                    'FPPG': player['FPPG'],
                    'Salary': player['Salary'],
                    'Defense_Rank': defense['Pass_Defense_Rank'],
                    'YPG_Allowed': defense['Pass_YPG_Allowed'],
                    'Matchup_Quality': player['Matchup_Quality']
                })
        
        if qb_matchups:
            qb_df = pd.DataFrame(qb_matchups).sort_values('FPPG', ascending=False).head(num_per_position)
            position_matchups['QB'] = qb_df
        
        # RB Matchups (vs worst rush defenses)
        rb_matchups = []
        worst_rush_def = rush_def_mapped.tail(10)  # Bottom 10 rush defenses
        for _, defense in worst_rush_def.iterrows():
            players_vs_def = df[(df['Opponent'] == defense['Team']) & (df['Position'] == 'RB')]
            for _, player in players_vs_def.iterrows():
                rb_matchups.append({
                    'Player': player['Nickname'],
                    'Team': player['Team'],
                    'vs': defense['Team'],
                    'FPPG': player['FPPG'],
                    'Salary': player['Salary'],
                    'Defense_Rank': defense['Rush_Defense_Rank'],
                    'YPG_Allowed': defense['Rush_YPG_Allowed'],
                    'Matchup_Quality': player['Matchup_Quality']
                })
        
        if rb_matchups:
            rb_df = pd.DataFrame(rb_matchups).sort_values('FPPG', ascending=False).head(num_per_position)
            position_matchups['RB'] = rb_df
        
        # WR Matchups (vs worst pass defenses)
        wr_matchups = []
        for _, defense in worst_pass_def.iterrows():
            players_vs_def = df[(df['Opponent'] == defense['Team']) & (df['Position'] == 'WR')]
            for _, player in players_vs_def.iterrows():
                wr_matchups.append({
                    'Player': player['Nickname'],
                    'Team': player['Team'],
                    'vs': defense['Team'],
                    'FPPG': player['FPPG'],
                    'Salary': player['Salary'],
                    'Defense_Rank': defense['Pass_Defense_Rank'],
                    'YPG_Allowed': defense['Pass_YPG_Allowed'],
                    'Matchup_Quality': player['Matchup_Quality']
                })
        
        if wr_matchups:
            wr_df = pd.DataFrame(wr_matchups).sort_values('FPPG', ascending=False).head(num_per_position)
            position_matchups['WR'] = wr_df
        
        # TE Matchups (vs worst pass defenses)
        te_matchups = []
        for _, defense in worst_pass_def.iterrows():
            players_vs_def = df[(df['Opponent'] == defense['Team']) & (df['Position'] == 'TE')]
            for _, player in players_vs_def.iterrows():
                te_matchups.append({
                    'Player': player['Nickname'],
                    'Team': player['Team'],
                    'vs': defense['Team'],
                    'FPPG': player['FPPG'],
                    'Salary': player['Salary'],
                    'Defense_Rank': defense['Pass_Defense_Rank'],
                    'YPG_Allowed': defense['Pass_YPG_Allowed'],
                    'Matchup_Quality': player['Matchup_Quality']
                })
        
        if te_matchups:
            te_df = pd.DataFrame(te_matchups).sort_values('FPPG', ascending=False).head(num_per_position)
            position_matchups['TE'] = te_df
        
        return position_matchups
            
    except Exception as e:
        return {}

def generate_lineups(df, weighted_pools, num_simulations, stack_probability, elite_target_boost, great_target_boost, player_selections=None):
    """Generate optimized lineups with optional player selection constraints"""
    stacked_lineups = []
    salary_cap = 60000
    
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
    
    for simulation in range(num_simulations):
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                lineup_players = []
                
                # Handle must-include players first
                if player_selections:
                    # Must include QB
                    if player_selections['QB']['must_include']:
                        must_qb_name = player_selections['QB']['must_include'][0]  # Take first if multiple
                        qb = weighted_pools['QB'][weighted_pools['QB']['Nickname'] == must_qb_name]
                        if len(qb) == 0:
                            continue  # Skip if player not found
                    else:
                        qb_pool = weighted_pools['QB']
                        qb = qb_pool.sample(1, weights=qb_pool['Selection_Weight'])
                else:
                    # Regular QB selection
                    qb_pool = weighted_pools['QB']
                    qb = qb_pool.sample(1, weights=qb_pool['Selection_Weight'])
                
                qb_team = qb['Team'].iloc[0]
                lineup_players.append(qb)
                
                # Handle must-include RBs
                selected_rbs = pd.DataFrame()
                if player_selections and player_selections['RB']['must_include']:
                    for must_rb_name in player_selections['RB']['must_include'][:2]:  # Max 2 RBs
                        must_rb = weighted_pools['RB'][weighted_pools['RB']['Nickname'] == must_rb_name]
                        if len(must_rb) > 0:
                            selected_rbs = pd.concat([selected_rbs, must_rb])
                
                # Fill remaining RB spots
                remaining_rb_spots = 2 - len(selected_rbs)
                if remaining_rb_spots > 0:
                    available_rbs = weighted_pools['RB']
                    if len(selected_rbs) > 0:
                        used_rb_names = set(selected_rbs['Nickname'])
                        available_rbs = available_rbs[~available_rbs['Nickname'].isin(used_rb_names)]
                    
                    if len(available_rbs) >= remaining_rb_spots:
                        additional_rbs = available_rbs.sample(remaining_rb_spots, weights=available_rbs['Selection_Weight'])
                        rb = pd.concat([selected_rbs, additional_rbs])
                    else:
                        continue
                else:
                    rb = selected_rbs
                
                lineup_players.append(rb)
                
                # WR/TE selection with stacking (modified for must-include)
                wr_pool = weighted_pools['WR']
                te_pool = weighted_pools['TE']
                
                # Handle must-include WRs
                selected_wrs = pd.DataFrame()
                if player_selections and player_selections['WR']['must_include']:
                    for must_wr_name in player_selections['WR']['must_include'][:3]:  # Max 3 WRs
                        must_wr = wr_pool[wr_pool['Nickname'] == must_wr_name]
                        if len(must_wr) > 0:
                            selected_wrs = pd.concat([selected_wrs, must_wr])
                
                # Handle must-include TEs
                selected_te = pd.DataFrame()
                if player_selections and player_selections['TE']['must_include']:
                    must_te_name = player_selections['TE']['must_include'][0]  # Max 1 TE
                    must_te = te_pool[te_pool['Nickname'] == must_te_name]
                    if len(must_te) > 0:
                        selected_te = must_te
                
                # Fill remaining WR/TE spots with stacking logic
                remaining_wr_spots = 3 - len(selected_wrs)
                need_te = len(selected_te) == 0
                
                if remaining_wr_spots > 0 or need_te:
                    # Simplified stacking for must-include constraints
                    attempt_stack = random.random() < stack_probability and remaining_wr_spots > 0
                    
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
                
                # FLEX selection
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
                lineup = pd.concat(lineup_players)
                
                # Validate lineup
                if (len(lineup) == 9 and 
                    len(lineup['Nickname'].unique()) == 9 and 
                    lineup['Salary'].sum() <= salary_cap):
                    
                    total_points = lineup['Adjusted_FPPG'].sum()
                    total_salary = lineup['Salary'].sum()
                    
                    # Check stacking
                    qb_team = lineup[lineup['Position'] == 'QB']['Team'].iloc[0]
                    wr_teams = lineup[lineup['Position'] == 'WR']['Team'].tolist()
                    te_teams = lineup[lineup['Position'] == 'TE']['Team'].tolist()
                    
                    stacked_wrs_count = sum(1 for team in wr_teams if team == qb_team)
                    stacked_tes_count = sum(1 for team in te_teams if team == qb_team)
                    qb_wr_te_count = stacked_wrs_count + stacked_tes_count
                    
                    stacked_lineups.append((total_points, lineup, total_salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count))
                    break
                    
            except Exception as e:
                continue
        
        # Update progress
        progress = (simulation + 1) / num_simulations
        progress_bar.progress(progress)
        if (simulation + 1) % 100 == 0:
            status_text.text(f"Generated {simulation + 1:,} / {num_simulations:,} lineups...")
    
    progress_bar.empty()
    status_text.empty()
    
    return stacked_lineups

def main():
    st.markdown('<h1 class="main-header">🏈 FanDuel NFL DFS Optimizer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Advanced DFS lineup simulation with:**
    - ✅ QB-WR/TE Stacking with Enhanced Multi-Receiver Logic
    - ✅ Defensive Matchup Targeting (Attack Worst Defenses)
    - ✅ Fantasy Performance Boosts (WR: Targets/Receptions/FDPt, RB: FDPt/Attempts/Receptions)
    - ✅ Tournament-Optimized Strategy for 12-Person Leagues
    - ✅ Salary Cap Optimization ($60,000)
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Optimization Settings")
        
        num_simulations = st.slider("Number of Simulations", 1000, 20000, 10000, step=1000)
        stack_probability = st.slider("Stacking Probability", 0.0, 1.0, 0.55, step=0.05)
        elite_target_boost = st.slider("Elite Target Boost", 0.0, 1.0, 0.45, step=0.05)
        great_target_boost = st.slider("Great Target Boost", 0.0, 1.0, 0.25, step=0.05)
        
        st.header("� Player Selection")
        enable_player_selection = st.checkbox("Enable Player Include/Exclude", value=False)
        
        st.header("�📊 Display Settings")
        num_lineups_display = st.slider("Number of Top Lineups to Show", 5, 50, 20, step=5)
        
        generate_button = st.button("🚀 Generate Lineups", type="primary")
    
    # Load data
    with st.spinner("Loading player data..."):
        df = load_player_data()
        
    if df is not None:
        with st.spinner("Loading defensive matchup data..."):
            pass_defense, rush_defense = load_defensive_data()
            
        with st.spinner("Loading fantasy performance data..."):
            fantasy_data = load_fantasy_data()
        
        # Apply analysis
        with st.spinner("Applying matchup analysis..."):
            df = apply_matchup_analysis(df, pass_defense, rush_defense)
            
        with st.spinner("Creating performance boosts..."):
            wr_performance_boosts, rb_performance_boosts = create_performance_boosts(fantasy_data)
            
        with st.spinner("Creating weighted player pools..."):
            weighted_pools = create_weighted_pools(df, wr_performance_boosts, rb_performance_boosts, elite_target_boost, great_target_boost)
        
        # Display data summary and top matchups
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        with col1:
            st.metric("Total Players", len(df))
        with col2:
            st.metric("Elite Targets", len(df[df['Matchup_Quality'] == 'ELITE TARGET']))
        with col3:
            st.metric("WR Performance Boosts", len(wr_performance_boosts))
        with col4:
            st.metric("RB Performance Boosts", len(rb_performance_boosts))
        with col5:
            st.markdown("### 🎯 Top 5 Matchups by Position")
            
            # Get top matchups by position
            position_matchups = get_top_matchups(df, pass_defense, rush_defense, num_per_position=5)
            
            if position_matchups:
                # Create tabs for each position
                pos_tabs = st.tabs(["QB", "RB", "WR", "TE"])
                
                positions = ['QB', 'RB', 'WR', 'TE']
                emojis = ['🎯', '🏈', '⚡', '🎪']
                
                for i, (tab, pos, emoji) in enumerate(zip(pos_tabs, positions, emojis)):
                    with tab:
                        if pos in position_matchups and len(position_matchups[pos]) > 0:
                            for j, (_, matchup) in enumerate(position_matchups[pos].iterrows()):
                                if j < 3:  # Show top 3 in each tab to save space
                                    quality_icon = "🔥" if matchup['Matchup_Quality'] == 'ELITE TARGET' else ("⭐" if matchup['Matchup_Quality'] == 'Great Target' else "")
                                    st.markdown(
                                        f"**{emoji} {matchup['Player']}** vs {matchup['vs']} {quality_icon}  \n"
                                        f"${matchup['Salary']:,} | {matchup['FPPG']:.1f} pts", 
                                        help=f"Defense Rank: #{matchup['Defense_Rank']} ({matchup['YPG_Allowed']:.1f} YPG allowed)"
                                    )
                        else:
                            st.info(f"No {pos} matchups found")
            else:
                st.info("Top matchups will appear here once data is loaded")
        
        # Player Selection Interface
        if enable_player_selection:
            st.markdown('<h2 class="sub-header">👥 Player Selection</h2>', unsafe_allow_html=True)
            
            # Create tabs for different positions
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["QB", "RB", "WR", "TE", "DEF"])
            
            player_selections = {}
            
            with tab1:
                st.subheader("Quarterbacks")
                qb_players = df[df['Position'] == 'QB'].sort_values(['Team', 'Salary'], ascending=[True, False])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    must_include_qb = st.multiselect(
                        "Force these QBs in lineups",
                        options=qb_players['Nickname'].tolist(),
                        key="must_qb"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_qb = st.multiselect(
                        "Remove these QBs from consideration",
                        options=qb_players['Nickname'].tolist(),
                        key="exclude_qb"
                    )
                
                player_selections['QB'] = {'must_include': must_include_qb, 'exclude': exclude_qb}
                
                # Show QB options with salary/matchup info
                with st.expander("View All QB Options"):
                    qb_display = qb_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    qb_display['Salary'] = qb_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(qb_display, use_container_width=True)
            
            with tab2:
                st.subheader("Running Backs")
                rb_players = df[df['Position'] == 'RB'].sort_values(['Team', 'Salary'], ascending=[True, False])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    must_include_rb = st.multiselect(
                        "Force these RBs in lineups",
                        options=rb_players['Nickname'].tolist(),
                        key="must_rb"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_rb = st.multiselect(
                        "Remove these RBs from consideration",
                        options=rb_players['Nickname'].tolist(),
                        key="exclude_rb"
                    )
                
                player_selections['RB'] = {'must_include': must_include_rb, 'exclude': exclude_rb}
                
                with st.expander("View All RB Options"):
                    rb_display = rb_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    rb_display['Salary'] = rb_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(rb_display, use_container_width=True)
            
            with tab3:
                st.subheader("Wide Receivers")
                wr_players = df[df['Position'] == 'WR'].sort_values(['Team', 'Salary'], ascending=[True, False])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    must_include_wr = st.multiselect(
                        "Force these WRs in lineups",
                        options=wr_players['Nickname'].tolist(),
                        key="must_wr"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_wr = st.multiselect(
                        "Remove these WRs from consideration",
                        options=wr_players['Nickname'].tolist(),
                        key="exclude_wr"
                    )
                
                player_selections['WR'] = {'must_include': must_include_wr, 'exclude': exclude_wr}
                
                with st.expander("View All WR Options"):
                    wr_display = wr_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    wr_display['Salary'] = wr_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(wr_display, use_container_width=True)
            
            with tab4:
                st.subheader("Tight Ends")
                te_players = df[df['Position'] == 'TE'].sort_values(['Team', 'Salary'], ascending=[True, False])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    must_include_te = st.multiselect(
                        "Force these TEs in lineups",
                        options=te_players['Nickname'].tolist(),
                        key="must_te"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_te = st.multiselect(
                        "Remove these TEs from consideration",
                        options=te_players['Nickname'].tolist(),
                        key="exclude_te"
                    )
                
                player_selections['TE'] = {'must_include': must_include_te, 'exclude': exclude_te}
                
                with st.expander("View All TE Options"):
                    te_display = te_players[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    te_display['Salary'] = te_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(te_display, use_container_width=True)
            
            with tab5:
                st.subheader("Defense/Special Teams")
                def_players_tab = df[df['Position'] == 'D'].sort_values(['Team', 'Salary'], ascending=[True, False])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Must Include:**")
                    must_include_def = st.multiselect(
                        "Force these DEF in lineups",
                        options=def_players_tab['Nickname'].tolist(),
                        key="must_def"
                    )
                
                with col2:
                    st.write("**Exclude:**")
                    exclude_def = st.multiselect(
                        "Remove these DEF from consideration",
                        options=def_players_tab['Nickname'].tolist(),
                        key="exclude_def"
                    )
                
                player_selections['D'] = {'must_include': must_include_def, 'exclude': exclude_def}
                
                with st.expander("View All DEF Options"):
                    def_display = def_players_tab[['Nickname', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    def_display['Salary'] = def_display['Salary'].apply(lambda x: f"${x:,}")
                    st.dataframe(def_display, use_container_width=True)
        
        else:
            player_selections = None
        
        if generate_button:
            with st.spinner(f"Generating {num_simulations:,} optimized lineups..."):
                stacked_lineups = generate_lineups(df, weighted_pools, num_simulations, stack_probability, elite_target_boost, great_target_boost, player_selections)
                st.session_state.stacked_lineups = stacked_lineups
                st.session_state.lineups_generated = True
        
        # Display results
        if st.session_state.lineups_generated and st.session_state.stacked_lineups:
            stacked_lineups = st.session_state.stacked_lineups
            
            st.markdown('<h2 class="sub-header">🏆 Optimized Lineups</h2>', unsafe_allow_html=True)
            
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
            
            # Display lineups
            for i, (points, lineup, salary, stacked_wrs_count, stacked_tes_count, qb_wr_te_count) in enumerate(top_lineups, 1):
                with st.expander(f"Lineup #{i}: {points:.2f} points | ${salary:,} | {'QB+' + str(qb_wr_te_count) + ' receivers' if qb_wr_te_count > 0 else 'No stack'}"):
                    
                    # Create lineup display
                    lineup_display = lineup[['Nickname', 'Position', 'Team', 'Salary', 'FPPG', 'Matchup_Quality']].copy()
                    lineup_display['Salary'] = lineup_display['Salary'].apply(lambda x: f"${x:,}")
                    lineup_display['FPPG'] = lineup_display['FPPG'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(lineup_display, use_container_width=True)
                    
                    # Show boosts
                    fantasy_boosted = 0
                    elite_targets = 0
                    qb_team = lineup[lineup['Position'] == 'QB']['Team'].iloc[0]
                    
                    for _, player in lineup.iterrows():
                        if player['Position'] == 'WR' and player['Nickname'] in wr_performance_boosts:
                            fantasy_boosted += 1
                        elif player['Position'] == 'RB' and player['Nickname'] in rb_performance_boosts:
                            fantasy_boosted += 1
                        
                        if player['Matchup_Quality'] == 'ELITE TARGET':
                            elite_targets += 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"🏈 Fantasy-boosted players: {fantasy_boosted}")
                    with col2:
                        st.write(f"🎯 Elite targets: {elite_targets}")
            
            # Stacking analysis
            if len(stacked_lineups) > 0:
                st.markdown('<h2 class="sub-header">📊 Stacking Analysis</h2>', unsafe_allow_html=True)
                
                no_stack = [lineup for lineup in stacked_lineups if lineup[5] == 0]
                single_stack = [lineup for lineup in stacked_lineups if lineup[5] == 1]
                double_stack = [lineup for lineup in stacked_lineups if lineup[5] == 2]
                triple_stack = [lineup for lineup in stacked_lineups if lineup[5] >= 3]
                
                stack_data = {
                    'Stack Type': ['No Stack', '1 Receiver', '2 Receivers', '3+ Receivers'],
                    'Count': [len(no_stack), len(single_stack), len(double_stack), len(triple_stack)],
                    'Avg Points': [
                        np.mean([lineup[0] for lineup in no_stack]) if no_stack else 0,
                        np.mean([lineup[0] for lineup in single_stack]) if single_stack else 0,
                        np.mean([lineup[0] for lineup in double_stack]) if double_stack else 0,
                        np.mean([lineup[0] for lineup in triple_stack]) if triple_stack else 0
                    ]
                }
                
                stack_df = pd.DataFrame(stack_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(stack_df, x='Stack Type', y='Count', title='Stacking Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(stack_df, x='Stack Type', y='Avg Points', title='Average Points by Stack Type')
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
