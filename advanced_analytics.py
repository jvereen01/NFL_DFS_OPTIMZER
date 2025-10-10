"""
Advanced analytics dashboard for DFS optimizer
Includes correlation analysis, ownership projections, ROI tracking, and performance insights
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AdvancedAnalytics:
    """Comprehensive analytics for DFS optimization"""
    
    def __init__(self):
        self.analytics_cache = {}
    
    def calculate_correlation_matrix(self, df: pd.DataFrame, lineup_data: List) -> pd.DataFrame:
        """Calculate player correlation matrix for stack optimization"""
        if not lineup_data:
            return pd.DataFrame()
        
        # Create player-lineup matrix
        player_names = set()
        for _, lineup, _, _, _, _ in lineup_data:
            player_names.update(lineup['Nickname'].tolist())
        
        player_names = sorted(list(player_names))
        lineup_matrix = []
        
        for _, lineup, _, _, _, _ in lineup_data:
            lineup_players = lineup['Nickname'].tolist()
            row = [1 if player in lineup_players else 0 for player in player_names]
            lineup_matrix.append(row)
        
        lineup_df = pd.DataFrame(lineup_matrix, columns=player_names)
        correlation_matrix = lineup_df.corr()
        
        return correlation_matrix
    
    def generate_ownership_projections(self, df: pd.DataFrame, lineup_data: List, 
                                     num_lineups: int = 10000) -> pd.DataFrame:
        """Generate ownership projections based on lineup generation patterns"""
        if not lineup_data:
            return pd.DataFrame()
        
        player_usage = {}
        total_lineups = len(lineup_data)
        
        for _, lineup, _, _, _, _ in lineup_data:
            for _, player in lineup.iterrows():
                player_name = player['Nickname']
                position = player['Position']
                
                if player_name not in player_usage:
                    player_usage[player_name] = {
                        'position': position,
                        'salary': player['Salary'],
                        'fppg': player['FPPG'],
                        'usage_count': 0,
                        'team': player['Team']
                    }
                player_usage[player_name]['usage_count'] += 1
        
        # Calculate ownership percentages and projections
        ownership_data = []
        for player_name, data in player_usage.items():
            ownership_pct = (data['usage_count'] / total_lineups) * 100
            
            # Project ownership in larger tournament field
            tournament_ownership = self._project_tournament_ownership(ownership_pct, data['salary'], data['position'])
            
            ownership_data.append({
                'Player': player_name,
                'Position': data['position'],
                'Team': data['team'],
                'Salary': data['salary'],
                'FPPG': data['fppg'],
                'Our_Usage_%': ownership_pct,
                'Projected_Tournament_%': tournament_ownership,
                'Value_Score': self._calculate_value_score(data['fppg'], data['salary'], ownership_pct),
                'Leverage_Score': self._calculate_leverage_score(data['fppg'], tournament_ownership)
            })
        
        return pd.DataFrame(ownership_data).sort_values('Our_Usage_%', ascending=False)
    
    def _project_tournament_ownership(self, our_ownership: float, salary: int, position: str) -> float:
        """Project tournament ownership based on our usage and player characteristics"""
        # Base tournament ownership adjustments
        base_adjustments = {
            'QB': 0.8,  # QBs typically lower owned
            'RB': 1.1,  # RBs typically higher owned
            'WR': 1.0,  # Baseline
            'TE': 0.9,  # TEs typically lower owned
            'D': 0.7    # Defenses typically very low owned
        }
        
        position_multiplier = base_adjustments.get(position, 1.0)
        
        # Salary-based adjustments (higher salary = higher ownership typically)
        if salary >= 9000:
            salary_multiplier = 1.3
        elif salary >= 7000:
            salary_multiplier = 1.1
        elif salary >= 5000:
            salary_multiplier = 1.0
        else:
            salary_multiplier = 0.8
        
        projected = our_ownership * position_multiplier * salary_multiplier
        
        # Cap projections at reasonable levels
        max_ownership = {'QB': 25, 'RB': 35, 'WR': 30, 'TE': 20, 'D': 15}
        return min(projected, max_ownership.get(position, 30))
    
    def _calculate_value_score(self, fppg: float, salary: int, ownership: float) -> float:
        """Calculate value score combining points per dollar and ownership"""
        ppd = (fppg / salary) * 1000  # Points per $1000
        ownership_factor = max(0.1, (100 - ownership) / 100)  # Bonus for low ownership
        return ppd * ownership_factor
    
    def _calculate_leverage_score(self, fppg: float, projected_ownership: float) -> float:
        """Calculate leverage score for tournament play"""
        if projected_ownership <= 0:
            return 0
        return fppg / projected_ownership
    
    def analyze_position_stacking(self, lineup_data: List) -> Dict:
        """Analyze stacking patterns and effectiveness"""
        stack_analysis = {
            'qb_stack_performance': [],
            'rb_correlation': [],
            'defense_correlation': [],
            'salary_efficiency': []
        }
        
        for points, lineup, salary, stacked_wrs, stacked_tes, total_stack in lineup_data:
            qb_team = lineup[lineup['Position'] == 'QB']['Team'].iloc[0]
            
            # QB stack analysis
            if total_stack > 0:
                stack_analysis['qb_stack_performance'].append({
                    'stack_size': total_stack,
                    'points': points,
                    'salary': salary,
                    'qb_team': qb_team
                })
        
        return stack_analysis
    
    def generate_lineup_performance_insights(self, lineup_data: List) -> Dict:
        """Generate comprehensive performance insights"""
        if not lineup_data:
            return {}
        
        points_list = [lineup[0] for lineup in lineup_data]
        salary_list = [lineup[2] for lineup in lineup_data]
        stack_sizes = [lineup[5] for lineup in lineup_data]
        
        insights = {
            'performance_metrics': {
                'mean_points': np.mean(points_list),
                'median_points': np.median(points_list),
                'std_points': np.std(points_list),
                'min_points': np.min(points_list),
                'max_points': np.max(points_list),
                'points_75th_percentile': np.percentile(points_list, 75),
                'points_90th_percentile': np.percentile(points_list, 90)
            },
            'salary_metrics': {
                'mean_salary': np.mean(salary_list),
                'median_salary': np.median(salary_list),
                'salary_efficiency': np.mean([p/s*1000 for p, s in zip(points_list, salary_list)])
            },
            'stacking_metrics': {
                'avg_stack_size': np.mean(stack_sizes),
                'no_stack_count': sum(1 for size in stack_sizes if size == 0),
                'single_stack_count': sum(1 for size in stack_sizes if size == 1),
                'multi_stack_count': sum(1 for size in stack_sizes if size >= 2)
            }
        }
        
        # Calculate optimal lineup characteristics
        top_10_percent = sorted(lineup_data, key=lambda x: x[0], reverse=True)[:max(1, len(lineup_data)//10)]
        
        if top_10_percent:
            top_points = [lineup[0] for lineup in top_10_percent]
            top_salaries = [lineup[2] for lineup in top_10_percent]
            top_stacks = [lineup[5] for lineup in top_10_percent]
            
            insights['top_10_percent'] = {
                'avg_points': np.mean(top_points),
                'avg_salary': np.mean(top_salaries),
                'avg_stack_size': np.mean(top_stacks),
                'stack_distribution': {
                    'no_stack': sum(1 for size in top_stacks if size == 0) / len(top_stacks),
                    'single_stack': sum(1 for size in top_stacks if size == 1) / len(top_stacks),
                    'multi_stack': sum(1 for size in top_stacks if size >= 2) / len(top_stacks)
                }
            }
        
        return insights
    
    def create_advanced_visualizations(self, df: pd.DataFrame, lineup_data: List, 
                                     ownership_df: pd.DataFrame) -> Dict:
        """Create advanced visualization charts"""
        charts = {}
        
        if not lineup_data:
            return charts
        
        # 1. Points vs Salary Scatter with Stack Size
        points_list = [lineup[0] for lineup in lineup_data]
        salary_list = [lineup[2] for lineup in lineup_data]
        stack_sizes = [lineup[5] for lineup in lineup_data]
        
        fig = px.scatter(
            x=salary_list, 
            y=points_list, 
            color=stack_sizes,
            title="Points vs Salary by Stack Size",
            labels={"x": "Salary", "y": "Projected Points", "color": "Stack Size"},
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=500)
        charts['points_vs_salary'] = fig
        
        # 2. Ownership vs Value Chart
        if not ownership_df.empty:
            fig = px.scatter(
                ownership_df,
                x="Projected_Tournament_%",
                y="Value_Score",
                color="Position",
                size="FPPG",
                hover_data=["Player", "Salary"],
                title="Tournament Ownership vs Value Score"
            )
            fig.update_layout(height=500)
            charts['ownership_vs_value'] = fig
        
        # 3. Position Distribution Pie Chart
        if not ownership_df.empty:
            position_usage = ownership_df.groupby('Position')['Our_Usage_%'].mean().reset_index()
            fig = px.pie(
                position_usage,
                values='Our_Usage_%',
                names='Position',
                title="Average Usage by Position"
            )
            charts['position_distribution'] = fig
        
        # 4. Correlation Heatmap (top correlated players)
        correlation_matrix = self.calculate_correlation_matrix(df, lineup_data)
        if not correlation_matrix.empty and len(correlation_matrix) > 1:
            # Get top correlated pairs
            top_corr_players = correlation_matrix.abs().unstack().sort_values(ascending=False)
            top_corr_players = top_corr_players[top_corr_players < 1.0]  # Remove self-correlations
            
            if len(top_corr_players) > 0:
                # Create heatmap of top 20 most correlated players
                top_players = list(set(top_corr_players.head(40).index.get_level_values(0).tolist() + 
                                     top_corr_players.head(40).index.get_level_values(1).tolist()))[:20]
                
                subset_corr = correlation_matrix.loc[top_players, top_players]
                
                fig = px.imshow(
                    subset_corr,
                    title="Player Correlation Matrix (Top Correlated)",
                    aspect="auto",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=600)
                charts['correlation_heatmap'] = fig
        
        # 5. Stack Performance Distribution
        stack_performance = {}
        for points, lineup, salary, _, _, stack_size in lineup_data:
            if stack_size not in stack_performance:
                stack_performance[stack_size] = []
            stack_performance[stack_size].append(points)
        
        if len(stack_performance) > 1:
            stack_data = []
            for stack_size, points_list in stack_performance.items():
                for points in points_list:
                    stack_data.append({"Stack_Size": f"{stack_size} receivers", "Points": points})
            
            stack_df = pd.DataFrame(stack_data)
            fig = px.box(
                stack_df,
                x="Stack_Size",
                y="Points",
                title="Points Distribution by Stack Size"
            )
            charts['stack_performance'] = fig
        
        return charts
    
    def generate_roi_projections(self, lineup_data: List, entry_fee: float = 0.25, 
                               tournament_size: int = 100000) -> Dict:
        """Generate ROI projections for tournament play"""
        if not lineup_data:
            return {}
        
        points_list = [lineup[0] for lineup in lineup_data]
        
        # Simulate tournament results
        simulations = []
        for _ in range(1000):  # 1000 tournament simulations
            # Random sample of our lineups
            our_lineup_points = np.random.choice(points_list)
            
            # Simulate field distribution (normal distribution with lower mean)
            field_mean = np.mean(points_list) * 0.85  # Field typically scores lower
            field_std = np.std(points_list) * 1.2     # Field has higher variance
            field_scores = np.random.normal(field_mean, field_std, tournament_size - 1)
            
            all_scores = np.append(field_scores, our_lineup_points)
            rank = stats.rankdata(-all_scores)[tournament_size - 1]  # Our rank (negative for descending)
            
            # Calculate payout based on rank
            payout = self._calculate_tournament_payout(rank, tournament_size, entry_fee)
            roi = (payout - entry_fee) / entry_fee if entry_fee > 0 else 0
            
            simulations.append({
                'rank': rank,
                'percentile': rank / tournament_size,
                'payout': payout,
                'roi': roi,
                'our_score': our_lineup_points
            })
        
        simulation_df = pd.DataFrame(simulations)
        
        return {
            'avg_roi': simulation_df['roi'].mean(),
            'median_roi': simulation_df['roi'].median(),
            'positive_roi_rate': (simulation_df['roi'] > 0).mean(),
            'avg_rank': simulation_df['rank'].mean(),
            'top_1_percent_rate': (simulation_df['percentile'] <= 0.01).mean(),
            'top_10_percent_rate': (simulation_df['percentile'] <= 0.10).mean(),
            'cash_rate': (simulation_df['percentile'] <= 0.20).mean(),  # Assuming top 20% cash
            'simulation_data': simulation_df
        }
    
    def _calculate_tournament_payout(self, rank: int, tournament_size: int, entry_fee: float) -> float:
        """Calculate tournament payout based on rank"""
        percentile = rank / tournament_size
        
        # Typical tournament payout structure
        if percentile <= 0.001:  # Top 0.1%
            return entry_fee * tournament_size * 0.15  # 15% of prize pool to winner
        elif percentile <= 0.01:  # Top 1%
            return entry_fee * 50
        elif percentile <= 0.05:  # Top 5%
            return entry_fee * 10
        elif percentile <= 0.10:  # Top 10%
            return entry_fee * 5
        elif percentile <= 0.20:  # Top 20% (cash line)
            return entry_fee * 2
        else:
            return 0  # No payout

class AdvancedMetrics:
    """Additional advanced metrics for lineup optimization"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float]) -> float:
        """Calculate Sharpe ratio for lineup performance"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal bet sizing using Kelly Criterion"""
        if avg_loss <= 0:
            return 0
        
        b = avg_win / avg_loss  # Ratio of win to loss
        p = win_rate  # Probability of winning
        q = 1 - p     # Probability of losing
        
        kelly_pct = (b * p - q) / b
        return max(0, kelly_pct)  # Don't bet negative amounts