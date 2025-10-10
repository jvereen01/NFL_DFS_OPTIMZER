"""
Data validation and cleaning module for DFS optimizer
Provides robust validation, error reporting, and automatic data cleaning
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import re
from datetime import datetime

class DataValidator:
    """Comprehensive data validation for DFS optimizer"""
    
    def __init__(self):
        self.validation_results = {
            'errors': [],
            'warnings': [],
            'fixes_applied': [],
            'data_quality_score': 100
        }
        
        # Expected data schemas
        self.expected_schemas = {
            'player_data': {
                'required_columns': ['Nickname', 'Position', 'Team', 'Salary', 'FPPG', 'Opponent', 'Id'],
                'optional_columns': ['Injury Indicator', 'Played'],
                'data_types': {
                    'Salary': 'numeric',
                    'FPPG': 'numeric',
                    'Position': 'categorical',
                    'Team': 'categorical'
                },
                'constraints': {
                    'Salary': {'min': 3000, 'max': 15000},
                    'FPPG': {'min': 0, 'max': 50},
                    'Position': {'values': ['QB', 'RB', 'WR', 'TE', 'D']}
                }
            },
            'fantasy_data': {
                'required_columns': ['Player', 'FantPos', 'FDPt'],
                'optional_columns': ['Tgt', 'Rec', 'Att_1', 'PosRank', 'TD_3', 'Yds_2'],
                'data_types': {
                    'FDPt': 'numeric',
                    'Tgt': 'numeric',
                    'Rec': 'numeric'
                }
            }
        }
    
    def validate_player_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Comprehensive validation of player data"""
        self.validation_results = {'errors': [], 'warnings': [], 'fixes_applied': [], 'data_quality_score': 100}
        
        if df is None or df.empty:
            self.validation_results['errors'].append("Player data is empty or None")
            self.validation_results['data_quality_score'] = 0
            return df, self.validation_results
        
        # Check required columns
        schema = self.expected_schemas['player_data']
        missing_columns = [col for col in schema['required_columns'] if col not in df.columns]
        if missing_columns:
            self.validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            self.validation_results['data_quality_score'] -= 20
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate data types and fix common issues
        df = self._fix_numeric_columns(df, ['Salary', 'FPPG'])
        df = self._fix_categorical_columns(df, ['Position', 'Team', 'Opponent'])
        
        # Validate salary ranges
        if 'Salary' in df.columns:
            salary_issues = df[(df['Salary'] < 3000) | (df['Salary'] > 15000)]
            if len(salary_issues) > 0:
                self.validation_results['warnings'].append(f"{len(salary_issues)} players with unusual salary values")
                self.validation_results['data_quality_score'] -= 5
        
        # Validate FPPG ranges
        if 'FPPG' in df.columns:
            fppg_issues = df[(df['FPPG'] < 0) | (df['FPPG'] > 50)]
            if len(fppg_issues) > 0:
                self.validation_results['warnings'].append(f"{len(fppg_issues)} players with unusual FPPG values")
                self.validation_results['data_quality_score'] -= 5
        
        # Check for duplicate players
        if 'Nickname' in df.columns:
            duplicates = df[df.duplicated(subset=['Nickname'], keep=False)]
            if len(duplicates) > 0:
                self.validation_results['warnings'].append(f"{len(duplicates)} duplicate players found")
                # Keep first occurrence of each duplicate
                df = df.drop_duplicates(subset=['Nickname'], keep='first')
                self.validation_results['fixes_applied'].append("Removed duplicate players")
        
        # Validate team names
        df = self._standardize_team_names(df)
        
        # Check position distribution
        if 'Position' in df.columns:
            position_counts = df['Position'].value_counts()
            expected_positions = ['QB', 'RB', 'WR', 'TE', 'D']
            missing_positions = [pos for pos in expected_positions if pos not in position_counts.index]
            if missing_positions:
                self.validation_results['warnings'].append(f"Missing positions: {missing_positions}")
                self.validation_results['data_quality_score'] -= 10
        
        return df, self.validation_results
    
    def validate_fantasy_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate fantasy performance data"""
        validation_results = {'errors': [], 'warnings': [], 'fixes_applied': [], 'data_quality_score': 100}
        
        if df is None or df.empty:
            validation_results['warnings'].append("Fantasy data is empty - performance boosts disabled")
            return df, validation_results
        
        # Check required columns
        schema = self.expected_schemas['fantasy_data']
        missing_columns = [col for col in schema['required_columns'] if col not in df.columns]
        if missing_columns:
            validation_results['errors'].append(f"Missing fantasy data columns: {missing_columns}")
            validation_results['data_quality_score'] -= 30
        
        # Fix numeric columns
        numeric_columns = ['Tgt', 'Rec', 'FDPt', 'Att_1', 'PosRank', 'TD_3', 'Yds_2']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        if 'FDPt' in df.columns:
            before_count = len(df)
            df = df.dropna(subset=['FDPt'])
            after_count = len(df)
            if before_count != after_count:
                validation_results['fixes_applied'].append(f"Removed {before_count - after_count} rows with missing FDPt")
        
        return df, validation_results
    
    def _fix_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fix common issues in numeric columns"""
        for col in columns:
            if col in df.columns:
                # Remove currency symbols and commas
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill NaN values with median for critical columns
                if col in ['Salary', 'FPPG'] and df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.validation_results['fixes_applied'].append(f"Filled missing {col} values with median ({median_val:.2f})")
        
        return df
    
    def _fix_categorical_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fix common issues in categorical columns"""
        for col in columns:
            if col in df.columns:
                # Strip whitespace and convert to string
                df[col] = df[col].astype(str).str.strip()
                
                # Fix common team name variations
                if col in ['Team', 'Opponent']:
                    df[col] = df[col].replace({
                        'JAX': 'JAC',  # Jacksonville
                        'WSH': 'WAS',  # Washington
                        'LV': 'LVR',   # Las Vegas Raiders
                        'LA': 'LAR',   # Clarify LA teams
                    })
        
        return df
    
    def _standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize team name formats"""
        team_mappings = {
            # Common variations to standard abbreviations
            'arizona cardinals': 'ARI', 'cardinals': 'ARI',
            'atlanta falcons': 'ATL', 'falcons': 'ATL',
            'baltimore ravens': 'BAL', 'ravens': 'BAL',
            'buffalo bills': 'BUF', 'bills': 'BUF',
            'carolina panthers': 'CAR', 'panthers': 'CAR',
            'chicago bears': 'CHI', 'bears': 'CHI',
            'cincinnati bengals': 'CIN', 'bengals': 'CIN',
            'cleveland browns': 'CLE', 'browns': 'CLE',
            'dallas cowboys': 'DAL', 'cowboys': 'DAL',
            'denver broncos': 'DEN', 'broncos': 'DEN',
            'detroit lions': 'DET', 'lions': 'DET',
            'green bay packers': 'GB', 'packers': 'GB',
            'houston texans': 'HOU', 'texans': 'HOU',
            'indianapolis colts': 'IND', 'colts': 'IND',
            'jacksonville jaguars': 'JAC', 'jaguars': 'JAC',
            'kansas city chiefs': 'KC', 'chiefs': 'KC',
            'las vegas raiders': 'LVR', 'raiders': 'LVR',
            'los angeles chargers': 'LAC', 'chargers': 'LAC',
            'los angeles rams': 'LAR', 'rams': 'LAR',
            'miami dolphins': 'MIA', 'dolphins': 'MIA',
            'minnesota vikings': 'MIN', 'vikings': 'MIN',
            'new england patriots': 'NE', 'patriots': 'NE',
            'new orleans saints': 'NO', 'saints': 'NO',
            'new york giants': 'NYG', 'giants': 'NYG',
            'new york jets': 'NYJ', 'jets': 'NYJ',
            'philadelphia eagles': 'PHI', 'eagles': 'PHI',
            'pittsburgh steelers': 'PIT', 'steelers': 'PIT',
            'san francisco 49ers': 'SF', '49ers': 'SF',
            'seattle seahawks': 'SEA', 'seahawks': 'SEA',
            'tampa bay buccaneers': 'TB', 'buccaneers': 'TB',
            'tennessee titans': 'TEN', 'titans': 'TEN',
            'washington commanders': 'WAS', 'commanders': 'WAS'
        }
        
        for col in ['Team', 'Opponent']:
            if col in df.columns:
                # Apply mappings for full names to abbreviations
                df[col] = df[col].str.lower().replace(team_mappings).str.upper()
        
        return df
    
    def generate_data_quality_report(self, validation_results: Dict) -> str:
        """Generate a human-readable data quality report"""
        report = []
        
        score = validation_results['data_quality_score']
        if score >= 95:
            report.append("✅ **Excellent data quality**")
        elif score >= 85:
            report.append("⚠️ **Good data quality with minor issues**")
        elif score >= 70:
            report.append("🔶 **Acceptable data quality with some concerns**")
        else:
            report.append("❌ **Poor data quality - significant issues detected**")
        
        if validation_results['errors']:
            report.append("\n**Errors:**")
            for error in validation_results['errors']:
                report.append(f"• {error}")
        
        if validation_results['warnings']:
            report.append("\n**Warnings:**")
            for warning in validation_results['warnings']:
                report.append(f"• {warning}")
        
        if validation_results['fixes_applied']:
            report.append("\n**Automatic fixes applied:**")
            for fix in validation_results['fixes_applied']:
                report.append(f"• {fix}")
        
        return "\n".join(report)

class DataCleaner:
    """Advanced data cleaning utilities"""
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """Detect outliers using IQR or Z-score method"""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            return z_scores > 3
        
        return pd.Series([False] * len(df))
    
    @staticmethod
    def suggest_data_fixes(df: pd.DataFrame) -> List[str]:
        """Suggest potential data quality improvements"""
        suggestions = []
        
        # Check for missing data patterns
        missing_data = df.isnull().sum()
        if missing_data.any():
            suggestions.append(f"Consider filling missing data in: {missing_data[missing_data > 0].index.tolist()}")
        
        # Check for potential duplicate entries
        if 'Nickname' in df.columns:
            duplicates = df['Nickname'].duplicated().sum()
            if duplicates > 0:
                suggestions.append(f"Found {duplicates} potential duplicate players")
        
        # Check salary distribution
        if 'Salary' in df.columns:
            salary_outliers = DataCleaner.detect_outliers(df, 'Salary')
            if salary_outliers.any():
                suggestions.append(f"Found {salary_outliers.sum()} players with unusual salary values")
        
        return suggestions

# Usage example for integration with main app
def integrate_data_validation():
    """Example of how to integrate data validation into the main app"""
    
    validator = DataValidator()
    
    # In your load_player_data function:
    def enhanced_load_player_data():
        # ... existing loading code ...
        
        # Add validation
        validated_df, validation_results = validator.validate_player_data(df)
        
        # Display validation results
        if validation_results['data_quality_score'] < 90:
            with st.expander("📊 Data Quality Report", expanded=True):
                report = validator.generate_data_quality_report(validation_results)
                st.markdown(report)
        
        return validated_df