"""
Export templates module for DFS optimizer
Support for multiple platforms: FanDuel, DraftKings, SuperDraft, Yahoo, etc.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import csv
import io

@dataclass
class ExportTemplate:
    """Template definition for DFS platform exports"""
    platform: str
    file_format: str  # 'csv', 'json', 'xml'
    roster_positions: List[str]
    required_columns: List[str]
    optional_columns: List[str]
    position_mapping: Dict[str, str]
    salary_cap: int
    max_players_per_team: int
    file_extension: str
    header_row: bool = True
    
class PlatformTemplates:
    """Collection of templates for different DFS platforms"""
    
    @staticmethod
    def get_fanduel_template() -> ExportTemplate:
        """FanDuel template configuration"""
        return ExportTemplate(
            platform="FanDuel",
            file_format="csv",
            roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"],
            required_columns=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"],
            optional_columns=["lineup_name", "captain"],
            position_mapping={"D": "DEF", "DEF": "DEF"},
            salary_cap=60000,
            max_players_per_team=4,
            file_extension=".csv"
        )
    
    @staticmethod
    def get_draftkings_template() -> ExportTemplate:
        """DraftKings template configuration"""
        return ExportTemplate(
            platform="DraftKings",
            file_format="csv", 
            roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
            required_columns=["LineupName", "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"],
            optional_columns=["entry_fee", "contest_id"],
            position_mapping={"D": "DST", "DEF": "DST"},
            salary_cap=50000,
            max_players_per_team=8,
            file_extension=".csv"
        )
    
    @staticmethod
    def get_superdraft_template() -> ExportTemplate:
        """SuperDraft template configuration"""
        return ExportTemplate(
            platform="SuperDraft",
            file_format="csv",
            roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
            required_columns=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"],
            optional_columns=["lineup_name"],
            position_mapping={"D": "DST", "DEF": "DST"},
            salary_cap=50000,
            max_players_per_team=6,
            file_extension=".csv"
        )
    
    @staticmethod
    def get_yahoo_template() -> ExportTemplate:
        """Yahoo template configuration"""
        return ExportTemplate(
            platform="Yahoo",
            file_format="csv",
            roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"],
            required_columns=["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DEF"],
            optional_columns=["lineup_name", "entry_id"],
            position_mapping={"D": "DEF"},
            salary_cap=200,  # Yahoo uses different salary structure
            max_players_per_team=6,
            file_extension=".csv"
        )

class LineupExporter:
    """Export lineups to various DFS platforms"""
    
    def __init__(self):
        self.templates = {
            'fanduel': PlatformTemplates.get_fanduel_template(),
            'draftkings': PlatformTemplates.get_draftkings_template(),
            'superdraft': PlatformTemplates.get_superdraft_template(),
            'yahoo': PlatformTemplates.get_yahoo_template()
        }
    
    def export_lineups(self, lineup_data: List[Tuple], platform: str, 
                      contest_info: Optional[Dict] = None, 
                      max_lineups: int = 150) -> str:
        """Export lineups for specified platform"""
        
        if platform.lower() not in self.templates:
            raise ValueError(f"Unsupported platform: {platform}. Supported: {list(self.templates.keys())}")
        
        template = self.templates[platform.lower()]
        export_lineups = sorted(lineup_data, key=lambda x: x[0], reverse=True)[:max_lineups]
        
        if platform.lower() == 'fanduel':
            return self._export_fanduel(export_lineups, template, contest_info)
        elif platform.lower() == 'draftkings':
            return self._export_draftkings(export_lineups, template, contest_info)
        elif platform.lower() == 'superdraft':
            return self._export_superdraft(export_lineups, template, contest_info)
        elif platform.lower() == 'yahoo':
            return self._export_yahoo(export_lineups, template, contest_info)
        else:
            return self._export_generic(export_lineups, template, contest_info)
    
    def _export_fanduel(self, lineup_data: List[Tuple], template: ExportTemplate, 
                       contest_info: Optional[Dict] = None) -> str:
        """Export lineups for FanDuel"""
        csv_lines = ['QB,RB,RB,WR,WR,WR,TE,FLEX,DEF']
        
        csv_data = []
        for i, (points, lineup, salary, _, _, _) in enumerate(lineup_data):
            # Build roster mapping
            positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DEF': []}
            
            for _, player in lineup.iterrows():
                pos = player['Position']
                player_id = player['Id']
                
                if pos == 'D':  # Map D to DEF
                    positions['DEF'].append(player_id)
                elif pos in positions:
                    positions[pos].append(player_id)
            
            # Validate lineup completeness
            if (len(positions['QB']) < 1 or len(positions['RB']) < 2 or 
                len(positions['WR']) < 3 or len(positions['TE']) < 1 or 
                len(positions['DEF']) < 1):
                continue
            
            # Build FanDuel format: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF
            row = [
                positions['QB'][0],     # QB
                positions['RB'][0],     # RB1
                positions['RB'][1],     # RB2  
                positions['WR'][0],     # WR1
                positions['WR'][1],     # WR2
                positions['WR'][2],     # WR3
                positions['TE'][0],     # TE
                '',                     # FLEX
                positions['DEF'][0]     # DEF
            ]
            
            # Determine FLEX player
            if len(positions['RB']) > 2:
                row[7] = positions['RB'][2]
            elif len(positions['WR']) > 3:
                row[7] = positions['WR'][3]
            elif len(positions['TE']) > 1:
                row[7] = positions['TE'][1]
            else:
                continue
            
            # Add lineup data with commas
            lineup_data_str = ','.join(map(str, row))
            csv_lines.append(lineup_data_str)
            csv_data.append(row)
        
        return '\n'.join(csv_lines)
    
    def _export_draftkings(self, lineup_data: List[Tuple], template: ExportTemplate,
                          contest_info: Optional[Dict] = None) -> str:
        """Export lineups for DraftKings"""
        csv_lines = ['LineupName,QB,RB1,RB2,WR1,WR2,WR3,TE,FLEX,DST']
        
        for i, (points, lineup, salary, _, _, _) in enumerate(lineup_data):
            # Build roster mapping
            positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DST': []}
            
            for _, player in lineup.iterrows():
                pos = player['Position']
                player_name = player['Nickname']
                
                if pos == 'D':
                    positions['DST'].append(player_name)
                elif pos in positions:
                    positions[pos].append(player_name)
            
            # Validate lineup
            if (len(positions['QB']) < 1 or len(positions['RB']) < 2 or 
                len(positions['WR']) < 3 or len(positions['TE']) < 1 or 
                len(positions['DST']) < 1):
                continue
            
            # Build DraftKings format
            lineup_name = f"Lineup_{i+1}"
            row = [
                lineup_name,
                positions['QB'][0],     # QB
                positions['RB'][0],     # RB1
                positions['RB'][1],     # RB2
                positions['WR'][0],     # WR1
                positions['WR'][1],     # WR2
                positions['WR'][2],     # WR3
                positions['TE'][0],     # TE
                '',                     # FLEX
                positions['DST'][0]     # DST
            ]
            
            # Determine FLEX
            if len(positions['RB']) > 2:
                row[8] = positions['RB'][2]
            elif len(positions['WR']) > 3:
                row[8] = positions['WR'][3]
            elif len(positions['TE']) > 1:
                row[8] = positions['TE'][1]
            else:
                continue
            
            csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)
    
    def _export_superdraft(self, lineup_data: List[Tuple], template: ExportTemplate,
                          contest_info: Optional[Dict] = None) -> str:
        """Export lineups for SuperDraft"""
        csv_lines = ['QB,RB,RB,WR,WR,WR,TE,FLEX,DST']
        
        for i, (points, lineup, salary, _, _, _) in enumerate(lineup_data):
            # Similar to DraftKings but without lineup names
            positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DST': []}
            
            for _, player in lineup.iterrows():
                pos = player['Position']
                player_name = player['Nickname']
                
                if pos == 'D':
                    positions['DST'].append(player_name)
                elif pos in positions:
                    positions[pos].append(player_name)
            
            # Validate and build lineup
            if (len(positions['QB']) >= 1 and len(positions['RB']) >= 2 and 
                len(positions['WR']) >= 3 and len(positions['TE']) >= 1 and 
                len(positions['DST']) >= 1):
                
                row = [
                    positions['QB'][0],
                    positions['RB'][0],
                    positions['RB'][1],
                    positions['WR'][0],
                    positions['WR'][1],
                    positions['WR'][2],
                    positions['TE'][0],
                    positions['RB'][2] if len(positions['RB']) > 2 else 
                    (positions['WR'][3] if len(positions['WR']) > 3 else 
                     (positions['TE'][1] if len(positions['TE']) > 1 else '')),
                    positions['DST'][0]
                ]
                
                if row[7]:  # Only add if FLEX is filled
                    csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)
    
    def _export_yahoo(self, lineup_data: List[Tuple], template: ExportTemplate,
                     contest_info: Optional[Dict] = None) -> str:
        """Export lineups for Yahoo"""
        csv_lines = ['QB,RB1,RB2,WR1,WR2,WR3,TE,FLEX,DEF']
        
        for i, (points, lineup, salary, _, _, _) in enumerate(lineup_data):
            positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DEF': []}
            
            for _, player in lineup.iterrows():
                pos = player['Position']
                player_name = player['Nickname']
                
                if pos == 'D':
                    positions['DEF'].append(player_name)
                elif pos in positions:
                    positions[pos].append(player_name)
            
            # Build Yahoo format (similar to DraftKings)
            if (len(positions['QB']) >= 1 and len(positions['RB']) >= 2 and 
                len(positions['WR']) >= 3 and len(positions['TE']) >= 1 and 
                len(positions['DEF']) >= 1):
                
                row = [
                    positions['QB'][0],
                    positions['RB'][0],
                    positions['RB'][1],
                    positions['WR'][0],
                    positions['WR'][1], 
                    positions['WR'][2],
                    positions['TE'][0],
                    positions['RB'][2] if len(positions['RB']) > 2 else
                    (positions['WR'][3] if len(positions['WR']) > 3 else
                     (positions['TE'][1] if len(positions['TE']) > 1 else '')),
                    positions['DEF'][0]
                ]
                
                if row[7]:
                    csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)
    
    def _export_generic(self, lineup_data: List[Tuple], template: ExportTemplate,
                       contest_info: Optional[Dict] = None) -> str:
        """Generic export format"""
        lines = [','.join(template.required_columns)]
        
        for i, (points, lineup, salary, _, _, _) in enumerate(lineup_data):
            row_data = []
            
            # Add basic lineup info
            row_data.append(f"Lineup_{i+1}")  # Lineup name
            row_data.append(str(points))       # Projected points
            row_data.append(str(salary))       # Total salary
            
            # Add player names by position
            for pos in template.roster_positions:
                pos_players = lineup[lineup['Position'] == pos]['Nickname'].tolist()
                row_data.append(pos_players[0] if pos_players else '')
            
            lines.append(','.join(row_data))
        
        return '\n'.join(lines)
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported platforms"""
        return list(self.templates.keys())
    
    def validate_lineup_for_platform(self, lineup: pd.DataFrame, platform: str) -> Dict[str, Any]:
        """Validate lineup against platform requirements"""
        if platform.lower() not in self.templates:
            return {'valid': False, 'errors': [f'Unsupported platform: {platform}']}
        
        template = self.templates[platform.lower()]
        errors = []
        warnings = []
        
        # Check salary cap
        total_salary = lineup['Salary'].sum()
        if total_salary > template.salary_cap:
            errors.append(f'Salary over cap: ${total_salary:,} > ${template.salary_cap:,}')
        
        # Check roster positions
        position_counts = lineup['Position'].value_counts()
        
        # Check required positions (simplified validation)
        required_positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'D': 1}
        
        for pos, min_count in required_positions.items():
            actual_count = position_counts.get(pos, 0)
            if actual_count < min_count:
                errors.append(f'Not enough {pos}: {actual_count} < {min_count}')
        
        # Check team limits
        team_counts = lineup['Team'].value_counts()
        max_team_count = team_counts.max() if len(team_counts) > 0 else 0
        
        if max_team_count > template.max_players_per_team:
            errors.append(f'Too many players from one team: {max_team_count} > {template.max_players_per_team}')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_salary': total_salary,
            'salary_remaining': template.salary_cap - total_salary
        }

class ExportManager:
    """Manage multiple export formats and batch operations"""
    
    def __init__(self):
        self.exporter = LineupExporter()
        self.export_history = []
    
    def export_to_multiple_platforms(self, lineup_data: List[Tuple], 
                                   platforms: List[str],
                                   contest_info: Optional[Dict] = None,
                                   max_lineups: int = 150) -> Dict[str, str]:
        """Export lineups to multiple platforms simultaneously"""
        exports = {}
        
        for platform in platforms:
            try:
                export_content = self.exporter.export_lineups(
                    lineup_data, platform, contest_info, max_lineups
                )
                exports[platform] = export_content
                
                # Track export
                self.export_history.append({
                    'platform': platform,
                    'timestamp': datetime.now(),
                    'lineup_count': len(lineup_data),
                    'status': 'success'
                })
                
            except Exception as e:
                exports[platform] = f"Export failed: {str(e)}"
                self.export_history.append({
                    'platform': platform,
                    'timestamp': datetime.now(),
                    'lineup_count': 0,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return exports
    
    def get_export_summary(self, lineup_data: List[Tuple]) -> Dict[str, Any]:
        """Get summary of lineup data for export"""
        if not lineup_data:
            return {'total_lineups': 0, 'error': 'No lineup data'}
        
        points_list = [lineup[0] for lineup in lineup_data]
        salary_list = [lineup[2] for lineup in lineup_data]
        
        return {
            'total_lineups': len(lineup_data),
            'avg_projected_points': np.mean(points_list),
            'avg_salary': np.mean(salary_list),
            'min_salary': np.min(salary_list),
            'max_salary': np.max(salary_list),
            'salary_range': np.max(salary_list) - np.min(salary_list),
            'top_10_avg_points': np.mean(sorted(points_list, reverse=True)[:10])
        }