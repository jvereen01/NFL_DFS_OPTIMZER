"""
Configuration management system for DFS optimizer
Centralized settings, user preferences, and environment configuration
"""
import json
import os
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Mock streamlit functions if not available
    class MockST:
        @staticmethod
        def warning(msg): print(f"WARNING: {msg}")
        @staticmethod
        def error(msg): print(f"ERROR: {msg}")
    st = MockST()

@dataclass
class OptimizationSettings:
    """Core optimization settings"""
    num_simulations: int = 5000  # Reduced from 10000 for faster generation
    stack_probability: float = 0.80
    elite_target_boost: float = 0.45
    great_target_boost: float = 0.25
    wr_boost_multiplier: float = 1.0
    rb_boost_multiplier: float = 1.0
    forced_player_boost: float = 0.3
    num_lineups_display: int = 20
    force_mode: str = "Soft Force (Boost Only)"

@dataclass  
class DataSettings:
    """Data source and file settings"""
    required_csv_file: str = "FanDuel-NFL-2025 EST-11 EST-23 EST-123168-players-listtoday.csv"
    excel_file: str = "NFL.xlsx"
    auto_validate_data: bool = True
    cache_duration_hours: int = 1
    backup_data: bool = False

@dataclass
class UISettings:
    """User interface settings"""
    theme: str = "wide"
    show_debug_info: bool = False
    enable_advanced_analytics: bool = True
    auto_refresh_data: bool = False
    compact_lineup_display: bool = False

@dataclass
class ExportSettings:
    """Export and output settings"""
    default_export_format: str = "fanduel"
    max_export_lineups: int = 150
    include_contest_info: bool = True
    auto_generate_entry_ids: bool = True
    base_entry_id: int = 3584175604
    default_contest_id: str = "121309-276916553"
    default_contest_name: str = "$60K Sun NFL Hail Mary (Only $0.25 to Enter)"
    default_entry_fee: str = "0.25"

@dataclass
class PerformanceSettings:
    """Performance and optimization settings"""
    enable_caching: bool = True
    parallel_processing: bool = False
    memory_optimization: bool = True
    max_memory_usage_mb: int = 1024
    enable_profiling: bool = False

@dataclass
class DFSOptimizerConfig:
    """Complete configuration for DFS optimizer"""
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    data: DataSettings = field(default_factory=DataSettings)
    ui: UISettings = field(default_factory=UISettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    version: str = "2.1.0"
    last_updated: str = ""

class ConfigManager:
    """Manages configuration loading, saving, and validation"""
    
    def __init__(self, config_file: str = "dfs_config.json"):
        self.config_file = Path(config_file)
        self.config = DFSOptimizerConfig()
        self.user_config_file = Path("user_preferences.json")
        
    def load_config(self) -> DFSOptimizerConfig:
        """Load configuration from file or create default"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                    self.config = self._dict_to_config(config_dict)
            else:
                # Create default config file
                self.save_config()
                
        except Exception as e:
            st.warning(f"Could not load config file: {e}. Using defaults.")
            self.config = DFSOptimizerConfig()
            
        # Load user preferences overlay
        self._load_user_preferences()
        return self.config
    
    def save_config(self, config: Optional[DFSOptimizerConfig] = None):
        """Save configuration to file"""
        if config:
            self.config = config
            
        self.config.last_updated = datetime.now().isoformat()
        
        try:
            config_dict = asdict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            st.error(f"Could not save config file: {e}")
    
    def _load_user_preferences(self):
        """Load user-specific preferences"""
        try:
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r') as f:
                    user_prefs = json.load(f)
                    # Apply user preferences to config
                    self._apply_user_preferences(user_prefs)
        except Exception as e:
            # User preferences are optional
            pass
    
    def save_user_preferences(self, preferences: Dict[str, Any]):
        """Save user-specific preferences"""
        try:
            existing_prefs = {}
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r') as f:
                    existing_prefs = json.load(f)
            
            existing_prefs.update(preferences)
            existing_prefs['last_updated'] = datetime.now().isoformat()
            
            with open(self.user_config_file, 'w') as f:
                json.dump(existing_prefs, f, indent=2)
                
        except Exception as e:
            st.warning(f"Could not save user preferences: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> DFSOptimizerConfig:
        """Convert dictionary to config object"""
        config = DFSOptimizerConfig()
        
        if 'optimization' in config_dict:
            config.optimization = OptimizationSettings(**config_dict['optimization'])
        if 'data' in config_dict:
            config.data = DataSettings(**config_dict['data'])
        if 'ui' in config_dict:
            config.ui = UISettings(**config_dict['ui'])
        if 'export' in config_dict:
            config.export = ExportSettings(**config_dict['export'])
        if 'performance' in config_dict:
            config.performance = PerformanceSettings(**config_dict['performance'])
            
        config.version = config_dict.get('version', '2.1.0')
        config.last_updated = config_dict.get('last_updated', '')
        
        return config
    
    def _apply_user_preferences(self, preferences: Dict[str, Any]):
        """Apply user preferences to current config"""
        # Update optimization settings from user preferences
        if 'optimization' in preferences:
            opt_prefs = preferences['optimization']
            for key, value in opt_prefs.items():
                if hasattr(self.config.optimization, key):
                    setattr(self.config.optimization, key, value)
        
        # Update UI settings
        if 'ui' in preferences:
            ui_prefs = preferences['ui']
            for key, value in ui_prefs.items():
                if hasattr(self.config.ui, key):
                    setattr(self.config.ui, key, value)
    
    def get_environment_config(self) -> Dict[str, str]:
        """Get environment-specific configuration"""
        env_config = {
            'DATA_PATH': os.getenv('DFS_DATA_PATH', '.'),
            'CACHE_ENABLED': os.getenv('DFS_CACHE_ENABLED', 'true'),
            'DEBUG_MODE': os.getenv('DFS_DEBUG_MODE', 'false'),
            'MAX_SIMULATIONS': os.getenv('DFS_MAX_SIMULATIONS', '50000'),
            'STREAMLIT_THEME': os.getenv('STREAMLIT_THEME', 'wide')
        }
        return env_config
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Validate optimization settings
        if self.config.optimization.num_simulations < 100:
            issues.append("Number of simulations too low (minimum 100)")
        elif self.config.optimization.num_simulations > 100000:
            issues.append("Number of simulations too high (maximum 100,000)")
            
        if not (0 <= self.config.optimization.stack_probability <= 1):
            issues.append("Stack probability must be between 0 and 1")
            
        # Validate file paths
        data_path = Path(self.config.data.excel_file)
        if not data_path.exists():
            issues.append(f"Excel file not found: {self.config.data.excel_file}")
            
        csv_path = Path(self.config.data.required_csv_file)
        if not csv_path.exists():
            issues.append(f"CSV file not found: {self.config.data.required_csv_file}")
            
        return issues

class ConfigUI:
    """Streamlit UI components for configuration management"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
    
    def render_settings_panel(self) -> Dict[str, Any]:
        """Render settings panel in Streamlit sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Configuration")
        
        with st.sidebar.expander("🎯 Optimization Settings", expanded=False):
            # Load current values from config
            num_simulations = st.slider(
                "Number of Simulations", 
                1000, 50000, 
                self.config.optimization.num_simulations, 
                step=1000,
                key="config_simulations"
            )
            
            stack_probability = st.slider(
                "Stacking Probability", 
                0.0, 1.0, 
                self.config.optimization.stack_probability, 
                step=0.05,
                key="config_stack_prob"
            )
            
            elite_target_boost = st.slider(
                "Elite Target Boost", 
                0.0, 1.0, 
                self.config.optimization.elite_target_boost, 
                step=0.05,
                key="config_elite_boost"
            )
            
            great_target_boost = st.slider(
                "Great Target Boost", 
                0.0, 1.0, 
                self.config.optimization.great_target_boost, 
                step=0.05,
                key="config_great_boost"
            )
        
        with st.sidebar.expander("📊 Display Settings", expanded=False):
            num_lineups_display = st.slider(
                "Number of Top Lineups", 
                5, 50, 
                self.config.optimization.num_lineups_display, 
                step=5,
                key="config_lineups_display"
            )
            
            show_debug = st.checkbox(
                "Show Debug Information", 
                self.config.ui.show_debug_info,
                key="config_debug"
            )
            
            enable_analytics = st.checkbox(
                "Enable Advanced Analytics", 
                self.config.ui.enable_advanced_analytics,
                key="config_analytics"
            )
        
        with st.sidebar.expander("📁 Export Settings", expanded=False):
            max_export = st.slider(
                "Max Export Lineups", 
                10, 500, 
                self.config.export.max_export_lineups, 
                step=10,
                key="config_max_export"
            )
            
            contest_id = st.text_input(
                "Default Contest ID", 
                self.config.export.default_contest_id,
                key="config_contest_id"
            )
            
            entry_fee = st.text_input(
                "Default Entry Fee", 
                self.config.export.default_entry_fee,
                key="config_entry_fee"
            )
        
        # Save settings button
        if st.sidebar.button("💾 Save Settings", key="save_config"):
            # Update config with new values
            self.config.optimization.num_simulations = num_simulations
            self.config.optimization.stack_probability = stack_probability
            self.config.optimization.elite_target_boost = elite_target_boost
            self.config.optimization.great_target_boost = great_target_boost
            self.config.optimization.num_lineups_display = num_lineups_display
            
            self.config.ui.show_debug_info = show_debug
            self.config.ui.enable_advanced_analytics = enable_analytics
            
            self.config.export.max_export_lineups = max_export
            self.config.export.default_contest_id = contest_id
            self.config.export.default_entry_fee = entry_fee
            
            # Save to file
            self.config_manager.save_config(self.config)
            st.sidebar.success("✅ Settings saved!")
        
        # Reset to defaults button
        if st.sidebar.button("🔄 Reset to Defaults", key="reset_config"):
            self.config = DFSOptimizerConfig()
            self.config_manager.save_config(self.config)
            st.sidebar.success("✅ Reset to defaults!")
            st.experimental_rerun()
        
        return asdict(self.config)
    
    def render_config_info(self):
        """Render configuration information"""
        if self.config.ui.show_debug_info:
            with st.expander("🔧 Configuration Details"):
                st.json(asdict(self.config))
                
                # Validation results
                issues = self.config_manager.validate_config()
                if issues:
                    st.warning("Configuration Issues:")
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ Configuration is valid")

# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config() -> DFSOptimizerConfig:
    """Load configuration (convenience function)"""
    return get_config_manager().load_config()

def save_config(config: DFSOptimizerConfig):
    """Save configuration (convenience function)"""
    get_config_manager().save_config(config)

# Environment variable helpers
def get_env_bool(var_name: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    return os.getenv(var_name, str(default)).lower() in ('true', '1', 'yes', 'on')

def get_env_int(var_name: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.getenv(var_name, str(default)))
    except ValueError:
        return default

def get_env_float(var_name: str, default: float = 0.0) -> float:
    """Get float environment variable"""
    try:
        return float(os.getenv(var_name, str(default)))
    except ValueError:
        return default