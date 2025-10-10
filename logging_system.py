"""
Comprehensive logging system for DFS optimizer
Provides debugging, performance monitoring, and user activity tracking
"""
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import traceback
from functools import wraps
import time
import psutil

class DFSLogger:
    """Enhanced logging system for DFS optimizer"""
    
    def __init__(self, log_dir: str = "logs", app_name: str = "dfs_optimizer"):
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.main_logger = self._setup_main_logger()
        self.performance_logger = self._setup_performance_logger()
        self.error_logger = self._setup_error_logger()
        self.user_activity_logger = self._setup_user_activity_logger()
        
        # Performance tracking
        self.performance_metrics = {}
        self.session_start_time = datetime.now()
        
        # Error tracking
        self.error_count = 0
        self.last_errors = []
        
    def _setup_main_logger(self) -> logging.Logger:
        """Setup main application logger"""
        logger = logging.getLogger(f"{self.app_name}_main")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.app_name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance monitoring logger"""
        logger = logging.getLogger(f"{self.app_name}_performance")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        handler = logging.FileHandler(
            self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s | PERF | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_error_logger(self) -> logging.Logger:
        """Setup error tracking logger"""
        logger = logging.getLogger(f"{self.app_name}_errors")
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()
        
        handler = logging.FileHandler(
            self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s | ERROR | %(funcName)s:%(lineno)d | %(message)s\n%(exc_info)s\n'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_user_activity_logger(self) -> logging.Logger:
        """Setup user activity tracking logger"""
        logger = logging.getLogger(f"{self.app_name}_activity")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        handler = logging.FileHandler(
            self.log_dir / f"user_activity_{datetime.now().strftime('%Y%m%d')}.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s | ACTIVITY | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def info(self, message: str, context: Optional[Dict] = None):
        """Log info message with optional context"""
        if context:
            message = f"{message} | Context: {json.dumps(context, default=str)}"
        self.main_logger.info(message)
    
    def warning(self, message: str, context: Optional[Dict] = None):
        """Log warning message"""
        if context:
            message = f"{message} | Context: {json.dumps(context, default=str)}"
        self.main_logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict] = None):
        """Log error message with optional exception and context"""
        self.error_count += 1
        
        error_data = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        if exception:
            error_data['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        self.last_errors.append(error_data)
        
        # Keep only last 10 errors in memory
        if len(self.last_errors) > 10:
            self.last_errors = self.last_errors[-10:]
        
        # Log to error file
        log_message = f"{message}"
        if context:
            log_message += f" | Context: {json.dumps(context, default=str)}"
        if exception:
            log_message += f" | Exception: {str(exception)}"
        
        self.error_logger.error(log_message, exc_info=exception)
        self.main_logger.error(log_message)
    
    def log_performance(self, operation: str, duration: float, 
                       details: Optional[Dict] = None):
        """Log performance metrics"""
        metric_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        # Store in memory for analysis
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(metric_data)
        
        # Log to file
        message = f"{operation} | Duration: {duration:.3f}s"
        if details:
            message += f" | Details: {json.dumps(details, default=str)}"
        
        self.performance_logger.info(message)
    
    def log_user_activity(self, activity: str, details: Optional[Dict] = None):
        """Log user activity"""
        activity_data = {
            'activity': activity,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        message = f"{activity}"
        if details:
            message += f" | {json.dumps(details, default=str)}"
        
        self.user_activity_logger.info(message)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, metrics in self.performance_metrics.items():
            durations = [m['duration_seconds'] for m in metrics]
            summary[operation] = {
                'count': len(metrics),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }
        
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            'total_errors': self.error_count,
            'recent_errors': self.last_errors,
            'session_duration_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
            'error_rate_per_hour': self.error_count / max(1, (datetime.now() - self.session_start_time).total_seconds() / 3600)
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    self.info(f"Deleted old log file: {log_file.name}")
                except Exception as e:
                    self.warning(f"Failed to delete log file {log_file.name}: {str(e)}")

class PerformanceTracker:
    """Track performance metrics for functions and operations"""
    
    def __init__(self, logger: DFSLogger):
        self.logger = logger
        self.active_operations = {}
    
    def track_function(self, operation_name: str = None):
        """Decorator to track function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Track system resources before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                cpu_before = process.cpu_percent()
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Calculate metrics
                    duration = time.time() - start_time
                    memory_after = process.memory_info().rss / 1024 / 1024
                    cpu_after = process.cpu_percent()
                    
                    details = {
                        'memory_before_mb': memory_before,
                        'memory_after_mb': memory_after,
                        'memory_increase_mb': memory_after - memory_before,
                        'cpu_before_percent': cpu_before,
                        'cpu_after_percent': cpu_after,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs),
                        'success': True
                    }
                    
                    self.logger.log_performance(op_name, duration, details)
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    details = {
                        'memory_before_mb': memory_before,
                        'error': str(e),
                        'success': False
                    }
                    
                    self.logger.log_performance(op_name, duration, details)
                    self.logger.error(f"Performance tracked function failed: {op_name}", e)
                    raise
            
            return wrapper
        return decorator
    
    def start_operation(self, operation_name: str):
        """Start tracking an operation"""
        self.active_operations[operation_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    def end_operation(self, operation_name: str, details: Optional[Dict] = None):
        """End tracking an operation"""
        if operation_name not in self.active_operations:
            self.logger.warning(f"Attempted to end untracked operation: {operation_name}")
            return
        
        start_data = self.active_operations[operation_name]
        duration = time.time() - start_data['start_time']
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        operation_details = {
            'memory_start_mb': start_data['start_memory'],
            'memory_end_mb': current_memory,
            'memory_increase_mb': current_memory - start_data['start_memory']
        }
        
        if details:
            operation_details.update(details)
        
        self.logger.log_performance(operation_name, duration, operation_details)
        del self.active_operations[operation_name]

class StreamlitLogger:
    """Streamlit-specific logging utilities"""
    
    def __init__(self, logger: DFSLogger):
        self.logger = logger
    
    def log_user_interaction(self, component_type: str, component_name: str, 
                           value: Any = None, session_id: str = None):
        """Log Streamlit user interactions"""
        details = {
            'component_type': component_type,
            'component_name': component_name,
            'session_id': session_id or 'unknown'
        }
        
        if value is not None:
            # Safely log value (avoid logging sensitive data)
            if isinstance(value, (int, float, bool, str)) and len(str(value)) < 100:
                details['value'] = value
            else:
                details['value_type'] = type(value).__name__
                details['value_length'] = len(str(value)) if hasattr(value, '__len__') else 'unknown'
        
        self.logger.log_user_activity(f"streamlit_{component_type}", details)
    
    def log_page_view(self, page_name: str, session_id: str = None):
        """Log page views"""
        self.logger.log_user_activity("page_view", {
            'page': page_name,
            'session_id': session_id or 'unknown'
        })
    
    def log_app_startup(self, config: Optional[Dict] = None):
        """Log application startup"""
        startup_details = {
            'app_version': '2.1.0',
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        if config:
            startup_details['config'] = config
        
        self.logger.log_user_activity("app_startup", startup_details)

# Context managers for logging
class LogContext:
    """Context manager for logging operations"""
    
    def __init__(self, logger: DFSLogger, operation_name: str, 
                 details: Optional[Dict] = None):
        self.logger = logger
        self.operation_name = operation_name
        self.details = details or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}", self.details)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation_name} in {duration:.3f}s")
            self.logger.log_performance(self.operation_name, duration, self.details)
        else:
            self.logger.error(f"Failed operation: {self.operation_name}", exc_val, self.details)
            self.logger.log_performance(f"{self.operation_name}_failed", duration, 
                                      {**self.details, 'error': str(exc_val)})

# Global logger instance
_dfs_logger = None

def get_logger() -> DFSLogger:
    """Get global logger instance"""
    global _dfs_logger
    if _dfs_logger is None:
        _dfs_logger = DFSLogger()
    return _dfs_logger

def init_logging(log_dir: str = "logs", app_name: str = "dfs_optimizer") -> DFSLogger:
    """Initialize logging system"""
    global _dfs_logger
    _dfs_logger = DFSLogger(log_dir, app_name)
    _dfs_logger.info("Logging system initialized")
    return _dfs_logger

# Convenience functions
def log_info(message: str, context: Optional[Dict] = None):
    """Log info message"""
    get_logger().info(message, context)

def log_warning(message: str, context: Optional[Dict] = None):
    """Log warning message"""
    get_logger().warning(message, context)

def log_error(message: str, exception: Optional[Exception] = None, context: Optional[Dict] = None):
    """Log error message"""
    get_logger().error(message, exception, context)

def log_performance(operation: str, duration: float, details: Optional[Dict] = None):
    """Log performance metric"""
    get_logger().log_performance(operation, duration, details)

def log_user_activity(activity: str, details: Optional[Dict] = None):
    """Log user activity"""
    get_logger().log_user_activity(activity, details)

# Decorator shortcuts
def performance_track(operation_name: str = None):
    """Decorator to track function performance"""
    return PerformanceTracker(get_logger()).track_function(operation_name)

def log_operation(operation_name: str, details: Optional[Dict] = None):
    """Context manager for logging operations"""
    return LogContext(get_logger(), operation_name, details)