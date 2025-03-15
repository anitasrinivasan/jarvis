import os
import shutil
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Maintenance:
    def __init__(self, temp_dir="temp", logs_dir="logs"):
        self.temp_dir = temp_dir
        self.logs_dir = logs_dir
        
    def cleanup_temp_files(self, max_age_days=1):
        """Clean up temporary files older than max_age_days"""
        try:
            if not os.path.exists(self.temp_dir):
                return
                
            now = datetime.now()
            count = 0
            
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if not os.path.isfile(filepath):
                    continue
                    
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - file_modified > timedelta(days=max_age_days):
                    os.remove(filepath)
                    count += 1
                    
            logger.info(f"Cleaned up {count} temporary files")
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
            
    def rotate_logs(self, max_logs=7):
        """Rotate log files, keeping only the most recent ones"""
        try:
            if not os.path.exists(self.logs_dir):
                return
                
            log_files = []
            for filename in os.listdir(self.logs_dir):
                if filename.startswith("app_") and filename.endswith(".log"):
                    filepath = os.path.join(self.logs_dir, filename)
                    log_files.append((filepath, os.path.getmtime(filepath)))
                    
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess log files
            for filepath, _ in log_files[max_logs:]:
                os.remove(filepath)
                logger.info(f"Removed old log file: {filepath}")
                
        except Exception as e:
            logger.error(f"Error rotating logs: {str(e)}")
            
    def clear_all_temp(self):
        """Clear all temporary files (use with caution)"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir)
                logger.info("Cleared all temporary files")
        except Exception as e:
            logger.error(f"Error clearing temp directory: {str(e)}")
            
    def run_maintenance(self):
        """Run all maintenance tasks"""
        self.cleanup_temp_files()
        self.rotate_logs()
        logger.info("Completed maintenance tasks") 