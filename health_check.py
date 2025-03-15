import logging
from datetime import datetime
import time
from typing import Dict

logger = logging.getLogger(__name__)

class HealthCheck:
    def __init__(self, llm, supabase):
        self.llm = llm
        self.supabase = supabase
        self.last_check = None
        self.status = {
            "openai": "unknown",
            "supabase": "unknown",
            "timestamp": None
        }

    def check_services(self) -> Dict[str, str]:
        """Check health of external API dependencies"""
        try:
            # Check OpenAI
            self.llm.predict("test")
            self.status["openai"] = "healthy"
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            self.status["openai"] = "unhealthy"

        try:
            # Check Supabase
            self.supabase.table("documents").select("count", count="exact").execute()
            self.status["supabase"] = "healthy"
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            self.status["supabase"] = "unhealthy"

        self.status["timestamp"] = datetime.now().isoformat()
        self.last_check = time.time()
        return self.status

    def should_check(self, interval_seconds: int = 300) -> bool:
        """Determine if it's time to run another health check"""
        if self.last_check is None:
            return True
        return time.time() - self.last_check >= interval_seconds

    def get_status(self, force_check: bool = False) -> Dict[str, str]:
        """Get current health status"""
        if force_check or self.should_check():
            return self.check_services()
        return self.status

class MetricsLogger:
    def __init__(self, supabase, openai_limiter, supabase_limiter):
        self.supabase = supabase
        self.openai_limiter = openai_limiter
        self.supabase_limiter = supabase_limiter
        self.last_log = None

    def log_metrics(self):
        """Log application metrics"""
        try:
            # Document metrics
            doc_count = len(self.supabase.table("documents").select("id").execute().data)
            logger.info(f"Total documents: {doc_count}")
            
            # User metrics
            user_count = len(self.supabase.table("user_profiles").select("id").execute().data)
            logger.info(f"Total users: {user_count}")
            
            # API usage metrics
            logger.info(f"OpenAI requests queue: {self.openai_limiter.requests.qsize()}")
            logger.info(f"Supabase requests queue: {self.supabase_limiter.requests.qsize()}")
            
            self.last_log = time.time()
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")

    def should_log(self, interval_seconds: int = 300) -> bool:
        """Determine if it's time to log metrics"""
        if self.last_log is None:
            return True
        return time.time() - self.last_log >= interval_seconds 