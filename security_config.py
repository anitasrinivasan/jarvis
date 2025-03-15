import os
from typing import List, Optional
import secrets
import re

class SecurityConfig:
    def __init__(self):
        self.allowed_file_types = {'pdf', 'txt', 'md'}
        self.max_file_size_mb = 10
        self.max_tokens_per_request = 4000
        self.admin_secret = os.getenv('ADMIN_SECRET')
        
    def validate_file(self, filename: str, file_size_bytes: int) -> tuple[bool, str]:
        """Validate file upload"""
        # Check file extension
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if ext not in self.allowed_file_types:
            return False, f"File type not allowed. Allowed types: {', '.join(self.allowed_file_types)}"
            
        # Check file size
        if file_size_bytes > (self.max_file_size_mb * 1024 * 1024):
            return False, f"File too large. Maximum size: {self.max_file_size_mb}MB"
            
        return True, "File valid"

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove any potential script tags or dangerous HTML
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and presence"""
        if not api_key:
            return False
        # Add specific validation for OpenAI API key format
        if api_key.startswith('sk-') and len(api_key) > 20:
            return True
        return False

    def generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)

    def rate_limit_config(self) -> dict:
        """Get rate limiting configuration"""
        return {
            'openai': {
                'max_requests': 50,
                'time_window': 60
            },
            'supabase': {
                'max_requests': 100,
                'time_window': 60
            }
        }

    def validate_user_input(self, name: str, interests: List[str], projects: List[str]) -> tuple[bool, str]:
        """Validate user profile input"""
        if not name or len(name.strip()) < 2:
            return False, "Name must be at least 2 characters long"
        
        if not interests:
            return False, "At least one interest must be specified"
            
        if not projects:
            return False, "At least one project must be specified"
            
        # Sanitize all inputs
        name = self.sanitize_input(name)
        interests = [self.sanitize_input(i) for i in interests]
        projects = [self.sanitize_input(p) for p in projects]
        
        return True, "Input valid" 