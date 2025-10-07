import os
from typing import Optional

# basic config for the AI backend
class Config:

    # ollama settings
    OLLAMA_URL: str = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'llama3:latest')  # default model
    OLLAMA_TIMEOUT: int = int(os.getenv('OLLAMA_TIMEOUT', '60'))  # 60 sec timeout

    # server config
    AI_BACKEND_PORT: int = int(os.getenv('AI_BACKEND_PORT', '5247'))  # our port
    AI_BACKEND_HOST: str = os.getenv('AI_BACKEND_HOST', 'localhost')
    DEBUG: bool = os.getenv('AI_DEBUG', 'false').lower() == 'true'  # dev mode

    # TODO: make CORS more restrictive for production
    CORS_ORIGINS: list = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')

    STREAMLIT_BASE_PORT: int = int(os.getenv('STREAMLIT_BASE_PORT', '8501'))
    MAX_DASHBOARD_SIZE: str = os.getenv('MAX_DASHBOARD_SIZE', '10MB')
    DASHBOARD_TIMEOUT: int = int(os.getenv('DASHBOARD_TIMEOUT', '30000'))
    DEFAULT_CHART_TYPE: str = os.getenv('DEFAULT_CHART_TYPE', 'auto')

    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DASHBOARD_DIR: str = os.path.join(PROJECT_ROOT, 'generated-dashboards')
    DATA_DIR: str = os.path.join(PROJECT_ROOT, 'data')
    LOGS_DIR: str = os.path.join(PROJECT_ROOT, 'logs')

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'info').upper()

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        for directory in [cls.DASHBOARD_DIR, cls.DATA_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def validate_ollama_connection(cls) -> dict:
        """Validate Ollama connection"""
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                return {
                    'connected': True,
                    'models': model_names,
                    'has_required_model': cls.OLLAMA_MODEL in model_names
                }
            else:
                return {'connected': False, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    @classmethod
    def get_status(cls) -> dict:
        """Get system status"""
        ollama_status = cls.validate_ollama_connection()
        return {
            'ollama': ollama_status,
            'directories': {
                'dashboard_dir': os.path.exists(cls.DASHBOARD_DIR),
                'data_dir': os.path.exists(cls.DATA_DIR),
                'logs_dir': os.path.exists(cls.LOGS_DIR)
            },
            'config': {
                'port': cls.AI_BACKEND_PORT,
                'debug': cls.DEBUG,
                'model': cls.OLLAMA_MODEL
            }
        }