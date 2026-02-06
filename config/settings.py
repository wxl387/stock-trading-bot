"""
Application settings and configuration management.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


class Settings:
    """Application settings loaded from environment and config files."""

    # Trading Mode
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")

    # WeBull Credentials
    WEBULL_EMAIL: str = os.getenv("WEBULL_EMAIL", "")
    WEBULL_PASSWORD: str = os.getenv("WEBULL_PASSWORD", "")
    WEBULL_TRADE_PIN: str = os.getenv("WEBULL_TRADE_PIN", "")
    WEBULL_DEVICE_ID: str = os.getenv("WEBULL_DEVICE_ID", "")

    # Notification Settings
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # Data Sources
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def load_trading_config(cls) -> dict:
        """Load trading configuration from YAML file."""
        config_path = CONFIG_DIR / "trading_config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to parse {config_path}: {e}")
                return {}
        return {}

    @classmethod
    def is_paper_trading(cls) -> bool:
        """Check if running in paper trading mode."""
        return cls.TRADING_MODE.lower() == "paper"


def setup_logging():
    """Configure application logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, Settings.LOG_LEVEL.upper(), logging.INFO)

    # Create logs directory if needed
    LOGS_DIR.mkdir(exist_ok=True)

    # Configure root logger with rotating file handler (10MB max, 5 backups)
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                LOGS_DIR / "trading_bot.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
            )
        ]
    )

    return logging.getLogger("trading_bot")


# Initialize logger
logger = setup_logging()
settings = Settings()
