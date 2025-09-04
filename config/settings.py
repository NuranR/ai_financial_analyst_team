"""Configuration settings for A-FIN system."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API Keys (optional during development)
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    # openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")  # Kept for future use
    # anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")  # Kept for future use
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    news_relevance_threshold: float = 0.25

    # Model Configuration
    default_llm_model: str = Field("gemini-2.0-flash-exp", env="DEFAULT_LLM_MODEL")
    temperature: float = Field(0.1, env="TEMPERATURE")
    max_tokens: int = Field(4000, env="MAX_TOKENS")

    # Data Sources
    alpha_vantage_base_url: str = Field(
        "https://www.alphavantage.co/query", env="ALPHA_VANTAGE_BASE_URL"
    )
    news_api_base_url: str = Field("https://newsapi.org/v2", env="NEWS_API_BASE_URL")
    sec_edgar_base_url: str = Field(
        "https://www.sec.gov/Archives/edgar", env="SEC_EDGAR_BASE_URL"
    )

    # Application Settings
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_articles_per_search: int = Field(50, env="MAX_ARTICLES_PER_SEARCH")
    stock_data_period: str = Field("1y", env="STOCK_DATA_PERIOD")

    # MLflow Configuration
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        "afin_anomaly_detection", env="MLFLOW_EXPERIMENT_NAME"
    )

    # Database Configuration
    database_url: str = Field("sqlite:///afin.db", env="DATABASE_URL")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


# Global settings instance
settings = Settings()
