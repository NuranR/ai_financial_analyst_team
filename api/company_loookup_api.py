"""Company Lookup API"""

import yfinance as yf
from loguru import logger

class CompanyLookupAPI:
    """Interface for fetching company name from ticker symbol."""
        
    def get_company_name(self, ticker: str) -> str:
        """
        Fetch company name for a given ticker symbol.
        Args:
            ticker: Stock ticker symbol
        Returns:
            Name of the company or empty string if not found
        """
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName")
            logger.info(f"Fetched company name for {ticker}: {company_name}")
            return company_name
        except Exception as e:
            logger.error(f"Error fetching company name for {ticker}: {e}")
            return ""
    
# Fallback for when CompanyLookupAPI is not available
class MockCompanyLookupAPI:
    """Mock company lookup API for development/testing."""
    
    def get_company_name(ticker:str) -> str:
        """Return a mock company name based on ticker."""
        logger.warning(f"Using MockCompanyLookupAPI, returning placeholder company name for ticker {ticker}.")
        return f"Company_{ticker}"