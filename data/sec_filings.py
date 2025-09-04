"""SEC filing data fetcher for regulatory analysis."""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger


class SECFilingsFetcher:
    """Fetcher for SEC EDGAR database filings."""

    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": "A-FIN Academic Project contact@university.edu",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    def get_company_filings(
        self, ticker: str, filing_types: List[str] = None, limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get recent SEC filings for a company.

        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types to fetch (e.g., ['10-K', '10-Q'])
            limit: Maximum number of filings to return

        Returns:
            Dictionary containing filing information
        """
        try:
            if filing_types is None:
                filing_types = ["10-K", "10-Q", "8-K"]

            logger.info(f"Fetching SEC filings for {ticker}")

            # Get CIK (Central Index Key) for the ticker
            cik = self._get_cik_from_ticker(ticker)
            if not cik:
                return self._get_mock_filing_data(ticker)

            # Fetch filings data
            url = f"{self.base_url}/submissions/CIK{cik:010d}.json"
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                logger.warning(
                    f"Could not fetch SEC data for {ticker}, using mock data"
                )
                return self._get_mock_filing_data(ticker)

            data = response.json()

            # Extract recent filings
            recent_filings = []
            filings = data.get("filings", {}).get("recent", {})

            if filings:
                keys_to_check = [
                    "form",
                    "filingDate",
                    "accessionNumber",
                    "primaryDocument",
                ]
                min_len = min(len(filings.get(k, [])) for k in keys_to_check)

                for i in range(min(min_len, 1000)):
                    form = filings["form"][i]
                    if form in filing_types:
                        filing_date = filings["filingDate"][i]
                        filing_url = (
                            f"https://www.sec.gov/Archives/edgar/data/"
                            f"{cik}/{filings['accessionNumber'][i].replace('-', '')}/"
                            f"{filings['primaryDocument'][i]}"
                        )

                        recent_filings.append(
                            {
                                "form_type": form,
                                "filing_date": filing_date,
                                "description": (
                                    filings.get("description", [""] * min_len)[i]
                                ),
                                "url": filing_url,
                                "accession_number": filings["accessionNumber"][i],
                            }
                        )
                        if len(recent_filings) >= limit:
                            break

            company_info = {
                "name": data.get("name", ticker),
                "cik": str(cik),
                "sic": data.get("sic", ""),
                "sicDescription": data.get("sicDescription", ""),
                "category": data.get("category", ""),
                "entityType": data.get("entityType", ""),
            }

            return {
                "ticker": ticker,
                "company_info": company_info,
                "filings": recent_filings,
                "total_filings": len(recent_filings),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {str(e)}")
            return self._get_mock_filing_data(ticker)

    def _get_cik_from_ticker(self, ticker: str) -> Optional[int]:
        """Get CIK number from ticker symbol using SEC's official mapping."""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {"User-Agent": "Contact@Email.com"}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                for entry in data.values():
                    if entry["ticker"].upper() == ticker.upper():
                        return int(entry["cik_str"])

            logger.warning(
                f"CIK not found for ticker {ticker}, using fallback mapping."
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
        # Fallback mapping
        mock_ticker_to_cik = {
            "AAPL": 320193,
            "MSFT": 789019,
            "GOOGL": 1652044,
            "GOOG": 1652044,
            "TSLA": 1318605,
            "AMZN": 1018724,
            "META": 1326801,
            "NFLX": 1065280,
            "NVDA": 1045810,
            "AMD": 2488,
        }
        return mock_ticker_to_cik.get(ticker.upper())

    def _get_mock_filing_data(self, ticker: str) -> Dict[str, Any]:
        """Return mock SEC filing data for demonstration."""
        return {
            "ticker": ticker,
            "company_info": {
                "name": f"{ticker} Corporation",
                "cik": "1234567",
                "sic": "3571",
                "sicDescription": "Electronic Computers",
                "category": "Large accelerated filer",
                "entityType": "corporation",
            },
            "filings": [
                {
                    "form_type": "10-K",
                    "filing_date": "2024-12-31",
                    "description": "Annual Report (10-K)",
                    "url": "https://example.com/10k",
                    "accession_number": "0001234567-24-000001",
                },
                {
                    "form_type": "10-Q",
                    "filing_date": "2024-09-30",
                    "description": "Quarterly Report (10-Q)",
                    "url": "https://example.com/10q",
                    "accession_number": "0001234567-24-000002",
                },
                {
                    "form_type": "8-K",
                    "filing_date": "2024-07-15",
                    "description": "Current Report (8-K)",
                    "url": "https://example.com/8k",
                    "accession_number": "0001234567-24-000003",
                },
            ],
            "total_filings": 3,
            "last_updated": datetime.now().isoformat(),
            "note": "Mock data for demonstration purposes",
        }

    def extract_filing_insights(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from filing data."""
        try:
            filings = filing_data.get("filings", [])

            # Analyze filing patterns
            filing_frequency = {}
            recent_forms = []

            for filing in filings:
                form_type = filing["form_type"]
                filing_frequency[form_type] = filing_frequency.get(form_type, 0) + 1
                recent_forms.append(form_type)

            # Risk assessment based on filing types
            risk_signals = []
            if "8-K" in recent_forms[:3]:  # Recent 8-K filing
                risk_signals.append("Recent material event disclosure (8-K)")

            ten_k_filings = [f for f in filings if f["form_type"] == "10-K"]
            if ten_k_filings:
                latest_10k = max(
                    ten_k_filings,
                    key=lambda f: datetime.fromisoformat(f["filing_date"]),
                )
                days_since_10k = (
                    datetime.now() - datetime.fromisoformat(latest_10k["filing_date"])
                ).days
                if days_since_10k > 365:
                    risk_signals.append(
                        "No recent 10-K (annual report older than 12 months)"
                    )

            # Compliance assessment
            latest_filing = filings[0] if filings else None
            compliance_status = "Current" if latest_filing else "Unknown"

            if latest_filing:
                filing_date = datetime.fromisoformat(latest_filing["filing_date"])
                days_old = (datetime.now() - filing_date).days
                if days_old > 120:  # Older than 4 months
                    compliance_status = "Potentially outdated"

            return {
                "filing_summary": {
                    "total_filings": len(filings),
                    "filing_types": list(filing_frequency.keys()),
                    "latest_filing": (
                        latest_filing["form_type"] if latest_filing else None
                    ),
                    "latest_filing_date": (
                        latest_filing["filing_date"] if latest_filing else None
                    ),
                },
                "compliance_assessment": {
                    "status": compliance_status,
                    "risk_signals": risk_signals,
                    "filing_frequency": filing_frequency,
                },
                "regulatory_health": self._assess_regulatory_health(
                    filing_frequency, risk_signals
                ),
            }

        except Exception as e:
            logger.error(f"Error extracting filing insights: {str(e)}")
            return {"error": str(e)}

    def _assess_regulatory_health(
        self, filing_frequency: Dict[str, int], risk_signals: List[str]
    ) -> str:
        """Assess overall regulatory health."""
        score = 0

        # Positive indicators
        if filing_frequency.get("10-K", 0) > 0:
            score += 2
        if filing_frequency.get("10-Q", 0) > 0:
            score += 1

        # Negative indicators
        score -= len(risk_signals)

        if score >= 2:
            return "Good"
        elif score >= 0:
            return "Fair"
        else:
            return "Concerning"
