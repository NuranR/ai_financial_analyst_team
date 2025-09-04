"""Quantitative Analyst Agent for statistical analysis and anomaly detection."""

from typing import Dict, Any
from datetime import datetime
import numpy as np
from loguru import logger

from agents.base_agent import BaseAgent, AgentResult
from data.stock_data import StockDataFetcher
from config.prompts import QUANT_ANALYST_SYSTEM_PROMPT, QUANT_ANALYST_ANALYSIS_PROMPT
from config.settings import settings


class QuantitativeAnalystAgent(BaseAgent):
    """
    Agent specializing in quantitative analysis of stock data.

    Responsibilities:
    - Analyze historical price and volume data
    - Detect statistical anomalies and unusual patterns
    - Calculate technical indicators and risk metrics
    - Identify potential trading signals
    """

    def __init__(self, **kwargs):
        super().__init__("Quantitative Analyst")
        self.stock_fetcher = StockDataFetcher()

    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        """
        Perform quantitative analysis on stock data.

        Args:
            ticker: Stock ticker symbol
            period: Analysis period (default: 1y)
            include_technical: Whether to include technical analysis (default: True)

        Returns:
            AgentResult with quantitative analysis
        """
        try:
            # Validate inputs
            ticker = self._validate_ticker(ticker)
            period = kwargs.get("period", settings.stock_data_period)
            include_technical = kwargs.get("include_technical", True)

            logger.info(
                f"Starting quantitative analysis for {ticker} (period: {period})"
            )

            # Fetch stock data
            stock_data = self.stock_fetcher.get_stock_data(ticker, period=period)

            # Detect anomalies
            anomalies = self.stock_fetcher.detect_price_anomalies(
                stock_data["price_data"], stock_data["volume_data"]
            )

            # Calculate additional metrics
            technical_metrics = (
                self._calculate_technical_metrics(stock_data)
                if include_technical
                else {}
            )

            formatted_data, structured_output_components = (
                self._format_data_for_analysis(stock_data, anomalies, technical_metrics)
            )

            analysis_prompt = QUANT_ANALYST_ANALYSIS_PROMPT.format(
                company_name=stock_data["company_info"]["name"],
                ticker=ticker,
                price_data=formatted_data["price_summary"],
                volume_data=formatted_data["volume_summary"],
                anomaly_results=formatted_data["anomaly_summary"],
                technical_indicators=formatted_data["tech_summary"],
            )

            llm_analysis_text = self._call_llm(
                analysis_prompt, QUANT_ANALYST_SYSTEM_PROMPT
            )

            # Calculate confidence based on data quality and completeness
            confidence = self._calculate_confidence_score(
                llm_analysis_text,
                data_quality=self._assess_data_quality(stock_data),
                anomaly_significance=self._assess_anomaly_significance(anomalies),
                data_completeness=stock_data["data_points"]
                / 252,  # Assume 252 trading days per year
            )

            metadata = {
                "analysis_period": period,
                "data_points": stock_data["data_points"],
                "current_price": stock_data["current_price"],
                "volatility": stock_data["volatility"],
                "price_anomalies": len(anomalies.get("price_anomalies", [])),
                "volume_anomalies": len(anomalies.get("volume_anomalies", [])),
                "technical_indicators": (
                    list(technical_metrics.keys()) if technical_metrics else []
                ),
                "analysis_timestamp": datetime.now().isoformat(),
                "structured_analysis": structured_output_components,
                "llm_raw_analysis": llm_analysis_text,
            }

            logger.info(f"Completed quantitative analysis for {ticker}")

            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=llm_analysis_text,
                confidence_score=confidence,
                metadata=metadata,
                structured_data=structured_output_components,
            )

        except Exception as e:
            logger.error(f"Error in quantitative analysis for {ticker}: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="",
                confidence_score=0.0,
                errors=str(e),
            )

    def _calculate_technical_metrics(
        self, stock_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate technical indicators and metrics."""
        try:
            closes = np.array(stock_data["price_data"]["close"])
            highs = np.array(stock_data["price_data"]["high"])
            lows = np.array(stock_data["price_data"]["low"])
            volumes = (
                np.array(stock_data["volume_data"]["volume"])
                if stock_data["volume_data"]
                else None
            )

            metrics = {}

            # Simple Moving Averages
            if len(closes) >= 50:
                metrics["sma_20"] = round(np.mean(closes[-20:]), 2)
                metrics["sma_50"] = round(np.mean(closes[-50:]), 2)

                # Price vs MA signals
                current_price = closes[-1]
                metrics["price_vs_sma20"] = (
                    "above" if current_price > metrics["sma_20"] else "below"
                )
                metrics["price_vs_sma50"] = (
                    "above" if current_price > metrics["sma_50"] else "below"
                )

            # RSI calculation (simplified)
            if len(closes) >= 14:
                metrics["rsi"] = round(self._calculate_rsi(closes), 2)
                if metrics["rsi"] > 70:
                    metrics["rsi_signal"] = "overbought"
                elif metrics["rsi"] < 30:
                    metrics["rsi_signal"] = "oversold"
                else:
                    metrics["rsi_signal"] = "neutral"

            # Support and Resistance levels
            if len(closes) >= 20:
                metrics["support_level"] = round(
                    np.min(closes[-min(len(closes), 60) :]), 2
                )  # e.g., last 60 days
                metrics["resistance_level"] = round(
                    np.max(closes[-min(len(closes), 60) :]), 2
                )  # e.g., last 60 days

            # Price momentum
            if len(closes) >= 10:
                momentum = (closes[-1] - closes[-10]) / closes[-10] * 100
                metrics["10d_momentum"] = round(momentum, 2)
                metrics["momentum_signal"] = (
                    "bullish"
                    if momentum > 5
                    else "bearish" if momentum < -5 else "neutral"
                )

            if volumes is not None and len(volumes) >= 20:
                avg_volume = np.mean(volumes[-20:])
                recent_volume = volumes[-1]
                metrics["volume_ratio"] = round(recent_volume / avg_volume, 2)
                metrics["volume_signal"] = (
                    "high"
                    if metrics["volume_ratio"] > 1.5
                    else "low" if metrics["volume_ratio"] < 0.5 else "normal"
                )

            if "price_change_pct_1d" in stock_data:
                if stock_data["price_change_pct_1d"] > 0.5:
                    metrics["current_trend_short"] = "Uptrending"
                elif stock_data["price_change_pct_1d"] < -0.5:
                    metrics["current_trend_short"] = "Downtrending"
                else:
                    metrics["current_trend_short"] = "Sideways"

            return metrics

        except Exception as e:
            logger.error(f"Error calculating technical metrics: {str(e)}")
            return {}

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(closes) < period + 1:
            return np.nan

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _format_data_for_analysis(
        self,
        stock_data: Dict[str, Any],
        anomalies: Dict[str, Any],
        technical_metrics: Dict[str, Any],
    ) -> (Dict[str, str], Dict[str, Any]):
        """
        Format data for LLM analysis and also provide structured components
        for direct display in the UI.
        """
        structured_components = {
            "price_analysis": {},
            "technical_signals": {},
            "anomalies_risks": {},
            "bottom_line": "",
        }

        price_summary_text = f"""
Current Price: ${stock_data['current_price']:.2f}
1-Day Change: ${stock_data['price_change_1d']:.2f} ({stock_data['price_change_pct_1d']:.1f}%)
Period High: ${stock_data['period_high']:.2f}
Period Low: ${stock_data['period_low']:.2f}
Period Return: {stock_data['period_return_pct']:.1f}%
Volatility: {stock_data['volatility']:.1%}
Current Trend: {technical_metrics.get('current_trend_short', 'N/A')}, below Period High, recent 1-day drop of {stock_data['price_change_pct_1d']:.1f}%.
Key levels: Resistance at ${technical_metrics.get('resistance_level', 'N/A')}, support near ${technical_metrics.get('support_level', 'N/A')}.
"""
        structured_components["price_analysis"] = {
            "current_price": stock_data["current_price"],
            "price_change_1d_pct": stock_data["price_change_pct_1d"],
            "period_high": stock_data["period_high"],
            "period_low": stock_data["period_low"],
            "volatility": stock_data["volatility"],
            "current_trend": technical_metrics.get("current_trend_short", "N/A"),
            "resistance": technical_metrics.get("resistance_level", "N/A"),
            "support": technical_metrics.get("support_level", "N/A"),
        }

        volume_summary_text = ""
        if stock_data["volume_data"]:
            volume_summary_text = f"""
Average Volume: {stock_data['volume_data']['avg_volume']:,.0f}
Volume Trend: {stock_data['volume_data']['volume_trend']}
Latest Volume: {stock_data['volume_data']['volume'][-1]:,.0f}
"""

        anomaly_summary_text = f"""
Price Anomalies Detected: {len(anomalies.get('price_anomalies', []))}
Volume Anomalies Detected: {len(anomalies.get('volume_anomalies', []))}
Volatility Assessment: {anomalies.get('volatility_assessment', 'unknown')}

Recent Price Anomalies:
"""
        recent_price_anomalies_list = []
        for anomaly in anomalies.get("price_anomalies", [])[-3:]:  # Last 3
            anomaly_summary_text += f"- {anomaly['date']}: {anomaly['type']} of {anomaly['return']:.1f}% (severity: {anomaly['severity']:.1f})\n"
            recent_price_anomalies_list.append(
                f"{anomaly['date']}: {anomaly['type']} of {anomaly['return']:.1f}%"
            )

        recent_volume_anomalies_list = []
        if anomalies.get("volume_anomalies"):
            anomaly_summary_text += "\nRecent Volume Anomalies:\n"
            for anomaly in anomalies.get("volume_anomalies", [])[-3:]:  # Last 3
                anomaly_summary_text += f"- {anomaly['date']}: {anomaly['vs_average']:.1f}x average volume\n"
                recent_volume_anomalies_list.append(
                    f"{anomaly['date']}: {anomaly['vs_average']:.1f}x avg volume"
                )

        structured_components["anomalies_risks"] = {
            "price_anomalies_count": len(anomalies.get("price_anomalies", [])),
            "volume_anomalies_count": len(anomalies.get("volume_anomalies", [])),
            "volatility_assessment": anomalies.get("volatility_assessment", "unknown"),
            "recent_price_anomalies": recent_price_anomalies_list,
            "recent_volume_anomalies": recent_volume_anomalies_list,
            "risk_factors": "High volatility, recent price drops, and volume spikes warrant caution.",  # General risk statement
        }

        tech_summary_text = ""
        if technical_metrics:
            tech_summary_text = f"""
Technical Indicators:
- RSI: {technical_metrics.get('rsi', 'N/A')} ({technical_metrics.get('rsi_signal', 'N/A')})
- Price vs SMA20: {technical_metrics.get('price_vs_sma20', 'N/A')}
- 10-day Momentum: {technical_metrics.get('10d_momentum', 'N/A')}% ({technical_metrics.get('momentum_signal', 'N/A')})
- Volume Signal: {technical_metrics.get('volume_signal', 'N/A')}
"""
            structured_components["technical_signals"] = {
                "rsi": technical_metrics.get("rsi", "N/A"),
                "rsi_signal": technical_metrics.get("rsi_signal", "N/A"),
                "price_vs_sma20": technical_metrics.get("price_vs_sma20", "N/A"),
                "10d_momentum": technical_metrics.get("10d_momentum", "N/A"),
                "momentum_signal": technical_metrics.get("momentum_signal", "N/A"),
                "volume_signal": technical_metrics.get("volume_signal", "N/A"),
            }

        structured_components["bottom_line"] = (
            f"{stock_data['company_info']['name'] or 'The stock'} exhibits high volatility and recent negative "
            f"price action coupled with high volume, suggesting potential for further downside. "
            f"Monitoring volume and price action around key support levels is crucial."
        )

        llm_formatted_data = {
            "price_summary": price_summary_text,
            "volume_summary": volume_summary_text,
            "anomaly_summary": anomaly_summary_text,
            "tech_summary": tech_summary_text,
        }

        return llm_formatted_data, structured_components

    def _assess_data_quality(self, stock_data: Dict[str, Any]) -> float:
        """Assess the quality of the stock data."""
        score = 0.5  # Base score

        # More data points = higher quality
        if stock_data["data_points"] > 200:
            score += 0.2
        elif stock_data["data_points"] > 100:
            score += 0.1

        # Recent data = higher quality
        try:
            last_date = datetime.fromisoformat(stock_data["last_updated"])
            hours_old = (datetime.now() - last_date).total_seconds() / 3600
            if hours_old < 24:
                score += 0.2
            elif hours_old < 72:
                score += 0.1
        except:
            pass

        # Volume data available = higher quality
        if stock_data["volume_data"] and stock_data["volume_data"]["volume"]:
            score += 0.1

        return min(score, 1.0)

    def _assess_anomaly_significance(self, anomalies: Dict[str, Any]) -> float:
        """Assess the significance of detected anomalies."""
        if not anomalies or "price_anomalies" not in anomalies:
            return 0.5

        price_anomalies = anomalies["price_anomalies"]
        volume_anomalies = anomalies.get("volume_anomalies", [])

        # More recent anomalies = higher significance
        recent_anomalies = sum(
            1
            for a in price_anomalies
            if (datetime.now() - datetime.fromisoformat(a["date"])).days <= 7
        )

        significance = min(0.1 * (recent_anomalies + len(volume_anomalies)), 0.5)
        return significance
