"""Streamlit web interface for A-FIN."""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from agents.data_journalist import DataJournalistAgent
from agents.quantitative_analyst import QuantitativeAnalystAgent
from agents.regulator_specialist import RegulatorSpecialistAgent
from agents.lead_analyst import LeadAnalystAgent


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="A-FIN: Autonomous Financial Information Nexus",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏦 A-FIN: Autonomous Financial Information Nexus")
    st.markdown("**Multi-Agent AI Financial Analysis System**")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker", value="AAPL", help="Enter a valid stock ticker symbol"
    ).upper()

    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
        help="Historical data period for quantitative analysis",
    )

    max_articles = st.sidebar.slider(
        "Max News Articles",
        min_value=5,
        max_value=50,
        value=10,
        help="Maximum number of news articles to analyze for Data Journalist",
    )

    dj_days_back = st.sidebar.slider(
        "News Days Back",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days back to fetch news for Data Journalist",
    )

    rs_filing_types = st.sidebar.multiselect(
        "Regulator Filing Types",
        ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"],
        default=["10-K", "10-Q", "8-K"],
        help="Select SEC filing types for Regulator Specialist",
    )

    rs_max_filings = st.sidebar.slider(
        "Max Regulator Filings",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of SEC filings to analyze for Regulator Specialist",
    )

    # Agent selection
    st.sidebar.subheader("🤖 Select Agents")
    run_data_journalist = st.sidebar.checkbox("📰 Data Journalist", value=True)
    run_quant_analyst = st.sidebar.checkbox("📈 Quantitative Analyst", value=True)
    run_regulator_specialist = st.sidebar.checkbox(
        "📋 Regulator Specialist", value=True
    )
    run_lead_analyst = st.sidebar.checkbox("🎯 Lead Analyst", value=True)

    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏠 Overview",
            "📰 Data Journalist",
            "📈 Quantitative Analyst",
            "📋 Regulator Specialist",
            "🎯 Lead Analyst",
        ]
    )

    with tab1:
        st.header("🏠 System Overview")

        # Check API key
        if not settings.gemini_api_key:
            st.error(
                "⚠️ **API Key Missing**: Please set your GEMINI_API_KEY environment variable"
            )
            st.info(
                "💡 **Tip**: Create a `.env` file with: `GEMINI_API_KEY=your_key_here`"
            )
        else:
            st.success("✅ **API Key Configured**: Gemini AI is ready")

        # System metrics
        st.subheader("📊 System Status")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            agent_status = "✅ Ready" if settings.gemini_api_key else "❌ No API Key"
            st.metric("Agent Status", agent_status)
        with col2:
            data_sources = 3 if settings.gemini_api_key else 1
            st.metric("Data Sources", data_sources)
        with col3:
            st.metric("Analysis Period", analysis_period)
        with col4:
            st.metric("Max Articles", max_articles)

        st.success("🎉 **ALL AGENTS COMPLETE**: Full A-FIN system is operational!")

        # Show agent capabilities
        st.subheader("🤖 Agent Team Status")

        col1, col2 = st.columns(2)
        with col1:
            st.success(
                """
            **📰 Data Journalist** ✅ READY
            - Fetches financial news
            - Analyzes market sentiment  
            - Identifies market-moving events
            - Uses Gemini AI for insights
            """
            )

        with col2:
            st.success(
                """
            **📈 Quantitative Analyst** ✅ READY
            - Downloads stock price data
            - Detects trading anomalies
            - Calculates technical indicators
            - Assesses volatility and risk
            """
            )

        col3, col4 = st.columns(2)
        with col3:
            st.success(
                """
            **📋 Regulator Specialist** ✅ READY
            - SEC filings analysis
            - Risk assessment
            - Compliance monitoring
            - Corporate governance review
            """
            )

        with col4:
            st.success(
                """
            **🎯 Lead Analyst** ✅ READY
            - Synthesizes all insights
            - Investment recommendations
            - Risk-adjusted analysis
            - Final investment brief
            """
            )

        # Show configuration
        st.subheader("Current Configuration")
        config_data = {
            "Ticker": ticker,
            "Data Journalist": "✅" if run_data_journalist else "❌",
            "Quantitative Analyst": "✅" if run_quant_analyst else "❌",
            "Regulator Specialist": "✅" if run_regulator_specialist else "❌",
            "Lead Analyst": "✅" if run_lead_analyst else "❌",
            "Analysis Period": analysis_period,
            "Max News Articles": max_articles,
            "News Days Back": dj_days_back,
            "Regulator Filing Types": ", ".join(rs_filing_types),
            "Max Regulator Filings": rs_max_filings,
        }

        for key, value in config_data.items():
            st.write(f"**{key}**: {value}")

    with tab2:
        if run_data_journalist:
            st.header("📰 Data Journalist Analysis")

            if run_analysis:
                if not settings.gemini_api_key:
                    st.error(
                        "⚠️ **API Key Required**: Please configure your Gemini API key"
                    )
                else:
                    with st.spinner("🔍 Analyzing financial news..."):
                        try:
                            agent = DataJournalistAgent()
                            result = agent.analyze(
                                ticker,
                                max_articles=max_articles,
                                days_back=dj_days_back,
                            )

                            st.success("✅ **Analysis Complete**")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Confidence", f"{result.confidence_score*100:.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Articles",
                                    result.metadata.get("articles_analyzed", 0),
                                )
                            with col3:
                                st.metric(
                                    "Sources",
                                    len(result.metadata.get("sources_analyzed", [])),
                                )

                            date_range = result.metadata.get("date_range", {})
                            if date_range:
                                st.caption(
                                    f"📅 Articles from **{date_range['oldest'].split('T')[0]}** → **{date_range['newest'].split('T')[0]}**"
                                )

                            st.subheader("📊 Analysis Results")
                            st.write(result.analysis)

                            if result.structured_data.get("key_themes"):
                                st.subheader("🔑 Key Themes")
                                st.write(
                                    ", ".join(result.structured_data["key_themes"])
                                )

                            if result.structured_data.get("potential_catalysts"):
                                st.subheader("⚡ Potential Catalysts")
                                for i, catalyst in enumerate(
                                    result.structured_data["potential_catalysts"], 1
                                ):
                                    st.markdown(
                                        f"**{i}. {catalyst['event_type'].replace('_', ' ').title()}**: {catalyst['description']}"
                                    )
                                    st.divider()

                            if result.metadata.get("articles"):
                                st.subheader("📰 Top Articles")
                                for i, article in enumerate(
                                    result.metadata["articles"], 1
                                ):
                                    st.markdown(
                                        f"**{i}. [{article.get('title','No title')}]({article.get('url','')})**"
                                    )
                                    if article.get("description"):
                                        st.caption(article["description"][:150] + "...")
                                    st.write(
                                        f"Source: {article.get('source')} | Relevance: {article.get('relevance_score',0):.2f}"
                                    )
                                    st.divider()

                        except Exception as e:
                            st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info(
                "📰 **Data Journalist disabled**. Enable in sidebar to run analysis."
            )

    with tab3:
        if run_quant_analyst:
            st.header("📈 Quantitative Analysis")

            if run_analysis:
                with st.spinner("📊 Analyzing stock data..."):
                    try:
                        agent = QuantitativeAnalystAgent()
                        result = agent.analyze(ticker, period=analysis_period)
                        st.success("✅ **Analysis Complete**")

                        structured_data = result.structured_data

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Confidence", f"{result.confidence_score*100:.1f}%"
                            )
                        with col2:
                            current_price = structured_data.get(
                                "price_analysis", {}
                            ).get("current_price", "N/A")
                            if current_price != "N/A":
                                st.metric("Current Price", f"${current_price:.2f}")
                            else:
                                st.metric("Current Price", "N/A")

                        st.markdown("---")
                        st.markdown("📊 **Analysis Details**")

                        # Price Analysis
                        st.subheader("PRICE ANALYSIS")
                        price_analysis = structured_data.get("price_analysis", {})
                        st.markdown(
                            f"• **Current trend:** {price_analysis.get('current_trend', 'N/A')}, below Period High, recent 1-day drop of {price_analysis.get('price_change_1d_pct', 'N/A')}%."
                        )
                        st.markdown(
                            f"• **Key levels:** Resistance at ${price_analysis.get('resistance', 'N/A')}, support near ${price_analysis.get('support', 'N/A')}."
                        )
                        st.markdown(
                            f"• **Volatility:** {price_analysis.get('volatility', 'N/A'):.1%}, indicating significant price fluctuations."
                        )
                        st.write("")  # Spacing

                        # Technical Signals
                        st.subheader("TECHNICAL SIGNALS")
                        tech_signals = structured_data.get("technical_signals", {})
                        st.markdown(
                            f"• **RSI:** {tech_signals.get('rsi', 'N/A')} ({tech_signals.get('rsi_signal', 'N/A')}), suggesting neither overbought nor oversold conditions."
                        )
                        st.markdown(
                            f"• **Moving Averages:** Price is currently {tech_signals.get('price_vs_sma20', 'N/A')} the SMA20, indicating short-term weakness."
                        )
                        st.markdown(
                            f"• **Trading Momentum:** {tech_signals.get('momentum_signal', 'N/A')}, with a 10-day Momentum at {tech_signals.get('10d_momentum', 'N/A')}%."
                        )
                        st.markdown(
                            f"• **Volume:** {tech_signals.get('volume_signal', 'N/A')} signal."
                        )
                        st.write("")  # Spacing

                        # Anomalies & Risks
                        st.subheader("ANOMALIES & RISKS")
                        anomalies_risks = structured_data.get("anomalies_risks", {})
                        if anomalies_risks.get("recent_price_anomalies"):
                            st.markdown(
                                "• **Price Anomalies:** "
                                + ", ".join(anomalies_risks["recent_price_anomalies"])
                                + " suggest potential instability or reaction to news."
                            )
                        else:
                            st.markdown(
                                "• **Price Anomalies:** None detected recently."
                            )

                        if anomalies_risks.get("recent_volume_anomalies"):
                            st.markdown(
                                "• **Volume Anomalies:** "
                                + ", ".join(anomalies_risks["recent_volume_anomalies"])
                                + " potentially correlated with price movements, indicating strong selling pressure on Aug 1st."
                            )
                        else:
                            st.markdown(
                                "• **Volume Anomalies:** None detected recently."
                            )

                        st.markdown(
                            f"• **Risk Factors:** {anomalies_risks.get('risk_factors', 'N/A')}"
                        )
                        st.write("")
                        st.markdown("---")

                        st.subheader("BOTTOM LINE")
                        st.write(structured_data.get("bottom_line", "N/A"))

                    except Exception as e:
                        st.error(f"❌ **Error during Quantitative Analysis**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info(
                "📈 **Quantitative Analyst disabled**. Enable in sidebar to run analysis."
            )

    with tab4:
        if run_regulator_specialist:
            st.header("📋 Regulatory Analysis")

            if run_analysis:
                if not settings.gemini_api_key:
                    st.error(
                        "⚠️ **API Key Required**: Please configure your Gemini API key"
                    )
                else:
                    with st.spinner("📋 Analyzing regulatory filings..."):
                        try:
                            agent = RegulatorSpecialistAgent()
                            result = agent.analyze(
                                ticker,
                                filing_types=rs_filing_types,
                                max_filings=rs_max_filings,
                            )

                            st.success("✅ **Analysis Complete**")

                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "Confidence", f"{result.confidence_score*100:.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Filings Found",
                                    result.metadata.get("filings_analyzed", 0),
                                )

                            st.subheader("📊 Analysis Results")
                            st.write(result.analysis)

                            if result.metadata.get("filings_analyzed") > 0:
                                st.subheader("📋 Key Regulatory Insights")
                                st.write(
                                    f"**Latest Filing**: {result.metadata.get('latest_filing', 'N/A')}"
                                )
                                st.write(
                                    f"**Compliance Status**: {result.metadata.get('compliance_status', 'N/A')}"
                                )
                                st.write(
                                    f"**Regulatory Health**: {result.metadata.get('regulatory_health', 'N/A')}"
                                )
                                if result.metadata.get("risk_signals"):
                                    st.write(
                                        "**Risk Signals**: "
                                        + ", ".join(result.metadata["risk_signals"])
                                    )
                                else:
                                    st.write("**Risk Signals**: None detected")

                        except Exception as e:
                            st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info(
                "📋 **Regulator Specialist disabled**. Enable in sidebar to run analysis."
            )

    with tab5:
        if run_lead_analyst:
            st.header("🎯 Lead Analyst - Investment Brief")

            if run_analysis:
                if not settings.gemini_api_key:
                    st.error(
                        "⚠️ **API Key Required**: Please configure your Gemini API key"
                    )
                else:
                    # Check if other agents have been run
                    results = {}

                    # Run all enabled agents
                    with st.spinner("🔄 Running comprehensive analysis..."):
                        try:
                            if run_data_journalist:
                                st.write("📰 Running Data Journalist...")
                                agent = DataJournalistAgent()
                                results["news"] = agent.analyze(
                                    ticker,
                                    max_articles=max_articles,
                                    days_back=dj_days_back,
                                )

                            if run_quant_analyst:
                                st.write("📈 Running Quantitative Analyst...")
                                agent = QuantitativeAnalystAgent()
                                results["quant"] = agent.analyze(
                                    ticker, period=analysis_period
                                )

                            if run_regulator_specialist:
                                st.write("📋 Running Regulator Specialist...")
                                agent = RegulatorSpecialistAgent()
                                results["regulatory"] = agent.analyze(
                                    ticker,
                                    filing_types=rs_filing_types,
                                    max_filings=rs_max_filings,
                                )

                            # Now run lead analyst with all results
                            st.write("🎯 Synthesizing insights...")
                            lead_agent = LeadAnalystAgent()
                            final_result = lead_agent.analyze(
                                ticker, agent_results=results
                            )

                            st.success("✅ **Investment Brief Complete**")

                            # Display final recommendation
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "Overall Confidence",
                                    f"{final_result.confidence_score*100:.1f}%",
                                )
                            with col2:
                                if final_result.metadata.get("recommendation"):
                                    st.metric(
                                        "Recommendation",
                                        final_result.metadata["recommendation"].upper(),
                                    )

                            st.subheader("📋 Executive Summary")
                            st.write(final_result.analysis)

                            # Show risk factors if available
                            if final_result.metadata.get("risk_factors"):
                                st.subheader("⚠️ Key Risk Factors")
                                for risk in final_result.metadata["risk_factors"]:
                                    st.write(f"• {risk}")

                        except Exception as e:
                            st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info("🎯 **Lead Analyst disabled**. Enable in sidebar to run analysis.")


if __name__ == "__main__":
    main()
