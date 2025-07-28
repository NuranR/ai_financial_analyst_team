"""Streamlit web interface for A-FIN."""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append                st.success("""
                **📈 Quantitative Analyst** ✅ READYs.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from agents.data_journalist import DataJournalistAgent
from agents.quantitative_analyst import QuantitativeAnalystAgent
from agents.regulator_specialist import RegulatorSpecialistAgent
from agents.lead_analyst import LeadAnalystAgent


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="A-FIN: Autonomous Financial Information Nexus",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 A-FIN: Autonomous Financial Information Nexus")
    st.markdown("*Multi-Agent Financial Analysis System*")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Input section
        ticker = st.text_input(
            "Company Ticker",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        # Agent selection
        st.subheader("Select Agents")
        run_data_journalist = st.checkbox("📰 Data Journalist", value=True)
        run_quant_analyst = st.checkbox("📈 Quantitative Analyst", value=True)
        run_regulator_specialist = st.checkbox("📋 Regulator Specialist", value=True)
        run_lead_analyst = st.checkbox("🎯 Lead Analyst", value=True)
        
        # Analysis parameters
        st.subheader("Parameters")
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1w", "1m", "3m", "6m", "1y"],
            index=4
        )
        
        max_articles = st.slider("Max News Articles", 10, 100, 50)
        
        # Run analysis button
        run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # Main content area
    if not run_analysis:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("---")
            st.markdown("## Welcome to A-FIN")
            st.markdown("""
            A-FIN is a multi-agent AI system that provides comprehensive financial analysis 
            by combining insights from four specialized agents:
            
            🤖 **Our Agent Team:**
            - 📰 **Data Journalist**: Analyzes news and social media sentiment
            - 📈 **Quantitative Analyst**: Detects trading anomalies and patterns
            - 📋 **Regulator Specialist**: Reviews SEC filings and compliance
            - 🎯 **Lead Analyst**: Synthesizes all insights into investment recommendations
            
            **Getting Started:**
            1. Enter a stock ticker in the sidebar
            2. Select which agents to run
            3. Configure analysis parameters
            4. Click "Run Analysis" to begin
            """)
            
            st.markdown("---")
            st.info("💡 **Tip**: Start with a well-known ticker like AAPL or TSLA for best results!")
    
    else:
        # Analysis results
        if not ticker:
            st.error("Please enter a valid ticker symbol")
            return
        
        # Create tabs for results
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📰 News Analysis", 
            "📈 Quantitative Analysis", 
            "📋 Regulatory Analysis", 
            "🎯 Investment Brief"
        ])
        
        with tab1:
            st.header(f"Analysis for {ticker}")
            
            # Placeholder for now - we'll implement agents next
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                agent_status = "✅ Ready" if settings.gemini_api_key else "❌ No API Key"
                st.metric("Agent Status", agent_status)
            with col2:
                data_sources = 2 if settings.gemini_api_key else 1  # Stock data always works
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
                st.success("""
                **📰 Data Journalist** ✅ READY
                - Fetches financial news
                - Analyzes market sentiment  
                - Identifies market-moving events
                - Uses Gemini AI for insights
                """)
                
            with col2:
                st.info("""
                **� Quantitative Analyst** ✅ READY
                - Downloads stock price data
                - Detects trading anomalies
                - Calculates technical indicators
                - Assesses volatility and risk
                """)
            
            col3, col4 = st.columns(2)
            with col3:
                st.success("""
                **📋 Regulator Specialist** ✅ READY
                - SEC filings analysis
                - Risk assessment
                - Compliance monitoring
                - Corporate governance review
                """)
                
            with col4:
                st.success("""
                **🎯 Lead Analyst** ✅ READY
                - Synthesizes all insights
                - Investment recommendations
                - Risk-adjusted analysis
                - Final investment brief
                """)
            
            # Show configuration
            st.subheader("Current Configuration")
            config_data = {
                "Ticker": ticker,
                "Data Journalist": "✅" if run_data_journalist else "❌",
                "Quantitative Analyst": "✅" if run_quant_analyst else "❌",
                "Regulator Specialist": "✅" if run_regulator_specialist else "❌",
                "Lead Analyst": "✅" if run_lead_analyst else "❌",
                "Analysis Period": analysis_period,
                "Max Articles": max_articles
            }
            
            for key, value in config_data.items():
                st.write(f"**{key}**: {value}")
        
        with tab2:
            if run_data_journalist:
                st.header("📰 Data Journalist Analysis")
                
                with st.spinner("Analyzing news and social media..."):
                    try:
                        journalist = DataJournalistAgent()
                        news_result = journalist.analyze(
                            ticker, 
                            days_back=7,
                            max_articles=max_articles
                        )
                        
                        if news_result.errors:
                            st.error(f"Error: {news_result.errors}")
                            if "news data available" in news_result.errors:
                                st.info("💡 **Tip**: Add a NewsAPI key to your .env file for live news analysis")
                        else:
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Articles Analyzed", news_result.metadata.get('articles_analyzed', 0))
                            with col2:
                                st.metric("Confidence Score", f"{news_result.confidence_score:.2f}")
                            with col3:
                                st.metric("Date Range", news_result.metadata.get('date_range', 'N/A'))
                            
                            st.subheader("📊 Analysis Results")
                            # Clean up any markdown formatting issues
                            clean_analysis = news_result.analysis.replace('*', '').replace('_', '')
                            st.markdown(clean_analysis)
                            
                            # Show metadata
                            with st.expander("Analysis Details"):
                                st.json(news_result.metadata)
                    
                    except Exception as e:
                        st.error(f"Error running Data Journalist: {str(e)}")
            else:
                st.warning("Data Journalist agent not selected")
        
        with tab3:
            if run_quant_analyst:
                st.header("📈 Quantitative Analysis")
                
                with st.spinner("Analyzing stock data and detecting anomalies..."):
                    try:
                        quant = QuantitativeAnalystAgent()
                        quant_result = quant.analyze(
                            ticker, 
                            period=analysis_period,
                            include_technical=True
                        )
                        
                        if quant_result.errors:
                            st.error(f"Error: {quant_result.errors}")
                        else:
                            # Display key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"${quant_result.metadata.get('current_price', 0)}")
                            with col2:
                                st.metric("Volatility", f"{quant_result.metadata.get('volatility', 0):.1%}")
                            with col3:
                                st.metric("Price Anomalies", quant_result.metadata.get('price_anomalies', 0))
                            with col4:
                                st.metric("Confidence", f"{quant_result.confidence_score:.2f}")
                            
                            st.subheader("📈 Analysis Results")
                            # Clean up any markdown formatting issues and display properly
                            clean_analysis = quant_result.analysis.replace('*', '').replace('_', '')
                            st.markdown(clean_analysis)
                            
                            # Show technical indicators if available
                            if quant_result.metadata.get('technical_indicators'):
                                with st.expander("Technical Indicators"):
                                    st.write("Indicators calculated: " + ", ".join(quant_result.metadata['technical_indicators']))
                            
                            # Show analysis details
                            with st.expander("Analysis Details"):
                                st.json(quant_result.metadata)
                    
                    except Exception as e:
                        st.error(f"Error running Quantitative Analyst: {str(e)}")
            else:
                st.warning("Quantitative Analyst agent not selected")
        
        with tab4:
            if run_regulator_specialist:
                st.header("📋 Regulatory Analysis")
                
                with st.spinner("Analyzing SEC filings and compliance..."):
                    try:
                        regulator = RegulatorSpecialistAgent()
                        reg_result = regulator.analyze(ticker, max_filings=3)
                        
                        if reg_result.errors:
                            st.error(f"Error: {reg_result.errors}")
                        else:
                            # Display key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Filings Analyzed", reg_result.metadata.get('filings_analyzed', 0))
                            with col2:
                                st.metric("Compliance Status", reg_result.metadata.get('compliance_status', 'Unknown'))
                            with col3:
                                st.metric("Regulatory Health", reg_result.metadata.get('regulatory_health', 'Unknown'))
                            with col4:
                                st.metric("Confidence", f"{reg_result.confidence_score:.2f}")
                            
                            st.subheader("📋 Analysis Results")
                            # Clean up any markdown formatting issues
                            clean_analysis = reg_result.analysis.replace('*', '').replace('_', '')
                            st.markdown(clean_analysis)
                            
                            # Show filing details
                            if reg_result.metadata.get('risk_signals'):
                                with st.expander("Risk Signals Detected"):
                                    for signal in reg_result.metadata['risk_signals']:
                                        st.warning(f"⚠️ {signal}")
                            
                            # Show analysis details
                            with st.expander("Regulatory Details"):
                                st.json(reg_result.metadata)
                    
                    except Exception as e:
                        st.error(f"Error running Regulator Specialist: {str(e)}")
            else:
                st.warning("Regulator Specialist agent not selected")
        
        with tab5:
            if run_lead_analyst:
                st.header("🎯 Comprehensive Investment Brief")
                
                with st.spinner("Synthesizing all analyses into investment recommendation..."):
                    try:
                        lead_analyst = LeadAnalystAgent()
                        final_result = lead_analyst.analyze(
                            ticker,
                            enable_news=run_data_journalist,
                            enable_quant=run_quant_analyst,
                            enable_regulatory=run_regulator_specialist,
                            analysis_period=analysis_period
                        )
                        
                        if final_result.errors:
                            st.error(f"Error: {final_result.errors}")
                        else:
                            # Display key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Overall Confidence", f"{final_result.confidence_score:.2f}")
                            with col2:
                                agents_run = len(final_result.metadata.get('agents_run', []))
                                st.metric("Analyses Combined", agents_run)
                            with col3:
                                st.metric("Current Price", f"${final_result.metadata.get('current_price', 'N/A')}")
                            with col4:
                                basis = final_result.metadata.get('recommendation_basis', {})
                                outlook = basis.get('technical_outlook', 'Unknown')
                                st.metric("Technical Outlook", outlook)
                            
                            st.subheader("🎯 Final Investment Brief")
                            # Clean up formatting and display
                            clean_analysis = final_result.analysis.replace('*', '').replace('_', '')
                            st.markdown(clean_analysis)
                            
                            # Show recommendation basis
                            basis = final_result.metadata.get('recommendation_basis', {})
                            if basis:
                                with st.expander("Recommendation Basis"):
                                    for factor, value in basis.items():
                                        st.write(f"**{factor.replace('_', ' ').title()}**: {value}")
                            
                            # Show individual agent confidences
                            confidences = final_result.metadata.get('individual_confidences', {})
                            if confidences:
                                with st.expander("Individual Agent Performance"):
                                    for agent, confidence in confidences.items():
                                        st.write(f"**{agent.title()} Agent**: {confidence:.2f}")
                    
                    except Exception as e:
                        st.error(f"Error running Lead Analyst: {str(e)}")
            else:
                st.warning("Lead Analyst agent not selected")


if __name__ == "__main__":
    main()
