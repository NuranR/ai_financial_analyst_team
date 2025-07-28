"""Streamlit web interface for A-FIN."""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings


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
                st.metric("Agent Status", "Ready")
            with col2:
                st.metric("Data Sources", "4")
            with col3:
                st.metric("Analysis Period", analysis_period)
            with col4:
                st.metric("Articles Analyzed", max_articles)
            
            st.info("🚧 **Coming Soon**: Full agent analysis will be implemented in Phase 2!")
            
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
                st.info("This agent will analyze news articles, press releases, and social media mentions.")
                st.write("**Coming in Phase 2**: News sentiment analysis and market impact assessment")
            else:
                st.warning("Data Journalist agent not selected")
        
        with tab3:
            if run_quant_analyst:
                st.header("📈 Quantitative Analysis")
                st.info("This agent will analyze stock price data and detect trading anomalies.")
                st.write("**Coming in Phase 2**: Technical analysis and anomaly detection")
            else:
                st.warning("Quantitative Analyst agent not selected")
        
        with tab4:
            if run_regulator_specialist:
                st.header("📋 Regulatory Analysis")
                st.info("This agent will analyze SEC filings and regulatory documents.")
                st.write("**Coming in Phase 2**: Risk assessment and compliance analysis")
            else:
                st.warning("Regulator Specialist agent not selected")
        
        with tab5:
            if run_lead_analyst:
                st.header("🎯 Investment Brief")
                st.info("This agent will synthesize all insights into a comprehensive investment recommendation.")
                st.write("**Coming in Phase 2**: Complete investment thesis and recommendations")
            else:
                st.warning("Lead Analyst agent not selected")


if __name__ == "__main__":
    main()
