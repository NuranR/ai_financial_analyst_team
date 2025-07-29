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
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 A-FIN: Autonomous Financial Information Nexus")
    st.markdown("**Multi-Agent AI Financial Analysis System**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
        help="Historical data period for analysis"
    )
    
    max_articles = st.sidebar.slider(
        "Max News Articles",
        min_value=5,
        max_value=50,
        value=10,
        help="Maximum number of news articles to analyze"
    )
    
    # Agent selection
    st.sidebar.subheader("🤖 Select Agents")
    run_data_journalist = st.sidebar.checkbox("📰 Data Journalist", value=True)
    run_quant_analyst = st.sidebar.checkbox("📈 Quantitative Analyst", value=True)
    run_regulator_specialist = st.sidebar.checkbox("📋 Regulator Specialist", value=True)
    run_lead_analyst = st.sidebar.checkbox("🎯 Lead Analyst", value=True)
    
    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    # Test NewsAPI button
    test_news = st.sidebar.button("📰 Test NewsAPI", type="secondary")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "📰 Data Journalist", 
        "📈 Quantitative Analyst",
        "📋 Regulator Specialist",
        "🎯 Lead Analyst"
    ])
    
    with tab1:
        st.header("🏠 System Overview")
        
        # Check API key
        if not settings.gemini_api_key:
            st.error("⚠️ **API Key Missing**: Please set your GEMINI_API_KEY environment variable")
            st.info("💡 **Tip**: Create a `.env` file with: `GEMINI_API_KEY=your_key_here`")
        else:
            st.success("✅ **API Key Configured**: Gemini AI is ready")
        
        # System metrics
        st.subheader("📊 System Status")
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
            st.success("""
            **📈 Quantitative Analyst** ✅ READY
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
            
            if run_analysis:
                if not settings.gemini_api_key:
                    st.error("⚠️ **API Key Required**: Please configure your Gemini API key")
                else:
                    with st.spinner("🔍 Analyzing financial news..."):
                        try:
                            agent = DataJournalistAgent()
                            result = agent.analyze(ticker, max_articles=max_articles)
                            
                            st.success("✅ **Analysis Complete**")
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confidence", f"{result.confidence_score}%")
                            with col2:
                                st.metric("Data Points", len(result.metadata.get('articles', [])))
                            
                            st.subheader("📊 Analysis Results")
                            st.write(result.analysis)
                            
                            if result.metadata.get('articles'):
                                st.subheader("📰 Top Articles")
                                for i, article in enumerate(result.metadata['articles'][:3], 1):
                                    st.write(f"**{i}.** {article.get('title', 'No title')}")
                                    if article.get('description'):
                                        st.write(f"   {article['description'][:100]}...")
                        
                        except Exception as e:
                            st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info("📰 **Data Journalist disabled**. Enable in sidebar to run analysis.")
    
    with tab3:
        if run_quant_analyst:
            st.header("📈 Quantitative Analysis")
            
            if run_analysis:
                with st.spinner("📊 Analyzing stock data..."):
                    try:
                        agent = QuantitativeAnalystAgent()
                        result = agent.analyze(ticker, period=analysis_period)
                        
                        st.success("✅ **Analysis Complete**")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{result.confidence_score}%")
                        with col2:
                            if result.metadata.get('current_price'):
                                st.metric("Current Price", f"${result.metadata['current_price']:.2f}")
                        
                        st.subheader("📊 Analysis Results")
                        st.write(result.analysis)
                        
                        # Show key metrics if available
                        if result.metadata.get('metrics'):
                            st.subheader("📈 Key Metrics")
                            metrics = result.metadata['metrics']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if 'volatility' in metrics:
                                    st.metric("Volatility", f"{metrics['volatility']:.1%}")
                            with col2:
                                if 'return' in metrics:
                                    st.metric("Period Return", f"{metrics['return']:.1%}")
                            with col3:
                                if 'trend' in metrics:
                                    st.metric("Trend", metrics['trend'])
                    
                    except Exception as e:
                        st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info("📈 **Quantitative Analyst disabled**. Enable in sidebar to run analysis.")
    
    with tab4:
        if run_regulator_specialist:
            st.header("📋 Regulatory Analysis")
            
            if run_analysis:
                if not settings.gemini_api_key:
                    st.error("⚠️ **API Key Required**: Please configure your Gemini API key")
                else:
                    with st.spinner("📋 Analyzing regulatory filings..."):
                        try:
                            agent = RegulatorSpecialistAgent()
                            result = agent.analyze(ticker)
                            
                            st.success("✅ **Analysis Complete**")
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confidence", f"{result.confidence_score}%")
                            with col2:
                                st.metric("Filings Found", len(result.metadata.get('filings', [])))
                            
                            st.subheader("📊 Analysis Results")
                            st.write(result.analysis)
                            
                            if result.metadata.get('filings'):
                                st.subheader("📋 Recent Filings")
                                for filing in result.metadata['filings'][:5]:
                                    st.write(f"**{filing.get('form', 'Unknown')}** - {filing.get('filing_date', 'No date')}")
                                    if filing.get('description'):
                                        st.write(f"   {filing['description']}")
                        
                        except Exception as e:
                            st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info("📋 **Regulator Specialist disabled**. Enable in sidebar to run analysis.")
    
    with tab5:
        if run_lead_analyst:
            st.header("🎯 Lead Analyst - Investment Brief")
            
            if run_analysis:
                if not settings.gemini_api_key:
                    st.error("⚠️ **API Key Required**: Please configure your Gemini API key")
                else:
                    # Check if other agents have been run
                    results = {}
                    
                    # Run all enabled agents
                    with st.spinner("🔄 Running comprehensive analysis..."):
                        try:
                            if run_data_journalist:
                                st.write("📰 Running Data Journalist...")
                                agent = DataJournalistAgent()
                                results['news'] = agent.analyze(ticker, max_articles=max_articles)
                            
                            if run_quant_analyst:
                                st.write("📈 Running Quantitative Analyst...")
                                agent = QuantitativeAnalystAgent()
                                results['quant'] = agent.analyze(ticker, period=analysis_period)
                            
                            if run_regulator_specialist:
                                st.write("📋 Running Regulator Specialist...")
                                agent = RegulatorSpecialistAgent()
                                results['regulatory'] = agent.analyze(ticker)
                            
                            # Now run lead analyst with all results
                            st.write("🎯 Synthesizing insights...")
                            lead_agent = LeadAnalystAgent()
                            final_result = lead_agent.analyze(ticker, agent_results=results)
                            
                            st.success("✅ **Investment Brief Complete**")
                            
                            # Display final recommendation
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Overall Confidence", f"{final_result.confidence_score}%")
                            with col2:
                                if final_result.metadata.get('recommendation'):
                                    st.metric("Recommendation", final_result.metadata['recommendation'].upper())
                            
                            st.subheader("📋 Executive Summary")
                            st.write(final_result.analysis)
                            
                            # Show risk factors if available
                            if final_result.metadata.get('risk_factors'):
                                st.subheader("⚠️ Key Risk Factors")
                                for risk in final_result.metadata['risk_factors']:
                                    st.write(f"• {risk}")
                        
                        except Exception as e:
                            st.error(f"❌ **Error**: {str(e)}")
            else:
                st.info("👆 **Click 'Run Analysis' in the sidebar to start**")
        else:
            st.info("🎯 **Lead Analyst disabled**. Enable in sidebar to run analysis.")

    # Handle test news button
    if test_news:
        test_news_api_display(ticker)


def test_news_api_display(ticker: str):
    """Display NewsAPI test results in Streamlit"""
    from api.news_api import NewsAPI
    from data.stock_data import get_company_info
    
    st.header("📰 NewsAPI Test Results")
    
    try:
        # Get company info for proper name
        company_info = get_company_info(ticker)
        company_name = company_info.get('longName', ticker)
        
        # Test NewsAPI
        news_api = NewsAPI()
        with st.spinner(f"Fetching news for {company_name} ({ticker})..."):
            articles = news_api.get_company_news(company_name, ticker, days_back=7, max_articles=10)
        
        st.success(f"✅ Found {len(articles)} relevant articles for {ticker}")
        
        if articles:
            # Display articles
            for i, article in enumerate(articles):
                with st.expander(f"📰 Article {i+1}: {article['title']}", expanded=(i==0)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Source:** {article['source']}")
                        st.write(f"**Published:** {article['published_at']}")
                        if article['description']:
                            st.write(f"**Description:** {article['description']}")
                        st.write(f"**URL:** [Read Full Article]({article['url']})")
                    
                    with col2:
                        st.metric("Relevance Score", f"{article['relevance_score']:.2f}")
        else:
            st.warning("No relevant articles found.")
            
    except Exception as e:
        st.error(f"❌ **NewsAPI Test Failed**: {str(e)}")


if __name__ == "__main__":
    main()
