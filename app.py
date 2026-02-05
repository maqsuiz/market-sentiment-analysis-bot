"""
Sentiment Analysis Dashboard - Web Application
A visual interface for analyzing market sentiment with charts and links.
Uses Font Awesome icons instead of emojis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

from scraper import fetch_headlines
from analyzer import (
    analyze_headlines,
    calculate_distribution,
    calculate_market_score,
    get_extreme_headlines,
    generate_executive_summary,
    setup_gemini
)
from validator import evaluate_accuracy, get_validation_set
import database


# Page configuration
st.set_page_config(
    page_title="Market Sentiment Bot",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .header-icon {
        color: #667eea;
        margin-right: 10px;
    }
    .metric-card {
        background: rgba(26, 26, 46, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .disclaimer-box {
        background: rgba(255, 165, 2, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #ffa502;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        color: #ddd;
    }
    .positive { color: #00ff88; font-weight: 700; }
    .negative { color: #ff4757; font-weight: 700; }
    .neutral { color: #ffa502; font-weight: 700; }
    .fa-icon { margin-right: 8px; }
    .news-card, .news-card-bear {
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(5px);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        transition: background 0.2s ease, transform 0.2s ease;
        height: 70px; /* Fixed height for symmetry */
        display: flex;
        align-items: center;
        gap: 10px;
        border-left: 3px solid #667eea;
    }
    .news-card:hover, .news-card-bear:hover {
        background: rgba(30, 30, 46, 0.8);
        transform: translateX(3px);
    }
    .news-card-bear {
        border-left: 3px solid #ff4757;
    }
    .news-title {
        color: #ddd;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.2;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .section-header i {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .icon-positive { color: #00ff88; }
    .icon-negative { color: #ff4757; }
    .icon-neutral { color: #ffa502; }
    .icon-info { color: #667eea; }
    
    /* Sidebar Modernization */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stTextInput {
        margin-bottom: -10px;
    }
</style>
""", unsafe_allow_html=True)


# Font Awesome icon helper
def fa_icon(icon_class, color_class=""):
    """Generate Font Awesome icon HTML."""
    return f'<i class="fa-solid {icon_class} {color_class}"></i>'


def create_gauge_chart(score):
    """Create a perfectly centered, compact, and descriptive gauge."""
    if score >= 60:
        color, zone = "#00ff88", "EXTREME GREED"
    elif score >= 20:
        color, zone = "#a2ff00", "BULLISH"
    elif score <= -60:
        color, zone = "#ff4757", "EXTREME FEAR"
    elif score <= -20:
        color, zone = "#ff9f43", "BEARISH"
    else:
        color, zone = "#ffa502", "NEUTRAL"
    
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [-100, 100], 
                'tickwidth': 1, 
                'tickcolor': "rgba(255,255,255,0.3)", 
                'tickmode': 'array', 
                'tickvals': [-100, -50, 0, 50, 100],
                'tickfont': {'size': 10, 'color': '#888'}
            },
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': "rgba(255,255,255,0.03)",
            'borderwidth': 0,
            'steps': [
                {'range': [-100, -25], 'color': 'rgba(255, 71, 87, 0.1)'},
                {'range': [-25, 25], 'color': 'rgba(255, 165, 2, 0.05)'},
                {'range': [25, 100], 'color': 'rgba(0, 255, 136, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    # Zone Label (Top-Center of the arc area)
    fig.add_annotation(
        x=0.5, y=0.58,
        text=f"<b>{zone}</b>",
        showarrow=False,
        font=dict(size=13, color=color, family="Inter, sans-serif"),
        xref="paper", yref="paper",
        align="center"
    )
    
    # Score (Bottom-Center of the arc area)
    fig.add_annotation(
        x=0.5, y=0.28,
        text=f"<b>{score:+.0f}</b>",
        showarrow=False,
        font=dict(size=32, color="white", family="Arial Black"),
        xref="paper", yref="paper",
        align="center"
    )
    
    # Scale explanation
    fig.add_annotation(
        x=0.5, y=0.0,
        text="Scale: -100 (Fear) to +100 (Greed)",
        showarrow=False,
        font=dict(size=9, color="#555"),
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200, # Slimmers height
        margin=dict(t=20, b=20, l=40, r=40)
    )
    
    return fig


def create_pie_chart(distribution):
    """Create a sleek donut chart for sentiment distribution."""
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = ['#00ff88', '#ff4757', '#ffa502'] # Positive, Negative, Neutral
    
    fig = px.pie(
        values=values,
        names=labels,
        color_discrete_sequence=colors,
        hole=0.6,
        category_orders={"names": ["Positive", "Negative", "Neutral"]}
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#111', width=2)),
        hoverinfo='label+percent',
        pull=[0.05, 0.05, 0.05]
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'sans-serif'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25, # Pushed further down
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        height=300, # Increased height slightly for breathing room
        margin=dict(t=10, b=50, l=10, r=10) # More bottom margin
    )
    
    return fig


def create_sentiment_histogram(df):
    """Create a color-coded histogram of sentiment scores."""
    import numpy as np
    
    counts, bins = np.histogram(df.compound_score, bins=20, range=[-1, 1])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Color mapping: Red -> Yellow -> Green
    colors = []
    for val in bin_centers:
        if val <= -0.15: colors.append('#ff4757')
        elif val >= 0.15: colors.append('#00ff88')
        else: colors.append('#ffa502')
        
    fig = go.Figure(data=[go.Bar(
        x=bin_centers,
        y=counts,
        marker_color=colors,
        marker_line_width=0,
        width=0.08
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font={'color': 'white'},
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title="Sentiment Score", range=[-1, 1], gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Frequency", gridcolor='rgba(255,255,255,0.05)')
    )
    return fig


def create_source_chart(df):
    """Create a bar chart of average sentiment by source."""
    source_stats = df.groupby('source')['compound_score'].mean().reset_index()
    source_stats = source_stats.sort_values('compound_score', ascending=False)
    
    fig = px.bar(
        source_stats,
        x='source',
        y='compound_score',
        color='compound_score',
        color_continuous_scale=['#ff4757', '#ffa502', '#00ff88'],
        range_color=[-1, 1],
        labels={'compound_score': 'Sentiment', 'source': 'Source'}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font={'color': 'white'},
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    return fig


def create_keyword_chart(df):
    """Extract and visualize meaningful financial keywords."""
    from collections import Counter
    import re
    
    # Robust financial stop-word list
    stop_words = {
        'the', 'a', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'is', 'it', 'with', 'by', 'as', 'from',
        'bitcoin', 'crypto', 'market', 'price', 'news', 'headlines', 'after', 'over', 'under', 'below', 
        'above', 'into', 'about', 'more', 'less', 'than', 'will', 'this', 'that', 'were', 'been', 'near',
        'nearly', 'finance', 'yahoo', 'stocks', 'stock', 'investing', 'analysis', 'weekly', 'daily',
        'could', 'should', 'would', 'also', 'just', 'some', 'very', 'here', 'there', 'when', 'where',
        'how', 'which', 'who', 'whom', 'whose', 'these', 'those', 'must', 'might'
    }
    
    all_titles = " ".join(df['title'].tolist()).lower()
    words = re.findall(r'\w+', all_titles)
    # Only keep words that are not stop-words and are not pure numbers
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3 and not w.isdigit()]
    
    top_words = Counter(filtered_words).most_common(10)
    if not top_words:
        return go.Figure().update_layout(title="No significant keywords found")
        
    word_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    word_df = word_df.sort_values('Count', ascending=True)
    
    fig = px.bar(
        word_df,
        y='Word',
        x='Count',
        orientation='h',
        color='Count',
        color_continuous_scale='Viridis',
        labels={'Count': 'Frequency', 'Word': 'Significant Topic'}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font={'color': 'white'},
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', automargin=True)
    )
    return fig


def create_topic_chart(df):
    """Create a donut chart to show why the market is moving (Topic Modeling)."""
    # Specifically focus on Negative and Neutral to see "why the fear"
    # But for a general view, we can use the whole DF
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    fig = px.pie(
        topic_counts,
        values='Count',
        names='Topic',
        hole=0.6,
        color_discrete_sequence=['#667eea', '#764ba2', '#ff4757', '#ffa502', '#00ff88', '#888']
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#111', width=2))
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'sans-serif'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25, # Pushed further down
            xanchor="center",
            x=0.5
        ),
        height=300, # Increased height slightly for breathing room
        margin=dict(t=10, b=50, l=10, r=10) # More bottom margin
    )
    
    return fig


def get_sentiment_icon(sentiment):
    """Return Font Awesome icon based on sentiment."""
    if sentiment == "Positive":
        return '<i class="fa-solid fa-circle-check icon-positive"></i>'
    elif sentiment == "Negative":
        return '<i class="fa-solid fa-circle-xmark icon-negative"></i>'
    return '<i class="fa-solid fa-circle-minus icon-neutral"></i>'


@st.dialog("Model Validation Results", width="large")
def show_validation_dialog():
    """Show validation results in a popup dialog."""
    st.markdown('<h3><i class="fa-solid fa-flask icon-info"></i> Model Validation Results</h3>', unsafe_allow_html=True)
    
    metrics = evaluate_accuracy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall Accuracy",
            value=f"{metrics['overall_accuracy']}%"
        )
    
    with col2:
        st.metric(
            label="Correct Predictions",
            value=f"{metrics['correct_predictions']}/{metrics['total_samples']}"
        )
    
    with col3:
        st.metric(
            label="Validation Samples",
            value=metrics['total_samples']
        )
    
    st.divider()
    st.markdown("#### Per-Class Performance")
    class_data = []
    for cls, m in metrics['per_class_metrics'].items():
        class_data.append({
            'Class': cls,
            'Precision': f"{m['precision']}%",
            'Recall': f"{m['recall']}%",
            'F1-Score': f"{m['f1_score']}%",
            'Support': m['support']
        })
    
    st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)
    
    st.divider()
    st.markdown('''
    <p style="color: #888;">
        <i class="fa-solid fa-info-circle"></i>
        Validation performed on 30 manually labeled financial headlines.
    </p>
    ''', unsafe_allow_html=True)
    
    if st.button("Close", use_container_width=True):
        st.rerun()


def main():
    # Financial Disclaimer
    st.markdown('''
    <div class="disclaimer-box">
        <i class="fa-solid fa-triangle-exclamation" style="margin-right: 8px;"></i>
        <strong>DISCLAIMER:</strong> This tool is an AI-powered estimator based on news sentiment. 
        The analysis is for <strong>informational purposes only</strong> and does NOT constitute financial 
        or investment advice. Always perform your own due diligence before making trading decisions.
    </div>
    ''', unsafe_allow_html=True)

    # Header with Font Awesome icon
    st.markdown('''
    <h1 class="main-header">
        <i class="fa-solid fa-chart-line header-icon"></i>
        Market Sentiment Bot v1.8
    </h1>
    <p style="text-align: center; color: #aaa; margin-bottom: 2rem; font-size: 1.1rem;">
        <i class="fa-solid fa-brain" style="margin-right: 8px; color: #6C63FF;"></i>
        Decoding Market Psychology through Advanced Neural Analysis
    </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<h1 style="text-align: center; color: #6C63FF; font-size: 1.5rem; margin-bottom: 1rem;">📊 Bot v1.8 PRO</h1>', unsafe_allow_html=True)
        st.divider()
        
        # --- Section 1: Data Settings ---
        st.markdown('### <i class="fa-solid fa-database icon-info"></i> Data Settings', unsafe_allow_html=True)
        asset_type = st.selectbox(
            "Target Market",
            options=["crypto", "bist"],
            format_func=lambda x: "Cryptocurrency" if x == "crypto" else "BIST100 Stocks",
            key="sb_type"
        )
        
        if asset_type == "crypto":
            symbol = st.selectbox(
                "Select Asset",
                options=["bitcoin", "ethereum", "solana", "dogecoin", "ripple"],
                index=0,
                key="sb_asset"
            )
        else:
            symbol = "bist100"
            
        count = st.slider("Headline Count", 10, 100, 30, step=10, key="sb_count")

        st.divider()
        
        # --- Section 2: Brain Settings ---
        st.markdown('### <i class="fa-solid fa-brain icon-info"></i> Analysis Engine', unsafe_allow_html=True)
        engine = st.selectbox(
            "Select AI Engine",
            options=["vader", "finbert", "gemini"],
            format_func=lambda x: {
                "vader": "VADER (Basic/Fast)",
                "finbert": "FinBERT (Professional)",
                "gemini": "Gemini AI (Advanced)"
            }[x],
            key="sb_engine"
        )
        if engine == "finbert":
            st.warning("⚠️ FinBERT requires 400MB download on first run.")

        st.divider()
        
        # Fetch Gemini Key silently (removed from UI as per user request)
        gemini_key = os.getenv("GOOGLE_API_KEY", "")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("RUN ANALYSIS 🚀", use_container_width=True)
        st.divider()
        
        # Validation section
        st.markdown('<h3><i class="fa-solid fa-flask icon-info"></i> Model Validation</h3>', unsafe_allow_html=True)
        if st.button("Run Accuracy Test", use_container_width=True):
            show_validation_dialog()
    
    # Main content area
    if analyze_clicked:
        with st.spinner(f"Fetching {count} headlines for {symbol.upper()}..."):
            try:
                headlines = fetch_headlines(asset_type, symbol, count)
                
                if len(headlines) == 0:
                    st.error("No headlines found. Please try again later.")
                    return
                
                df = analyze_headlines(headlines, engine=engine, api_key=gemini_key)
                distribution = calculate_distribution(df)
                score, interpretation = calculate_market_score(df)
                most_positive, most_negative = get_extreme_headlines(df, n=5)
                
                # Calculate Uncertainty (Standard Deviation of scores)
                uncertainty = df['compound_score'].std() * 100
                
                # Generate results dictionary
                st.session_state['results'] = {
                    'df': df,
                    'distribution': distribution,
                    'score': score,
                    'interpretation': interpretation,
                    'most_positive': most_positive,
                    'most_negative': most_negative,
                    'uncertainty': uncertainty,
                    'symbol': symbol,
                    'count': len(df)
                }
                
                # Save to database (New in Technical Infra update)
                try:
                    database.init_db()
                    database.save_results(df, asset_type, symbol, engine=engine)
                except Exception as e:
                    st.error(f"Database Error: {e}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        df = results['df']
        distribution = results['distribution']
        score = results['score']
        interpretation = results['interpretation']
        
        # Top metrics row
        st.markdown('<p style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">📊 Market Sentiment Metrics</p>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <i class="fa-solid fa-arrow-trend-up icon-positive" style="font-size: 1.5rem;"></i>
                <h3 style="color: #00ff88; margin: 0.3rem 0;">%{distribution['Positive']:.0f}</h3>
                <p style="color: #888; font-size: 0.7rem; margin: 0;">Positive</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <i class="fa-solid fa-arrow-trend-down icon-negative" style="font-size: 1.5rem;"></i>
                <h3 style="color: #ff4757; margin: 0.3rem 0;">%{distribution['Negative']:.0f}</h3>
                <p style="color: #888; font-size: 0.7rem; margin: 0;">Negative</p>
            </div>
            ''', unsafe_allow_html=True)
            
        with col3:
            # Score card
            score_color = "#00ff88" if score >= 10 else "#ff4757" if score <= -10 else "#ffa502"
            st.markdown(f'''
            <div class="metric-card" style="border-top: 3px solid {score_color};">
                <i class="fa-solid fa-gauge-high" style="font-size: 1.5rem; color: {score_color};"></i>
                <h3 style="color: white; margin: 0.3rem 0;">{score:+.1f}</h3>
                <p style="color: {score_color}; font-weight: bold; font-size: 0.7rem; margin: 0;">{interpretation}</p>
            </div>
            ''', unsafe_allow_html=True)

        with col4:
            # Uncertainty card (Std Dev)
            u_val = results.get('uncertainty', 0)
            u_color = "#00ff88" if u_val < 30 else "#ffa502" if u_val < 60 else "#ff4757"
            u_text = "Stable" if u_val < 30 else "Mixed" if u_val < 60 else "High Risk"
            st.markdown(f'''
            <div class="metric-card">
                <i class="fa-solid fa-bolt" style="font-size: 1.5rem; color: {u_color};"></i>
                <h3 style="color: white; margin: 0.3rem 0;">%{u_val:.0f}</h3>
                <p style="color: #888; font-size: 0.7rem; margin: 0;">Uncertainty / {u_text}</p>
            </div>
            ''', unsafe_allow_html=True)
            
        with col5:
            st.markdown(f'''
            <div class="metric-card">
                <i class="fa-solid fa-newspaper icon-info" style="font-size: 1.5rem;"></i>
                <h3 style="color: white; margin: 0.3rem 0;">{results['count']}</h3>
                <p style="color: #888; font-size: 0.7rem; margin: 0;">Sample Count</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.divider()
        
        # Compact Analysis Row (3 Columns: Swipeable Charts, Bullish, Bearish)
        c_left, c_mid, c_right = st.columns([1.5, 1, 1])
        
        with c_left:
            st.markdown('<h4><i class="fa-solid fa-chart-line icon-info"></i> Insights</h4>', unsafe_allow_html=True)
            tab_pie, tab_gauge, tab_dist, tab_sources, tab_words, tab_topics = st.tabs([
                "📊 Distribution", 
                "📈 Gauge",
                "🎯 Scores",
                "📰 Sources",
                "🔤 Keywords",
                "🧠 Topics"
            ])
            with tab_pie:
                st.plotly_chart(create_pie_chart(distribution), use_container_width=True)
            with tab_gauge:
                st.plotly_chart(create_gauge_chart(score), use_container_width=True)
            with tab_dist:
                st.plotly_chart(create_sentiment_histogram(df), use_container_width=True)
            with tab_sources:
                st.plotly_chart(create_source_chart(df), use_container_width=True)
            with tab_words:
                st.plotly_chart(create_keyword_chart(df), use_container_width=True, config={'displayModeBar': False})
            with tab_topics:
                col_t1, col_t2 = st.columns([1, 1])
                with col_t1:
                    st.plotly_chart(create_topic_chart(df), use_container_width=True, config={'displayModeBar': False})
                with col_t2:
                    # Summary of topics for negative news
                    neg_df = df[df['sentiment'] == 'Negative']
                    if not neg_df.empty:
                        top_reason = neg_df['topic'].value_counts().idxmax()
                        st.markdown(f'''
                        <div style="background: rgba(255, 71, 87, 0.05); border-left: 3px solid #ff4757; padding: 15px; border-radius: 8px; margin-top: 20px;">
                            <h5 style="color: #ff4757; margin:0; font-weight: 700;">📉 Market Driver Analysis</h5>
                            <p style="font-size: 0.9rem; color: #bbb; margin: 10px 0;">
                                Primary catalyst for current market <b>Fear</b>: <br>
                                <span style="color: white; font-size: 1.1rem; display: block; margin-top: 5px;">📍 <strong style="color: #ff4757;">{top_reason}</strong></span>
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                        <div style="background: rgba(0, 255, 136, 0.05); border-left: 3px solid #00ff88; padding: 15px; border-radius: 8px; margin-top: 20px;">
                            <h5 style="color: #00ff88; margin:0; font-weight: 700;">✅ Bullish Context</h5>
                            <p style="font-size: 0.9rem; color: #bbb; margin: 10px 0;">No significant negative catalysts identified in current headlines.</p>
                        </div>
                        ''', unsafe_allow_html=True)
            
            # --- NEW: Historical Tab ---
            with st.expander("📈 View Historical Trends", expanded=False):
                st.markdown(f"#### Sentiment History for {symbol.upper()}")
                history_df = database.get_historical_trends(asset_type, symbol)
                if not history_df.empty:
                    # Convert timestamp strings to datetime objects for plotting
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    
                    fig_history = px.line(
                        history_df, 
                        x='timestamp', 
                        y='market_score',
                        title=f"{symbol.upper()} Sentiment Over Time",
                        labels={'market_score': 'Market Score', 'timestamp': 'Time'},
                        markers=True
                    )
                    fig_history.update_traces(line_color='#667eea', line_width=3)
                    fig_history.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.1)',
                        font={'color': 'white'}
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
                else:
                    st.info("No historical data found for this asset yet. Multiple analyses are needed to show trends.")
        
        with c_mid:
            st.markdown('<h4><i class="fa-solid fa-arrow-trend-up icon-positive"></i> Bullish</h4>', unsafe_allow_html=True)
            for _, row in results['most_positive'].iterrows():
                score_val = row['compound_score']
                topic = row.get('topic', 'General')
                st.markdown(f'''
                <div class="news-card">
                    <span class="positive" style="min-width: 45px;">+{score_val:.2f}</span>
                    <div style="flex-grow: 1;">
                        <span style="background: rgba(100,255,100,0.1); color: #00ff88; font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; font-weight: bold; margin-bottom: 2px; display: inline-block;">{topic}</span>
                        <div class="news-title">{row['title']}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        with c_right:
            st.markdown('<h4><i class="fa-solid fa-arrow-trend-down icon-negative"></i> Bearish</h4>', unsafe_allow_html=True)
            for _, row in results['most_negative'].iterrows():
                score_val = row['compound_score']
                topic = row.get('topic', 'General')
                st.markdown(f'''
                <div class="news-card-bear">
                    <span class="negative" style="min-width: 45px;">{score_val:.2f}</span>
                    <div style="flex-grow: 1;">
                        <span style="background: rgba(255,100,100,0.1); color: #ff4757; font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; font-weight: bold; margin-bottom: 2px; display: inline-block;">{topic}</span>
                        <div class="news-title">{row['title']}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.divider()
        
        # Full data table
        st.markdown('<h3><i class="fa-solid fa-table-list icon-info"></i> All Headlines</h3>', unsafe_allow_html=True)
        
        cols_to_show = ['title', 'source', 'compound_score', 'sentiment', 'reasoning', 'date']
        display_df = df[cols_to_show].copy()
        display_df.columns = ['Headline', 'Source', 'Score', 'Sentiment', 'AI Reasoning', 'Date']
        
        # Add clickable links if URL available
        if 'url' in df.columns:
            display_df['Link'] = df['url'].apply(
                lambda x: f"[Open]({x})" if x else ""
            )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.3f"),
                "Link": st.column_config.LinkColumn(),
                "AI Reasoning": st.column_config.TextColumn(width="large")
            }
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"sentiment_{results['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Validation results are now shown in a popup dialog
    
    # Footer
    st.markdown("---")
    st.markdown('''
    <p style="text-align: center; color: #666; font-size: 0.8rem;">
        <i class="fa-solid fa-code"></i> Developed by <strong>Talay</strong> | 
        <i class="fa-solid fa-brain"></i> Powered by advanced sentiment analysis
    </p>
    <p style="text-align: center; color: #444; font-size: 0.7rem; margin-top: -10px;">
        <i class="fa-solid fa-rss"></i> Data from CoinDesk, CryptoNews & Google News
    </p>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
