import streamlit as st
import json
from datetime import datetime, timedelta
import re
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# BRANDING - MoodLens
# =============================================================================
APP_NAME = "MoodLens"
APP_TAGLINE = "See Your Emotions Clearly with AI-Powered Insights"
APP_ICON = "🔍"
APP_VERSION = "1.0.0"
# =============================================================================

# Try importing sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    st.warning("Run: pip install vaderSentiment")

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Gradient Purple/Blue Theme with Better Fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f3f7 100%);
    }
    
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 400;
    }
    
    .section-header {
        color: #1a202c;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 4px solid #667eea;
        padding-bottom: 0.75rem;
        letter-spacing: -0.3px;
    }
    
    .emotion-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-left: 5px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .emotion-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-top: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 0.5rem;
        word-wrap: break-word;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .positive-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .negative-badge {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .neutral-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        border: none;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    .prompt-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #f59e0b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        font-size: 1.05rem;
        font-weight: 500;
        color: #78350f;
    }
    
    .gratitude-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    .keyword-badge {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        color: #3730a3;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    
    .stats-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTextArea>div>div>textarea {
        font-size: 1.05rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .entry-timestamp {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        color: #1e293b;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f5f9;
        border-radius: 8px;
        color: #1e293b;
        font-weight: 600;
        font-size: 1rem;
        padding: 0 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #0f172a;
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for storing entries
if 'entries' not in st.session_state:
    st.session_state.entries = []

if 'show_prompt' not in st.session_state:
    st.session_state.show_prompt = False


def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    if not VADER_AVAILABLE or not text.strip():
        return {
            'compound': 0.0,
            'pos': 0.33,
            'neu': 0.34,
            'neg': 0.33,
            'emotion': 'Neutral'
        }
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    # Determine emotion
    compound = scores['compound']
    if compound >= 0.5:
        emotion = 'Very Positive'
    elif compound >= 0.05:
        emotion = 'Positive'
    elif compound <= -0.5:
        emotion = 'Very Negative'
    elif compound <= -0.05:
        emotion = 'Negative'
    else:
        emotion = 'Neutral'
    
    scores['emotion'] = emotion
    return scores


def extract_keywords(text, top_n=10):
    """Extract meaningful keywords from text"""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
        'that', 'am', 'been', 'being', 'so', 'than', 'too', 'very', 'just',
        'dont', 'now', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'than', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'up', 'down', 'out', 'off', 'over', 'until', 'while',
        'about', 'get', 'got', 'like', 'really', 'also', 'today', 'day'
    }
    
    # Extract words (preserve emotions and meaningful terms)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]*\b', text.lower())
    
    # Filter words
    keywords = [
        w for w in words 
        if w not in stop_words 
        and len(w) > 3 
        and not w.isdigit()
    ]
    
    # Count frequency
    word_freq = Counter(keywords)
    return [word for word, count in word_freq.most_common(top_n)]


def get_writing_prompt():
    """Get a random writing prompt"""
    prompts = [
        "What made you smile today? Describe that moment in detail.",
        "Write about a challenge you're facing and how you might overcome it.",
        "Describe three things you're grateful for right now and why.",
        "What would you tell your younger self about the situation you're in today?",
        "Write about a person who positively influenced you recently.",
        "What are you most proud of accomplishing this week?",
        "Describe a moment when you felt completely at peace.",
        "What's one thing you'd like to improve about yourself, and what's your first step?",
        "Write about something you're looking forward to.",
        "What lessons have you learned from your recent experiences?",
        "Describe how you're feeling right now without judging those feelings.",
        "What would your ideal day look like from start to finish?",
        "Write about a time you showed kindness to yourself or others.",
        "What are three words that describe how you want to feel, and what can help you get there?",
        "Reflect on a difficult emotion you've experienced recently. What was it trying to tell you?",
        "What boundaries do you need to set for your mental health?",
        "Write about something you've been avoiding and why.",
        "What gives your life meaning and purpose?",
        "Describe a recent accomplishment, no matter how small.",
        "What do you need to let go of to move forward?"
    ]
    
    import random
    return random.choice(prompts)


def create_sentiment_chart(entries_df):
    """Create sentiment trend chart"""
    if entries_df.empty:
        return None
    
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=entries_df['date'],
        y=entries_df['compound'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3)
    
    fig.update_layout(
        title="Your Emotional Journey Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
        title_font=dict(size=18, family="Inter, sans-serif", color="#1e293b"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_emotion_distribution(entries_df):
    """Create emotion distribution pie chart"""
    if entries_df.empty:
        return None
    
    emotion_counts = entries_df['emotion'].value_counts()
    
    colors = {
        'Very Positive': '#10b981',
        'Positive': '#34d399',
        'Neutral': '#f59e0b',
        'Negative': '#f87171',
        'Very Negative': '#ef4444'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=emotion_counts.index,
        values=emotion_counts.values,
        marker=dict(colors=[colors.get(e, '#667eea') for e in emotion_counts.index]),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=14, family="Inter, sans-serif")
    )])
    
    fig.update_layout(
        title="Emotion Distribution",
        template='plotly_white',
        height=400,
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
        title_font=dict(size=18, family="Inter, sans-serif", color="#1e293b"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_keyword_chart(all_keywords):
    """Create keyword frequency chart"""
    if not all_keywords:
        return None
    
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(15))
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_keywords.values()),
            y=list(top_keywords.keys()),
            orientation='h',
            marker=dict(
                color=list(top_keywords.values()),
                colorscale='Viridis',
                showscale=False
            ),
            text=list(top_keywords.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Most Common Themes in Your Entries",
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        template='plotly_white',
        height=500,
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
        title_font=dict(size=18, family="Inter, sans-serif", color="#1e293b"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def display_entry_card(entry, index):
    """Display a single journal entry as a card"""
    emotion = entry['sentiment']['emotion']
    compound = entry['sentiment']['compound']
    
    # Emotion badge
    if compound >= 0.05:
        badge_class = "positive-badge"
        emoji = "😊"
    elif compound <= -0.05:
        badge_class = "negative-badge"
        emoji = "😔"
    else:
        badge_class = "neutral-badge"
        emoji = "😐"
    
    st.markdown(f"""
    <div class="emotion-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span class="entry-timestamp">{entry['timestamp']}</span>
            <span class="{badge_class}">{emoji} {emotion}</span>
        </div>
        <div style="font-size: 1.05rem; line-height: 1.6; color: #334155;">
            {entry['text'][:300]}{"..." if len(entry['text']) > 300 else ""}
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
            <strong style="color: #667eea;">Keywords:</strong>
            <div style="margin-top: 0.5rem;">
                {''.join([f'<span class="keyword-badge">{kw}</span>' for kw in entry['keywords'][:5]])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown(f"""
    <div class="app-header">
        <div class="app-title">{APP_ICON} {APP_NAME}</div>
        <div class="app-subtitle">{APP_TAGLINE}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### About {APP_NAME}")
        st.markdown("""
        Your personal AI-powered mental wellness companion:
        
        **Core Features**
        - Real-time emotion tracking
        - Sentiment trend analysis
        - Keyword pattern recognition
        - Smart writing prompts
        - Gratitude highlighting
        - Complete data export
        
        **Privacy First**
        - 100% local storage
        - No data collection
        - Completely private
        - Your eyes only
        """)
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.entries:
            st.markdown("### Quick Stats")
            st.metric("Total Entries", len(st.session_state.entries))
            
            # Calculate average sentiment
            avg_sentiment = sum([e['sentiment']['compound'] for e in st.session_state.entries]) / len(st.session_state.entries)
            sentiment_emoji = "😊" if avg_sentiment > 0.05 else ("😔" if avg_sentiment < -0.05 else "😐")
            st.metric("Average Mood", f"{sentiment_emoji} {avg_sentiment:.2f}")
        
        st.markdown("---")
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Write** your thoughts freely
        2. **Analyze** emotions automatically
        3. **Track** patterns over time
        4. **Reflect** on your journey
        5. **Export** your complete history
        """)
        
        st.markdown("---")
        
        # Clear data button
        if st.session_state.entries:
            if st.button("Clear All Data", type="secondary"):
                if st.checkbox("Confirm deletion"):
                    st.session_state.entries = []
                    st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "New Entry",
        "Analytics",
        "Gratitude Journal",
        "Export Data"
    ])
    
    # TAB 1: New Entry
    with tab1:
        st.markdown('<div class="section-header">Write Your Thoughts</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("Get Writing Prompt"):
                st.session_state.show_prompt = True
        
        if st.session_state.show_prompt:
            prompt = get_writing_prompt()
            st.markdown(f"""
            <div class="prompt-card">
                <strong>Writing Prompt:</strong><br>
                {prompt}
            </div>
            """, unsafe_allow_html=True)
        
        entry_text = st.text_area(
            "How are you feeling?",
            height=250,
            placeholder="Express yourself freely... Your thoughts are private and safe here.",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("Save Entry", type="primary"):
                if entry_text and len(entry_text.strip()) > 10:
                    # Analyze entry
                    sentiment = analyze_sentiment(entry_text)
                    keywords = extract_keywords(entry_text)
                    
                    # Create entry
                    entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'text': entry_text,
                        'sentiment': sentiment,
                        'keywords': keywords
                    }
                    
                    # Save entry
                    st.session_state.entries.insert(0, entry)
                    
                    # Show success with analysis
                    st.success("Entry saved successfully!")
                    
                    st.markdown('<div class="section-header">Instant Analysis</div>', unsafe_allow_html=True)
                    
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="font-size: 1.1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{sentiment['emotion']}</div>
                            <div class="metric-label">Detected Emotion</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{sentiment['compound']:.2f}</div>
                            <div class="metric-label">Sentiment Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(keywords)}</div>
                            <div class="metric-label">Keywords Found</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if keywords:
                        st.markdown("**Key Themes:**")
                        st.markdown(''.join([f'<span class="keyword-badge">{kw}</span>' for kw in keywords[:8]]), unsafe_allow_html=True)
                    
                    st.session_state.show_prompt = False
                    
                else:
                    st.error("Please write at least 10 characters.")
        
        # Recent entries
        if st.session_state.entries:
            st.markdown('<div class="section-header">Recent Entries</div>', unsafe_allow_html=True)
            
            for idx, entry in enumerate(st.session_state.entries[:5]):
                display_entry_card(entry, idx)
    
    # TAB 2: Analytics
    with tab2:
        if not st.session_state.entries:
            st.info("Start writing entries to see your analytics and emotional trends!")
        else:
            st.markdown('<div class="section-header">Your Emotional Intelligence Dashboard</div>', unsafe_allow_html=True)
            
            # Convert to DataFrame
            entries_data = []
            all_keywords = []
            
            for entry in st.session_state.entries:
                entries_data.append({
                    'date': entry['date'],
                    'compound': entry['sentiment']['compound'],
                    'emotion': entry['sentiment']['emotion'],
                    'pos': entry['sentiment']['pos'],
                    'neg': entry['sentiment']['neg'],
                    'neu': entry['sentiment']['neu']
                })
                all_keywords.extend(entry['keywords'])
            
            df = pd.DataFrame(entries_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_entries = len(st.session_state.entries)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_entries}</div>
                    <div class="metric-label">Total Entries</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_sentiment = df['compound'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_sentiment:.2f}</div>
                    <div class="metric-label">Avg Sentiment</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                positive_ratio = (df['compound'] > 0.05).sum() / len(df) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{positive_ratio:.0f}%</div>
                    <div class="metric-label">Positive Days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                streak = len([e for e in st.session_state.entries if e['sentiment']['compound'] > 0])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{streak}</div>
                    <div class="metric-label">Positive Entries</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sentiment trend chart
            st.markdown('<div class="section-header">Sentiment Trends Over Time</div>', unsafe_allow_html=True)
            sentiment_chart = create_sentiment_chart(df)
            if sentiment_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Emotion distribution
                st.markdown('<div class="section-header">Emotion Distribution</div>', unsafe_allow_html=True)
                emotion_chart = create_emotion_distribution(df)
                if emotion_chart:
                    st.plotly_chart(emotion_chart, use_container_width=True)
            
            with col2:
                # Sentiment breakdown
                st.markdown('<div class="section-header">Sentiment Breakdown</div>', unsafe_allow_html=True)
                avg_pos = df['pos'].mean() * 100
                avg_neu = df['neu'].mean() * 100
                avg_neg = df['neg'].mean() * 100
                
                st.markdown(f"""
                <div class="stats-container">
                    <div style="margin-bottom: 1rem;">
                        <strong style="color: #10b981;">😊 Positive:</strong>
                        <div style="background: #d1fae5; height: 30px; border-radius: 5px; width: {avg_pos}%; display: inline-block; margin-left: 1rem;"></div>
                        <span style="margin-left: 0.5rem; font-weight: 600;">{avg_pos:.1f}%</span>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong style="color: #f59e0b;">😐 Neutral:</strong>
                        <div style="background: #fef3c7; height: 30px; border-radius: 5px; width: {avg_neu}%; display: inline-block; margin-left: 1rem;"></div>
                        <span style="margin-left: 0.5rem; font-weight: 600;">{avg_neu:.1f}%</span>
                    </div>
                    <div>
                        <strong style="color: #ef4444;">😔 Negative:</strong>
                        <div style="background: #fecaca; height: 30px; border-radius: 5px; width: {avg_neg}%; display: inline-block; margin-left: 1rem;"></div>
                        <span style="margin-left: 0.5rem; font-weight: 600;">{avg_neg:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Keyword analysis
            st.markdown('<div class="section-header">Your Most Common Themes</div>', unsafe_allow_html=True)
            keyword_chart = create_keyword_chart(all_keywords)
            if keyword_chart:
                st.plotly_chart(keyword_chart, use_container_width=True)
            
            # Insights
            st.markdown('<div class="section-header">Personalized Insights</div>', unsafe_allow_html=True)
            
            insights = []
            
            if avg_sentiment > 0.3:
                insights.append("You're maintaining a very positive mindset! Keep nurturing these feelings.")
            elif avg_sentiment > 0.1:
                insights.append("Your overall mood is positive. Great emotional balance!")
            elif avg_sentiment < -0.1:
                insights.append("You've been experiencing challenging emotions. Remember, it's okay to seek support.")
            
            if positive_ratio > 70:
                insights.append(f"{positive_ratio:.0f}% of your entries show positive emotions. You're doing amazing!")
            
            recent_trend = df.tail(5)['compound'].mean()
            overall_trend = df['compound'].mean()
            if recent_trend > overall_trend + 0.1:
                insights.append("Your recent entries show improvement in mood. Keep up the positive momentum!")
            elif recent_trend < overall_trend - 0.1:
                insights.append("Recent entries show lower mood. Consider what might be affecting you and practice self-care.")
            
            top_keywords = Counter(all_keywords).most_common(3)
            if top_keywords:
                top_themes = ', '.join([kw[0] for kw in top_keywords])
                insights.append(f"Your most recurring themes: {top_themes}. These topics are central to your current experience.")
            
            for insight in insights:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    # TAB 3: Gratitude Journal
    with tab3:
        st.markdown('<div class="section-header">Your Gratitude Collection</div>', unsafe_allow_html=True)
        
        if not st.session_state.entries:
            st.info("Start writing entries to see your gratitude moments!")
        else:
            # Filter positive entries
            positive_entries = [
                e for e in st.session_state.entries 
                if e['sentiment']['compound'] > 0.05
            ]
            
            if not positive_entries:
                st.warning("No positive entries yet. Keep writing - positive moments are coming!")
            else:
                st.markdown(f"""
                <div class="gratitude-card">
                    <h3 style="margin-top: 0;">{len(positive_entries)} Positive Moments Captured</h3>
                    <p style="font-size: 1.05rem; color: #065f46;">
                        You've documented {len(positive_entries)} entries with positive emotions. 
                        These are your bright spots - moments worth celebrating!
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="section-header">Highlighted Gratitude Entries</div>', unsafe_allow_html=True)
                
                # Show positive entries
                for entry in positive_entries[:10]:
                    st.markdown(f"""
                    <div class="gratitude-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span class="entry-timestamp">{entry['timestamp']}</span>
                            <span style="color: #10b981; font-weight: 600;">
                                Sentiment: {entry['sentiment']['compound']:.2f}
                            </span>
                        </div>
                        <div style="font-size: 1.05rem; line-height: 1.6; color: #065f46; margin-top: 1rem;">
                            {entry['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 4: Export Data
    with tab4:
        st.markdown('<div class="section-header">Export Your Wellness Data</div>', unsafe_allow_html=True)
        
        if not st.session_state.entries:
            st.info("No entries to export yet. Start writing to build your wellness history!")
        else:
            st.markdown("""
            <div class="stats-container">
                <h3>Complete Data Package</h3>
                <p style="font-size: 1.05rem; color: #475569;">
                    Download your complete mental wellness journal including all entries, 
                    sentiment analyses, keywords, and timestamps. Your data is yours to keep forever.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.entries)}</div>
                    <div class="metric-label">Entries Ready</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_words = sum([len(e['text'].split()) for e in st.session_state.entries])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_words}</div>
                    <div class="metric-label">Total Words</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                date_range = f"{st.session_state.entries[-1]['date']} to {st.session_state.entries[0]['date']}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">📅</div>
                    <div class="metric-label">Date Range</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 0.85rem; color: #64748b;'>{date_range}</p>", unsafe_allow_html=True)
            
            # Export options
            st.markdown('<div class="section-header">Choose Export Format</div>', unsafe_allow_html=True)
            
            # JSON Export
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_entries': len(st.session_state.entries),
                'app_version': APP_VERSION,
                'entries': st.session_state.entries
            }
            
            json_data = json.dumps(export_data, indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"mindflow_journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV Export
                csv_data = []
                for entry in st.session_state.entries:
                    csv_data.append({
                        'Timestamp': entry['timestamp'],
                        'Text': entry['text'],
                        'Emotion': entry['sentiment']['emotion'],
                        'Sentiment_Score': entry['sentiment']['compound'],
                        'Positive': entry['sentiment']['pos'],
                        'Neutral': entry['sentiment']['neu'],
                        'Negative': entry['sentiment']['neg'],
                        'Keywords': ', '.join(entry['keywords'])
                    })
                
                csv_df = pd.DataFrame(csv_data)
                csv_string = csv_df.to_csv(index=False)
                
                st.download_button(
                    label="Download as CSV",
                    data=csv_string,
                    file_name=f"mindflow_journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("""
            <div class="stats-container" style="margin-top: 2rem;">
                <h4>Privacy Notice</h4>
                <p style="color: #64748b;">
                    Your exported data is completely private. All processing happens locally in your browser.
                    We never see, store, or transmit your journal entries.
                </p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    if not VADER_AVAILABLE:
        st.error("⚠️ VADER Sentiment library not installed. Run: pip install vaderSentiment")
        st.stop()
    
    main()
