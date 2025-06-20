import streamlit as st
import requests
from typing import Literal
import time
import os

# Set Streamlit page config as the very first Streamlit command
st.set_page_config(
    page_title="Opinionwave",
    page_icon="🥷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Constants
SOURCE_TYPES = Literal["news", "reddit", "both"]
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:1234")  # Allow override via env var

def format_source_option(option: str) -> str:
    """Format source options with emojis"""
    emoji_map = {
        "news": "🌐 News Only",
        "reddit": "📑 Reddit Only", 
        "both": "🔄 News + Reddit"
    }
    return emoji_map.get(option, option.capitalize())

def main():
    # Page config
    # st.set_page_config(
    #     page_title="Opinionwave",
    #     page_icon="🥷",
    #     layout="centered",
    #     initial_sidebar_state="expanded"
    # )
    st.title("🥷 Opinionwave")
    st.markdown("#### 🎙️ News & Reddit Audio Summarizer")
    st.markdown("Generate AI-powered audio summaries from news and social media discussions")
    
    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    if 'generated_audio' not in st.session_state:
        st.session_state.generated_audio = None
    if 'generated_summary' not in st.session_state:
        st.session_state.generated_summary = None

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        source_type = st.selectbox(
            "Select Data Sources",
            options=["both", "news", "reddit"],
            format_func=format_source_option,
            help="Choose which sources to analyze for your topics"
        )
        
        st.markdown("---")
        st.markdown("**ℹ️ About**")
        st.markdown("""
        Opinionwave analyzes topics from:
        - 📰 Latest news articles
        - 💬 Reddit discussions
        - 🤖 AI-generated summaries
        """)
        
        # Backend status check
        st.markdown("---")
        if st.button("🔍 Check Backend Status"):
            check_backend_status()

    # Topic management section
    st.markdown("---")
    st.markdown("### 📝 Topic Management")
    
    # Input section
    col1, col2 = st.columns([4, 1])
    with col1:
        new_topic = st.text_input(
            "Enter a topic to analyze",
            key=f"topic_input_{st.session_state.input_key}",
            placeholder="e.g., Artificial Intelligence, Climate Change, Cryptocurrency",
            help="Add topics you want to get news and discussion summaries for"
        )
    
    with col2:
        # Limit to 5 topics max and ensure topic is not empty or duplicate
        add_disabled = (
            len(st.session_state.topics) >= 5 or 
            not new_topic.strip() or
            new_topic.strip().lower() in [t.lower() for t in st.session_state.topics]
        )
        
        if st.button("Add ➕", disabled=add_disabled, help="Add topic to analysis list"):
            if new_topic.strip():
                st.session_state.topics.append(new_topic.strip())
                st.session_state.input_key += 1
                st.rerun()

    # Show topic limit info
    if st.session_state.topics:
        st.info(f"📊 Topics added: {len(st.session_state.topics)}/5")
    
    # Display selected topics
    if st.session_state.topics:
        st.markdown("#### ✅ Selected Topics")
        
        for i, topic in enumerate(st.session_state.topics):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"**{i+1}.** {topic}")
            with col2:
                if st.button("🗑️", key=f"remove_{i}", help=f"Remove '{topic}'"):
                    st.session_state.topics.pop(i)
                    st.rerun()
        
        # Clear all topics button
        if len(st.session_state.topics) > 1:
            if st.button("🧹 Clear All Topics", type="secondary"):
                st.session_state.topics = []
                st.session_state.generated_audio = None
                st.session_state.generated_summary = None
                st.rerun()
    else:
        st.info("👆 Add topics above to get started")

    # Analysis section
    st.markdown("---")
    st.markdown("### 🔊 Audio Generation")
    
    # Show current settings
    with st.expander("📋 Current Settings", expanded=False):
        st.write(f"**Source Type:** {format_source_option(source_type)}")
        st.write(f"**Topics:** {len(st.session_state.topics)}")
        st.write(f"**Backend URL:** {BACKEND_URL}")
    
    # Generate button
    col1, col2 = st.columns([2, 1])
    with col1:
        generate_disabled = len(st.session_state.topics) == 0
        if st.button(
            "📝 Generate Summary", 
            disabled=generate_disabled,
            type="primary",
            help="Generate AI summary from selected topics and sources"
        ):
            st.session_state.generated_audio = None
            generate_summary(source_type)
    # Show summary and audio options
    if st.session_state.generated_summary:
        st.markdown("#### 📝 Generated Summary")
        st.write(st.session_state.generated_summary)
        if st.button("🔊 Generate Audio from Summary"):
            generate_audio_from_summary()
    if st.session_state.generated_audio:
        st.markdown("#### 🎵 Generated Audio Summary")
        st.audio(st.session_state.generated_audio, format="audio/mpeg")
        st.download_button(
            "💾 Download Audio",
            data=st.session_state.generated_audio,
            file_name=f"opinionwave-summary-{int(time.time())}.mp3",
            mime="audio/mpeg",
            help="Download the generated audio file"
        )

def check_backend_status():
    """Check if backend server is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Backend server is running")
        else:
            st.warning(f"⚠️ Backend responded with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server")
    except requests.exceptions.Timeout:
        st.error("⏱️ Backend server timeout")
    except Exception as e:
        st.error(f"❌ Error checking backend: {str(e)}")

def generate_summary(source_type):
    """Generate summary text from topics"""
    if not st.session_state.topics:
        st.error("❌ Please add at least one topic")
        return
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        progress_bar.progress(10)
        status_text.text("🔄 Connecting to backend...")
        payload = {
            "topics": st.session_state.topics,
            "source_type": source_type
        }
        progress_bar.progress(30)
        status_text.text("📡 Sending request to backend...")
        response = requests.post(
            f"{BACKEND_URL}/generate-news-audio?audio=false",
            json=payload,
            timeout=120
        )
        progress_bar.progress(70)
        status_text.text("📝 Processing summary response...")
        if response.status_code == 200:
            progress_bar.progress(100)
            status_text.text("✅ Summary generated successfully!")
            st.session_state.generated_summary = response.json().get("summary", "")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.success("📝 Summary generated successfully!")
        else:
            progress_bar.empty()
            status_text.empty()
            handle_api_error(response)
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"⚠️ **Unexpected Error**: {str(e)}")

def generate_audio_from_summary():
    """Generate audio from the summary text"""
    if not st.session_state.get("generated_summary"):
        st.error("No summary to convert to audio.")
        return
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        progress_bar.progress(10)
        status_text.text("🔄 Connecting to backend for audio...")
        payload = {
            "topics": st.session_state.topics,
            "source_type": st.session_state.get("source_type", "both")
        }
        progress_bar.progress(30)
        status_text.text("📡 Sending request to backend...")
        response = requests.post(
            f"{BACKEND_URL}/generate-news-audio?audio=true",
            json=payload,
            timeout=120
        )
        progress_bar.progress(70)
        status_text.text("🎵 Processing audio response...")
        if response.status_code == 200:
            progress_bar.progress(100)
            status_text.text("✅ Audio generated successfully!")
            st.session_state.generated_audio = response.content
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.success("🎉 Audio summary generated successfully!")
            st.balloons()
        else:
            progress_bar.empty()
            status_text.empty()
            handle_api_error(response)
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"⚠️ **Unexpected Error**: {str(e)}")

def handle_api_error(response):
    """Handle API error responses with detailed feedback"""
    try:
        error_data = response.json()
        error_detail = error_data.get("detail", "Unknown error occurred")
        
        # Provide specific error messages based on status code
        if response.status_code == 400:
            st.error(f"❌ **Bad Request**: {error_detail}")
            st.info("💡 Check your input parameters and try again")
        elif response.status_code == 401:
            st.error("🔐 **Authentication Error**: Invalid API credentials")
            st.info("💡 Check your API keys in the backend configuration")
        elif response.status_code == 429:
            st.error("🚦 **Rate Limit Exceeded**: Too many requests")
            st.info("💡 Please wait a moment before trying again")
        elif response.status_code == 500:
            st.error(f"🔧 **Server Error**: {error_detail}")
            st.info("💡 There's an issue with the backend server")
        else:
            st.error(f"❌ **API Error** ({response.status_code}): {error_detail}")
            
    except ValueError:
        # Response is not JSON
        st.error(f"❌ **Server Response Error** ({response.status_code})")
        with st.expander("🔍 Raw Response (for debugging)"):
            st.code(response.text[:500] + "..." if len(response.text) > 500 else response.text)

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
    .stButton > button {
        border-radius: 20px;
    }
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    load_custom_css()
    main()