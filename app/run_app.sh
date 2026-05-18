#!/bin/bash
# Launch the AI-Powered ATS Web Application
echo "🚀 Starting AI-Powered ATS..."
echo "📍 Open http://localhost:8501 in your browser"
echo ""
cd "$(dirname "$0")"
streamlit run app.py --server.port 8501 --server.headless false
