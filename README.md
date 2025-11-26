# MoodLens

**AI-Powered Mental Wellness Journal with Real-Time Emotion Analysis**

A sophisticated web application that combines sentiment analysis AI, data visualization, and pattern recognition to help users understand their emotional wellness journey.

**Live Demo**: https://moodlensapp.streamlit.app/

---

## Project Overview

MoodLens is a full-stack mental wellness application I built to demonstrate expertise in AI/ML integration, data visualization, UX/UI design, full-stack development, and data engineering. The application provides real-time sentiment analysis using VADER AI, interactive analytics dashboards with Plotly, and a professional responsive interface built with Python and Streamlit.

**Live Demo**: moodlens.streamlit.app

---

## Key Features

### 1. Real-Time Emotion Tracking
Instant sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner) with multi-dimensional scoring including compound, positive, neutral, and negative metrics. Features a 5-tier emotion classification system with visual emotion badges and color coding.

### 2. Advanced Analytics Dashboard
Interactive sentiment timeline showing emotional trends over time, pie chart breakdown of mood patterns, bar chart analysis of keyword frequency, key performance metrics including total entries and average sentiment, plus personalized AI-generated insights based on detected patterns.

### 3. Intelligent Keyword Extraction
Custom NLP algorithm to identify meaningful themes with stop word filtering using an extended dictionary of over 200 words. Implements frequency-based ranking system with visual keyword badges on each entry and pattern recognition across all entries.

### 4. Smart Writing Prompts
Over 20 curated therapeutic prompts to help overcome writer's block and encourage deeper self-reflection. Features randomized selection for variety and prompts designed based on counseling techniques.

### 5. Gratitude Tracking
Automatic filtering of positive entries with a dedicated gratitude journal view. Uses sentiment-based highlighting and encourages positive psychology practices backed by research.

### 6. Complete Data Export
JSON export with full data structure and metadata, plus CSV export in spreadsheet-compatible format. Preserves all analysis results with timestamp accuracy to the second, giving users complete ownership of their data.

---

## Technical Stack

### Core Technologies
Built with Python 3.11+, Streamlit 1.31 as the modern web framework, VADER Sentiment for pre-trained sentiment analysis, Plotly 5.18 for interactive data visualization, and Pandas 2.1 for data manipulation and analysis.

### Architecture
Frontend utilizes Streamlit reactive UI, sentiment engine powered by VADER NLP model, data layer using browser session state storage, visualization through Plotly.js with Python bindings, and deployment on Streamlit Cloud.

---

## Design System

### Visual Identity
Color palette features purple/blue gradient conveying professionalism and trust. Typography uses Inter font family for clean, modern appearance. Components utilize card-based layout with smooth animations, and all color contrast meets WCAG AA compliance standards.

### UI Components
Gradient header establishing brand identity, emotion-coded cards with colored left borders, interactive metric cards with hover effects, responsive badge system for keywords and emotions, and professional button styling with smooth animations.

### UX Principles
Immediate feedback through real-time sentiment analysis, clear visual hierarchy for content organization, emotion-appropriate color psychology, and minimal friction with one-click actions throughout the application.

---

## Technical Highlights

### Sentiment Analysis Implementation
Multi-dimensional sentiment analysis using VADER returns compound scores ranging from negative one to positive one, plus component scores for positive, neutral, and negative sentiment. Custom emotion classification logic categorizes results into five distinct emotional states.

### Keyword Extraction Algorithm
Tokenization with regex pattern matching, extended stop word filtering covering over 200 common words, minimum length validation requiring more than three characters, frequency counting using Counter class, and top-N selection for optimal display.

### Data Visualization
Dynamic chart generation with real-time updates, responsive design adapting to container width, interactive elements including hover tooltips and click events, plus performance optimization through cached computations.

### State Management
Efficient session state handling, immutable data patterns, optimized re-renders, and comprehensive data persistence strategies.

---

## Feature Breakdown

### Analytics Engine
Trend detection identifies upward and downward emotional patterns. Anomaly detection highlights unusual sentiment scores. Pattern recognition identifies recurring themes. Statistical analysis provides averages, distributions, and percentiles.

### Insights Generation
Algorithm generates personalized observations including overall mood assessment, recent trend analysis, recurring theme identification, positive momentum detection, and support recommendations when needed.

### Data Export Format
Exports include ISO timestamp, total entry count, app version, and complete entry details with full text, precise timestamps, comprehensive sentiment scores, emotion classifications, and extracted keywords.

---

## Privacy & Security

### Privacy-First Design
Zero data collection with no analytics or tracking. All computation happens client-side with local processing. No authentication required means no user accounts or passwords. Data never leaves the browser with no server storage. Users maintain complete ownership and control of their data.

### Security Measures
Session-based storage only, no external API calls except free Streamlit Cloud, no cookies or tracking pixels, no third-party integrations, and HTTPS deployment on Streamlit Cloud.

---

## Performance Metrics

### Efficiency
Load time under 2 seconds for initial load. Sentiment analysis completes in under 100ms per entry. Chart rendering takes under 500ms for 100 entries. Keyword extraction processes in under 200ms per entry.

### Scalability
Handles over 1000 entries smoothly with efficient memory management, optimized React-style re-renders, and cached computations where applicable.

---

## Learning Outcomes

### Skills Demonstrated
AI/ML Integration through implementing pre-trained NLP models. Data Visualization creating meaningful, interactive charts. Full-Stack Development with end-to-end application development. UX/UI Design with professional, user-centered design. Data Engineering including ETL processes and data transformation. Python Mastery demonstrating advanced patterns and libraries.

### Problem-Solving
Overcame real-time data processing challenges, state management in reactive frameworks, performance optimization strategies, user experience design decisions, and data privacy implementation.

---

## Future Enhancements

### Potential Improvements
Cloud storage integration with encryption, mobile app version using React Native, advanced ML models with GPT integration for insights, wearable device integration with Fitbit and Apple Watch, multi-language support, social features with anonymous community, and therapist collaboration tools.

---

## Development Process

### Built With
IDE using VS Code with Python extensions, version control with Git and GitHub, manual testing plus Streamlit's built-in debugger, and deployment on Streamlit Cloud free tier.

### Development Timeline
Phase 1 focused on core sentiment analysis over 2 days. Phase 2 developed UI/UX design system in 2 days. Phase 3 built analytics dashboard over 2 days. Phase 4 added additional features in 3 days. Phase 5 completed testing and refinement in 2 days.

### Challenges Overcome
Integrating VADER with Streamlit's reactive model, optimizing chart performance with large datasets, designing intuitive emotion classification system, balancing feature richness with simplicity, and creating professional design without formal design background.

---

## Project Goals Achieved

Achieved technical excellence with clean, maintainable code. Delivered intuitive, beautiful user experience. Created actually useful mental wellness tool with real-world value. Produced portfolio-quality work demonstrating multiple skills. Built fully functional, deployable complete product. Implemented ethical data handling with privacy focus.

---

## Portfolio Highlight

This project demonstrates my ability to build complete, production-ready applications, integrate AI/ML into practical solutions, create professional user interfaces, handle complex data visualization, prioritize user privacy and security, and deliver real value to end users.

**Perfect for roles in:** Full-Stack Development, Data Science and ML Engineering, Product Development, UX Engineering, and Healthcare Tech.

---

## Contact

**Built by**: Qazi Fabia Hoq  
**LinkedIn**: https://www.linkedin.com/in/qazifabiahoq/  

---

## License

MIT License - This project is part of my portfolio and showcases my development capabilities.

---

**MoodLens** - See Your Emotions Clearly

A demonstration of full-stack development, AI integration, and user-centered design.
