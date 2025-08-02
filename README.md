# 🎯 AI Resume Screening System

An intelligent resume screening application that uses AI to automatically analyze and rank candidate resumes against job descriptions. Built with Streamlit and powered by LangChain and Groq's LLaMA models.

## ✨ Features

- **🤖 AI-Powered Analysis**: Uses advanced LLaMA models for intelligent resume screening
- **📁 Smart Folder Selection**: Multiple methods to select resume folders (GUI browser + manual navigation)
- **📊 Interactive Dashboard**: Beautiful results visualization with metrics and rankings
- **📧 Automated Email Notifications**: Send personalized emails to candidates based on screening results
- **📄 PDF Processing**: Extracts text from PDF resumes automatically
- **💾 Export Results**: Download analysis results as CSV files
- **🎨 Modern UI**: Clean, responsive interface with custom styling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key (get from [Groq Console](https://console.groq.com/))

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd ai-resume-screening
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional)**
   Create a `.env` file in the project root:
   ```
   LLAMA_API_KEY=your_groq_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   Open your browser and go to `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Configure API Key
- Enter your Groq API key in the sidebar
- Select your preferred model (llama3-70b-8192 recommended)
- Adjust temperature if needed (0.0 for consistent results)

### Step 2: Select Resume Folder
Choose one of two methods:
- **📂 Direct Folder Browser**: Click "Browse for Folder" for a GUI dialog
- **🗂️ Manual Navigation**: Navigate through folders manually

### Step 3: Enter Job Description
Paste the complete job description in the text area

### Step 4: Run Analysis
- Click "🚀 Start Analysis" to begin processing
- Monitor progress as each resume is analyzed
- View results in the interactive dashboard

### Step 5: Send Notifications (Optional)
- Switch to the "Email Notifications" tab
- Review candidate list
- Click "Send Email Notifications" to send automated responses

## 📋 Output Format

The system provides detailed analysis for each candidate:

- **Name**: Extracted from resume
- **Email**: Contact email address
- **Matching Score**: Percentage match with job requirements
- **Perfect Match**: Boolean indicating ideal candidate
- **Good Match**: Boolean indicating acceptable candidate
- **Strong Areas**: Candidate's strengths
- **Areas for Improvement**: Skills gaps identified

## 🔧 Configuration Options

### Supported Models
- `llama3-70b-8192` (Recommended for best accuracy)
- `llama3-8b-8192` (Faster processing)
- `mixtral-8x7b-32768` (Alternative option)

### Email Configuration
The system uses Gmail SMTP for email notifications. To use your own email:

1. Update the email credentials in the `send_mail` function
2. Enable "Less secure app access" or use App Passwords for Gmail
3. Modify email templates as needed

## 📁 File Structure

```
ai-resume-screening/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (optional)
└── resume_screening.csv  # Generated results file
```

## 🛠️ Technical Details

### Key Components

1. **FolderBrowser**: Handles folder selection with tkinter GUI
2. **UIManager**: Manages Streamlit UI components and interactions
3. **PDF Processing**: Uses PyMuPDF (fitz) for text extraction
4. **AI Analysis**: LangChain + Groq integration for resume analysis
5. **Email System**: Automated notifications using yagmail

### Data Flow

1. PDF resumes → Text extraction
2. Text + Job description → AI analysis
3. AI response → JSON parsing & normalization
4. Results → Dashboard visualization
5. Results → Email notifications (optional)

## 🔍 Troubleshooting

### Common Issues

**"No PDF files found"**
- Ensure PDFs are in the selected folder
- Check file extensions are `.pdf`

**"Error initializing LLM"**
- Verify API key is correct
- Check internet connection
- Ensure Groq service is available

**"Permission denied"**
- Run with appropriate permissions
- Check folder access rights

**Email sending fails**
- Verify email credentials
- Check Gmail security settings
- Ensure less secure apps are enabled

### Performance Tips

- Use smaller batches for large resume sets
- Consider using `llama3-8b-8192` for faster processing
- Implement rate limiting for API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [LangChain](https://langchain.com/) for AI integration
- [Groq](https://groq.com/) for LLaMA model access
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the [Groq documentation](https://console.groq.com/docs)
3. Open an issue in the repository

## 🔄 Updates

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added enhanced folder browser and dashboard
- **v1.2.0**: Improved error handling and email notifications

---

**Made with ❤️ for efficient recruitment processes**
