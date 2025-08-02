import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
import fitz
import pandas as pd
import json
import time
import yagmail
import re
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog

# Custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2E86AB;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .folder-selector {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .folder-path-display {
        background-color: #e9ecef;
        padding: 0.75rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
        word-break: break-all;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class FolderBrowser:
    """Handles folder browsing functionality"""
    
    @staticmethod
    def browse_folder() -> Optional[str]:
        """Open a folder browser dialog and return selected folder path"""
        try:
            # Create a root window and hide it
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            
            # Open folder dialog
            folder_path = filedialog.askdirectory(
                title="Select Folder Containing Resume PDFs",
                mustexist=True
            )
            
            # Clean up
            root.destroy()
            return folder_path if folder_path else None
        except Exception as e:
            st.error(f"Error opening folder browser: {str(e)}")
            return None

class UIManager:
    """Manages UI components and interactions"""
    
    @staticmethod
    def get_enhanced_folder_browser():
        """Enhanced folder browser with multiple selection methods"""
        st.markdown('<div class="section-header">📁 Select Resume Folder</div>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'selected_folder' not in st.session_state:
            st.session_state.selected_folder = None
        if 'current_dir' not in st.session_state:
            st.session_state.current_dir = str(Path.home())
        
        with st.container():
            st.markdown('<div class="folder-selector">', unsafe_allow_html=True)
            
            # Method selection
            method = st.radio(
                "Choose folder selection method:",
                ["📂 Direct Folder Browser (Recommended)", "🗂️ Manual Navigation"],
                horizontal=True
            )
            
            if method == "📂 Direct Folder Browser (Recommended)":
                st.markdown("**Click the button below to open a folder browser dialog:**")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("🔍 Browse for Folder", type="primary", use_container_width=True):
                        with st.spinner("Opening folder browser..."):
                            selected = FolderBrowser.browse_folder()
                            if selected:
                                st.session_state.selected_folder = selected
                                st.rerun()
                
                # Display selected folder
                if st.session_state.selected_folder:
                    st.markdown("**Selected Folder:**")
                    st.markdown(f'<div class="folder-path-display">{st.session_state.selected_folder}</div>', 
                               unsafe_allow_html=True)
                    
                    # Validate folder and show PDF count
                    try:
                        pdf_files = [f for f in os.listdir(st.session_state.selected_folder) 
                                   if f.lower().endswith('.pdf')]
                        if pdf_files:
                            st.markdown(f'<div class="success-box">✅ Found {len(pdf_files)} PDF file(s) in selected folder</div>', 
                                       unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">⚠️ No PDF files found in selected folder</div>', 
                                       unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error accessing folder: {str(e)}</div>', 
                                   unsafe_allow_html=True)
                
            else:
                # Manual navigation method
                UIManager._manual_folder_navigation()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return st.session_state.selected_folder
    
    @staticmethod
    def _manual_folder_navigation():
        """Manual folder navigation interface"""
        st.markdown("**Navigate to your folder manually:**")
        
        # Current directory display
        st.markdown(f"**Current Directory:** `{st.session_state.current_dir}`")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🏠 Home", key="nav_home"):
                st.session_state.current_dir = str(Path.home())
                st.rerun()
        
        with col2:
            if st.button("⬆️ Parent", key="nav_parent"):
                parent = str(Path(st.session_state.current_dir).parent)
                if parent != st.session_state.current_dir:
                    st.session_state.current_dir = parent
                    st.rerun()
        
        # List directories
        try:
            current_path = Path(st.session_state.current_dir)
            directories = [item for item in current_path.iterdir() if item.is_dir()]
            directories.sort(key=lambda x: x.name.lower())
            
            if directories:
                st.markdown("**Available Folders:**")
                
                # Create columns for better layout
                num_cols = min(3, len(directories))
                cols = st.columns(num_cols)
                
                for idx, directory in enumerate(directories):
                    col = cols[idx % num_cols]
                    with col:
                        button_key = f"dir_{idx}_{directory.name}"
                        if st.button(f"📁 {directory.name}", key=button_key, use_container_width=True):
                            st.session_state.current_dir = str(directory)
                            st.rerun()
            else:
                st.info("No subdirectories found in current location")
        
        except PermissionError:
            st.error("❌ Permission denied to access this directory")
        except Exception as e:
            st.error(f"❌ Error browsing directory: {str(e)}")
        
        # Manual path input and selection
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_path = st.text_input(
                "Or enter folder path manually:",
                value=st.session_state.current_dir,
                key="manual_folder_input"
            )
        
        with col2:
            if st.button("📂 Select", key="select_manual", type="primary"):
                folder_path = manual_path if manual_path else st.session_state.current_dir
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    st.session_state.selected_folder = folder_path
                    st.success(f"✅ Selected: {folder_path}")
                else:
                    st.error("❌ Invalid folder path!")

# LangChain prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a resume analysis expert. You will compare the Resume and Job Description and provide the output ONLY in valid JSON format. Do not include any explanatory text before or after the JSON."),
    ("human", "Analyze the resume against the job description: \n \
     Resume: {resume} \n \
     Job Description: {Job_description} \n \
     Provide a JSON response with these exact keys: Name, email, is_perfect, is_okay, Matching Score in percentage, strong zone, Lack of Knowledge. \
     Make sure the response is valid JSON format only, no additional text.")
])

def load_pdf(pdf_file):
    """Extract text from PDF"""
    pdf_document = fitz.open(pdf_file)
    pdf_text_with_links = ""

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pdf_text_with_links += page.get_text("text")

        links = page.get_links()
        for link in links:
            if 'uri' in link:
                pdf_text_with_links += f"\n(Link: {link['uri']})"
    
    return pdf_text_with_links

def parser(aimessage: AIMessage) -> str:
    return aimessage.content

def extract_json_from_response(response_text):
    """Extract JSON from LLM response, handling various formats"""
    def normalize_json_keys(data):
        """Normalize JSON keys to expected format"""
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        key_mapping = {
            'name': 'Name',
            'email': 'email',
            'is_perfect': 'is_perfect',
            'is_okay': 'is_okay',
            'matching_score': 'Matching Score in percentage',
            'matching score': 'Matching Score in percentage',
            'matching_score_percentage': 'Matching Score in percentage',
            'score': 'Matching Score in percentage',
            'strong_zone': 'strong zone',
            'strengths': 'strong zone',
            'lack_of_knowledge': 'Lack of Knowledge',
            'weaknesses': 'Lack of Knowledge',
            'gaps': 'Lack of Knowledge'
        }
        
        for key, value in data.items():
            # Normalize key to lowercase for comparison
            key_lower = key.lower().replace(' ', '_').replace('-', '_')
            
            # Map to standard key or use original if no mapping found
            if key_lower in key_mapping:
                normalized[key_mapping[key_lower]] = value
            else:
                # Try partial matches for score
                if 'score' in key_lower and 'percentage' in key_lower:
                    normalized['Matching Score in percentage'] = value
                elif 'score' in key_lower:
                    normalized['Matching Score in percentage'] = value
                else:
                    normalized[key] = value
        
        # Ensure all required keys exist
        default_values = {
            "Name": "Unknown",
            "email": "unknown@email.com", 
            "is_perfect": False,
            "is_okay": False,
            "Matching Score in percentage": "0%",
            "strong zone": "Not specified",
            "Lack of Knowledge": "Not specified"
        }
        
        for req_key, default_val in default_values.items():
            if req_key not in normalized:
                normalized[req_key] = default_val
        
        return normalized
    
    try:
        data = json.loads(response_text)
        return normalize_json_keys(data)
    except json.JSONDecodeError:
        try:
            cleaned = response_text.replace('```json', '').replace('```', '').strip()
            data = json.loads(cleaned)
            return normalize_json_keys(data)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return normalize_json_keys(data)
                else:
                    return normalize_json_keys({})
            except:
                return normalize_json_keys({})

def resume_checker(resume, job_description, llm):
    """Generate resume analysis using LangChain"""
    Chain = prompt_template | llm | parser
    result = Chain.invoke({"resume": resume, "Job_description": job_description})
    return result

def send_mail(data_path):
    """Send email notifications"""
    csv_data = pd.read_csv(data_path)
    yag = yagmail.SMTP('alhasib.iu.cse@gmail.com', 'iyto heuk fbka mysg')
    
    # Find the score column dynamically
    score_column = None
    possible_score_columns = [
        'Matching Score in percentage', 
        'Matching Score', 
        'Score', 
        'matching_score', 
        'score'
    ]
    
    for col in possible_score_columns:
        if col in csv_data.columns:
            score_column = col
            break
    
    if not score_column:
        # Look for any column containing 'score'
        for col in csv_data.columns:
            if 'score' in col.lower():
                score_column = col
                break
    
    if not score_column:
        st.error("❌ No score column found in the data")
        return
    
    for i in range(csv_data.shape[0]):
        data = csv_data.iloc[i]
        name = data.get('Name', 'Unknown')
        email = data.get('email', 'unknown@email.com')
        score_val = data.get(score_column, '0%')
        
        # Parse score
        try:
            if "%" in str(score_val):
                score = int(str(score_val).replace("%", ""))
            else:
                score = int(float(score_val))
        except (ValueError, TypeError):
            score = 0
        
        weak_zone = data.get('Lack of Knowledge', 'Not specified')

        if score < 70:
            message = f"""Hello {name},

Thank you for your application for the Machine Learning position in our company. Unfortunately, we cannot consider you for the further process. We found some areas that don't match with our requirements:

{weak_zone}

You can upgrade yourself and try later.

Best Regards,
Md Abdullah Al Hasib"""
        else:
            message = f"""Hello {name},

Thank you for your application for the Machine Learning position in our company. We are pleased to consider you for the further process.

Best Regards,
Md Abdullah Al Hasib"""
        
        try:
            yag.send(email, 'Response to Your ML Engineer Application', message)
        except Exception as e:
            st.error(f"❌ Failed to send email to {email}: {str(e)}")

def display_results_dashboard(df):
    """Display results in a dashboard format"""
    st.markdown('<div class="section-header">📊 Analysis Results Dashboard</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_resumes = len(df)
    
    # Check for boolean values or string values for perfect/okay matches
    perfect_candidates = 0
    okay_candidates = 0
    
    if 'is_perfect' in df.columns:
        perfect_candidates = len(df[df['is_perfect'].isin([True, 'True', 'true', 'TRUE', 1, '1'])])
    
    if 'is_okay' in df.columns:
        okay_candidates = len(df[df['is_okay'].isin([True, 'True', 'true', 'TRUE', 1, '1'])])
    
    # Calculate average score
    avg_score = 0
    score_column = 'Matching Score in percentage'
    
    if score_column in df.columns:
        scores = []
        for score in df[score_column]:
            try:
                if isinstance(score, str):
                    # Remove % sign and convert to int
                    clean_score = score.replace('%', '').strip()
                    scores.append(int(float(clean_score)))
                elif isinstance(score, (int, float)):
                    scores.append(int(score))
                else:
                    scores.append(0)
            except (ValueError, TypeError):
                scores.append(0)
        avg_score = sum(scores) / len(scores) if scores else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #2E86AB; margin: 0;">📄</h3>
            <h2 style="margin: 0;">{total_resumes}</h2>
            <p style="margin: 0; color: #666;">Total Resumes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #28a745; margin: 0;">⭐</h3>
            <h2 style="margin: 0;">{perfect_candidates}</h2>
            <p style="margin: 0; color: #666;">Perfect Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #ffc107; margin: 0;">👍</h3>
            <h2 style="margin: 0;">{okay_candidates}</h2>
            <p style="margin: 0; color: #666;">Good Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #17a2b8; margin: 0;">📈</h3>
            <h2 style="margin: 0;">{avg_score:.1f}%</h2>
            <p style="margin: 0; color: #666;">Avg Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sort by score (high to low) and show detailed debug information
    if score_column in df.columns:
        # Create a copy of dataframe with numeric scores for sorting
        df_debug = df.copy()
        df_debug['numeric_score'] = 0
        
        for idx, score in enumerate(df[score_column]):
            try:
                if isinstance(score, str):
                    clean_score = score.replace('%', '').strip()
                    df_debug.iloc[idx, df_debug.columns.get_loc('numeric_score')] = int(float(clean_score))
                elif isinstance(score, (int, float)):
                    df_debug.iloc[idx, df_debug.columns.get_loc('numeric_score')] = int(score)
            except (ValueError, TypeError):
                df_debug.iloc[idx, df_debug.columns.get_loc('numeric_score')] = 0
        
        # Sort by numeric score (high to low)
        df_sorted = df_debug.sort_values('numeric_score', ascending=False)
        
        # Show detailed debug information for each candidate
        with st.expander("🔍 Detailed Debug Information (Sorted by Score)", expanded=True):
            st.markdown("**Candidates ranked by score (High to Low):**")
            
            for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
                score_val = row.get(score_column, 'N/A')
                name = row.get('Name', 'Unknown')
                email = row.get('email', 'N/A')
                is_perfect = row.get('is_perfect', 'N/A')
                is_okay = row.get('is_okay', 'N/A')
                strong_zone = row.get('strong zone', 'N/A')
                lack_knowledge = row.get('Lack of Knowledge', 'N/A')
                numeric_score = row.get('numeric_score', 0)
                
                # Color coding based on score
                if numeric_score >= 80:
                    color = "#28a745"  # Green
                    badge = "🏆 Excellent"
                elif numeric_score >= 70:
                    color = "#ffc107"  # Yellow
                    badge = "⭐ Good"
                elif numeric_score >= 50:
                    color = "#fd7e14"  # Orange
                    badge = "⚠️ Average"
                else:
                    color = "#dc3545"  # Red
                    badge = "❌ Poor"
                
                st.markdown(f"""
                <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);">
                    <h4 style="margin: 0; color: {color};">#{idx}. {name} - {badge}</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                        <div><strong>📧 Email:</strong> {email}</div>
                        <div><strong>📊 Score:</strong> {score_val} ({numeric_score})</div>
                        <div><strong>⭐ Perfect Match:</strong> {is_perfect}</div>
                        <div><strong>👍 Good Match:</strong> {is_okay}</div>
                    </div>
                    <div style="margin-top: 10px;">
                        <div><strong>💪 Strong Areas:</strong></div>
                        <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin: 5px 0;">
                            {strong_zone}
                        </div>
                        <div><strong>📚 Areas for Improvement:</strong></div>
                        <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin: 5px 0;">
                            {lack_knowledge}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Remove the numeric_score column before displaying the main table
        df_display = df_sorted.drop('numeric_score', axis=1)
    else:
        df_display = df
    
    # Detailed results table
    st.markdown("**Detailed Results Table:**")
    
    # Add color coding to the score column
    if score_column in df_display.columns:
        def highlight_scores(val):
            try:
                if isinstance(val, str):
                    score = int(float(val.replace('%', '').strip()))
                elif isinstance(val, (int, float)):
                    score = int(val)
                else:
                    return ''
                
                if score >= 80:
                    return 'background-color: #d4edda'
                elif score >= 60:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            except (ValueError, TypeError):
                return ''
        
        try:
            styled_df = df_display.style.applymap(highlight_scores, subset=[score_column])
            st.dataframe(styled_df, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not apply color coding: {str(e)}")
            st.dataframe(df_display, use_container_width=True)
    else:
        st.dataframe(df_display, use_container_width=True)

def main():
    load_dotenv()
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Resume Screening System</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input("Enter LLAMA API Key:", type="password", 
                               value=os.getenv("LLAMA_API_KEY", ""))
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your API key")
        
        # Model selection
        model_option = st.selectbox(
            "Select Model:",
            ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
        )
        
        # Temperature setting
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.0, 0.1)
    
    if not api_key:
        st.error("Please configure your LLAMA API key in the sidebar to continue.")
        return
    
    try:
        llm = ChatGroq(
            model=model_option,
            api_key=api_key,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return
    
    # Main content
    tab1, tab2 = st.tabs(["📁 Resume Analysis", "📧 Email Notifications"])
    
    with tab1:
        # Enhanced folder selection
        selected_folder = UIManager.get_enhanced_folder_browser()
        
        # Job description input
        st.markdown('<div class="section-header">📝 Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Enter the job description for comparison:",
            height=200,
            placeholder="Paste the job description here..."
        )
        
        # Process button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not selected_folder:
                st.error("❌ Please select a folder containing resume PDFs")
                return
            
            if not job_description.strip():
                st.error("❌ Please enter a job description")
                return
            
            if not os.path.exists(selected_folder):
                st.error("❌ Selected folder does not exist")
                return
            
            # Find PDF files
            pdf_files = [f for f in os.listdir(selected_folder) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                st.error("❌ No PDF files found in the selected folder")
                return
            
            st.markdown(f'<div class="info-box">📄 Found {len(pdf_files)} resume(s) to process</div>', 
                       unsafe_allow_html=True)
            
            # Processing
            information_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, pdf_file in enumerate(pdf_files):
                try:
                    status_text.text(f'🔄 Processing {pdf_file}... ({idx + 1}/{len(pdf_files)})')
                    progress_bar.progress((idx + 1) / len(pdf_files))
                    
                    pdf_path = os.path.join(selected_folder, pdf_file)
                    resume = load_pdf(pdf_path)
                    
                    if not resume.strip():
                        st.warning(f"⚠️ Could not extract text from {pdf_file}")
                        continue
                    
                    # Analysis
                    with st.spinner(f"Analyzing {pdf_file}..."):
                        checker = resume_checker(resume=resume, job_description=job_description, llm=llm)
                        data = extract_json_from_response(checker)
                        information_list.append(data)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    st.error(f"❌ Error processing {pdf_file}: {str(e)}")
                    information_list.append({
                        "Name": f"Error: {pdf_file}",
                        "email": "error@email.com", 
                        "is_perfect": False,
                        "is_okay": False,
                        "Matching Score in percentage": "0%",
                        "strong zone": "Processing failed",
                        "Lack of Knowledge": f"Error: {str(e)}"
                    })
            
            # Results
            if information_list:
                df = pd.DataFrame(information_list)
                df.to_csv("resume_screening.csv", index=False)
                
                st.markdown('<div class="success-box">✅ Analysis completed successfully!</div>', 
                           unsafe_allow_html=True)
                
                display_results_dashboard(df)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="resume_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("❌ No resumes were successfully processed")
    
    with tab2:
        st.markdown('<div class="section-header">📧 Email Notification System</div>', unsafe_allow_html=True)
        
        # Check if results file exists
        if os.path.exists("resume_screening.csv"):
            df = pd.read_csv("resume_screening.csv")
            
            st.markdown('<div class="info-box">📊 Results file found! Ready to send notifications.</div>', 
                       unsafe_allow_html=True)
            
            # Preview
            st.markdown("**Preview of candidates to be notified:**")
            
            # Find score column for preview
            score_column = None
            possible_score_columns = [
                'Matching Score in percentage', 
                'Matching Score', 
                'Score', 
                'matching_score', 
                'score'
            ]
            
            for col in possible_score_columns:
                if col in df.columns:
                    score_column = col
                    break
            
            if not score_column:
                for col in df.columns:
                    if 'score' in col.lower():
                        score_column = col
                        break
            
            # Show preview with available columns
            preview_columns = ['Name', 'email']
            if score_column:
                preview_columns.append(score_column)
            
            available_preview_columns = [col for col in preview_columns if col in df.columns]
            st.dataframe(df[available_preview_columns], use_container_width=True)
            
            # Send emails
            if st.button("📤 Send Email Notifications", type="primary", use_container_width=True):
                try:
                    with st.spinner("Sending emails..."):
                        send_mail("resume_screening.csv")
                    st.markdown('<div class="success-box">✅ All emails have been sent successfully!</div>', 
                               unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error sending emails: {str(e)}")
        else:
            st.markdown('<div class="warning-box">⚠️ No analysis results found. Please run the resume analysis first.</div>', 
                       unsafe_allow_html=True)

if __name__ == "__main__":
    main()