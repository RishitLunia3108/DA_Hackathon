import streamlit as st  
import pandas as pd  
from sklearn.metrics import f1_score, confusion_matrix, classification_report  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score  
import os  
from datetime import datetime  

# File paths  
TRUTH_FILE_PATH = r"D:\Hackathon\test_ans.csv"  # Replace with your actual path  
RESULTS_FILE_PATH = r"D:\Hackathon\f1_scores.csv"  # Path for saving results  

# Streamlit page configuration  
st.set_page_config(page_title="Hackathon", layout="wide")  
st.title("🎯 **DA Hackathon**")  

st.write("""  
### ⚠️ **Critical Note**:  
- **Do not change the order of the columns or rows** in the file you are uploading; it should be the same as the `train.csv` provided to you.  
- You only need to add a new column named `y` with your predictions to the provided file structure.  
- Any changes to the structure (e.g., reordering rows, renaming columns, or modifying existing data) will result in incorrect calculations.  
""")  

st.write("""  
### ⚠️ **Important Instructions**:  
1. The **prediction file** you are uploading must contain a column named `y`, which is the predicted column.  

Let's get started! 🚀  
""")  

# Function to save results to a CSV file  
def save_results(pred_filename, f1_score):  
    # Remove extensions from filenames  
    pred_file_name = pred_filename.replace(".csv", "")  
    
    # Create results DataFrame or load existing one  
    if os.path.exists(RESULTS_FILE_PATH):  
        results_df = pd.read_csv(RESULTS_FILE_PATH)  
    else:  
        results_df = pd.DataFrame(columns=['Prediction_File', 'F1_Score', 'Timestamp'])  
    
    # Add new result  
    new_result = pd.DataFrame({  
        'Prediction_File': [pred_file_name],  
        'F1_Score': [f1_score],  
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]  
    })  
    
    # Concatenate the new result  
    results_df = pd.concat([results_df, new_result], ignore_index=True)  
    
    # Keep only the highest F1 score for each team  
    results_df = results_df.sort_values(by=['Prediction_File', 'F1_Score'], ascending=[True, False])  
    results_df = results_df.drop_duplicates(subset=['Prediction_File'], keep='first')  
    
    # Sort by F1 Score to assign ranks  
    results_df = results_df.sort_values(by='F1_Score', ascending=False).reset_index(drop=True)  
    results_df['Rank'] = np.arange(1, len(results_df) + 1)  # Add Rank column  
    
    # Save to CSV  
    results_df.to_csv(RESULTS_FILE_PATH, index=False)  
    return results_df  

# Load ground truth data  
try:  
    truth_df = pd.read_csv(TRUTH_FILE_PATH)  
    if 'y' not in truth_df.columns:  
        st.error("❌ **Error**: Ground truth file must contain a column named `y`.")  
        st.stop()  
except Exception as e:  
    st.error(f"❌ **Error reading ground truth file**: {str(e)}")  
    st.stop()  

# File uploader for prediction CSV  
st.write("### 📂 **Upload Your Files**")  
pred_file = st.file_uploader("1️⃣ Upload your **Prediction CSV** file:", type=['csv'])  

# Add submit button  
submit_button = st.button("🚀 **Submit and Calculate F1 Score**")  

if submit_button:  
    # Ensure the file is uploaded  
    if not pred_file:  
        st.warning("⚠️ **Please upload your prediction CSV file.**")  
    else:  
        try:  
            # Read the prediction CSV file  
            pred_df = pd.read_csv(pred_file)  
            if 'y' not in pred_df.columns:  
                st.error("❌ **Error**: Prediction CSV file must contain a column named `y`.")  
            else:  
                # Display sample of prediction data only  
                st.write("### 🔍 **Preview of Prediction Data**")  
                st.write(pred_df.head())  
                
                try:  
                    # Calculate metrics  
                    f1 = f1_score(truth_df['y'], pred_df['y'], average='weighted') * 100  # Convert to percentage  
                    
                    # Save results to CSV  
                    results_df = save_results(pred_file.name, f1)  
                    
                    # Display metrics  
                    st.write("### 📊 **Results**")  
                    col1, col2 = st.columns(2)  
                    
                    with col1:  
                        st.metric("🎯 **F1 Score (%)**", f"{f1:.2f}%")  # Display F1 Score as percentage  
                    
                    with col2:  
                        accuracy = accuracy_score(truth_df['y'], pred_df['y']) * 100  # Convert to percentage  
                        st.metric("📈 **Accuracy (%)**", f"{accuracy:.2f}%")  
                    
                    # Confusion Matrix Visualization  
                    st.write("### 🔢 **Confusion Matrix**")  
                    cm = confusion_matrix(truth_df['y'], pred_df['y'])  
                    fig, ax = plt.subplots(figsize=(8, 6))  
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)  
                    plt.title('Confusion Matrix')  
                    plt.ylabel('True Label')  
                    plt.xlabel('Predicted Label')  
                    st.pyplot(fig)  
                    
                    # Detailed Classification Report  
                    st.write("### 📋 **Detailed Classification Report**")  
                    report = classification_report(truth_df['y'], pred_df['y'], output_dict=True)  
                    report_df = pd.DataFrame(report).drop('accuracy', axis=1).T  
                    report_df = report_df.round(4)  
                    st.dataframe(report_df)  
                     
                
                except Exception as e:  
                    st.error(f"❌ **Error calculating metrics**: {str(e)}")  
                    st.write("Please ensure the `y` column contains valid categorical or numerical data.")  
        
        except Exception as e:  
            st.error(f"❌ **Error reading prediction CSV file**: {str(e)}")  
            st.write("Please ensure your CSV file is properly formatted.")