import streamlit as st
import pandas as pd
import os
from PIL import Image

def load_data():
    # Load metadata
    metadata = pd.read_csv('Data/metadata.csv')
    
    # Load LLM responses
    gemini_df = pd.read_csv('Data/Gemini/llm_responses_combined.csv')
    gpt4_df = pd.read_csv('Data/GPT-4o/llm_responses_combined.csv')
    llama_df = pd.read_csv('Data/llama-3.2/llm_responses_combined.csv')
    
    return metadata, gemini_df, gpt4_df, llama_df

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def main():
    st.title("LLM Response Analysis Dashboard")
    
    # Load data
    metadata, gemini_df, gpt4_df, llama_df = load_data()
    
    # Image selector at the top
    selected_id = st.selectbox("Select Image ID", metadata['id'].unique())
    
    # Get image path and data
    image_name = metadata[metadata['id'] == selected_id]['image_name'].iloc[0]
    image_path = os.path.join("Data/images", image_name)
    
    # Get original title and body
    original_data = gemini_df[gemini_df['Id'] == selected_id].iloc[0]
    original_title = original_data['Title']
    original_body = original_data['Body']
    
    # Display image in the center
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = load_image(image_path)
        if image:
            st.image(image, caption=f"Image ID: {selected_id}", use_column_width=True)
        else:
            st.error(f"Could not load image: {image_path}")
    
    # Original content
    st.subheader("Original Content")
    st.write("**Title:**", original_title)
    st.write("**Body:**", original_body)
    
    # Create tabs for different approaches
    tabs = st.tabs(["Zero-shot", "Few-shot", "Chain-of-Thought"])
    
    with tabs[0]:  # Zero-shot
        st.subheader("Zero-shot Responses")
        for df, model in zip([gemini_df, gpt4_df, llama_df], ["Gemini", "GPT-4", "LLaMA"]):
            with st.expander(f"{model} Responses"):
                filtered_df = df[df['Id'] == selected_id]
                if not filtered_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Title:**")
                        st.write(filtered_df['llm_zero_shot_title'].iloc[0])
                    with col2:
                        st.write("**Body:**")
                        st.write(filtered_df['llm_zero_shot_body'].iloc[0])
                    st.write("**Combined:**")
                    st.write(filtered_df['llm_zero_shot_combined'].iloc[0])
    
    with tabs[1]:  # Few-shot
        st.subheader("Few-shot Responses")
        for df, model in zip([gemini_df, gpt4_df, llama_df], ["Gemini", "GPT-4", "LLaMA"]):
            with st.expander(f"{model} Responses"):
                filtered_df = df[df['Id'] == selected_id]
                if not filtered_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Title:**")
                        st.write(filtered_df['llm_few_shot_title'].iloc[0])
                    with col2:
                        st.write("**Body:**")
                        st.write(filtered_df['llm_few_shot_body'].iloc[0])
                    st.write("**Combined:**")
                    st.write(filtered_df['llm_few_shot_combined'].iloc[0])
    
    with tabs[2]:  # Chain-of-Thought
        st.subheader("Chain-of-Thought Responses")
        for df, model in zip([gemini_df, gpt4_df, llama_df], ["Gemini", "GPT-4", "LLaMA"]):
            with st.expander(f"{model} Responses"):
                filtered_df = df[df['Id'] == selected_id]
                if not filtered_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Title:**")
                        st.write(filtered_df['llm_cot_title'].iloc[0])
                    with col2:
                        st.write("**Body:**")
                        st.write(filtered_df['llm_cot_body'].iloc[0])
                    st.write("**Combined:**")
                    st.write(filtered_df['llm_cot_combined'].iloc[0])

if __name__ == "__main__":
    main()