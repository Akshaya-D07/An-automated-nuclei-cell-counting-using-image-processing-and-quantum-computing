
import streamlit as st

st.set_page_config(
    page_title=" Histological Nuclei Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from scipy import ndimage
from skimage.morphology import disk
import os

from NucleiCellsDetection import (
    visualize_connected_components,
    count_nuclei_cells,
    connected_components_labeling
)
from HistologicalImageProcessing import HistologicalImageProcessor

from MorphologicalProcessing import (
    apply_proposed_morphological_approach,
    visualize_morphological_steps
)
from QuantamcCustering import QuantumNucleiCounter,visualize_quantum_nuclei_analysis,QuantumNucleiFeatureEncoder


def create_download_link(fig, filename="analysis.png"):
    """Create download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily and return path"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def main():
    # Header
    st.title(" Histological Nuclei Analysis System")
     

    st.sidebar.header(" Analysis Configuration")
    
    overlap_threshold = st.sidebar.slider(
        "Quantum Overlap Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Threshold for quantum overlap detection"
    )
    
    threshold_value = st.sidebar.slider(
        "Binary Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Threshold for binary image conversion"
    )
    
    min_area = st.sidebar.number_input(
        "Minimum Nucleus Area",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Minimum area for valid nuclei (pixels)"
    )
    
    max_area = st.sidebar.number_input(
        "Maximum Nucleus Area",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=100,
        help="Maximum area for valid nuclei (pixels)"
    )
    
    connectivity = st.sidebar.selectbox(
        "Connectivity Type",
        options=[4, 8],
        index=1,
        help="Pixel connectivity for component analysis"
    )
    
    # File uploader
    st.header("Upload Histological Image")
    uploaded_file = st.file_uploader(
        "Choose a histological image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an H&E stained histological image for nuclei analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = save_uploaded_file(uploaded_file)
        
        
        st.subheader("Original Histological Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded histological image", use_column_width=True)
        
        # Start analysis button
        if st.button(" Start Complete Analysis Pipeline", type="primary"):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize processors
                status_text.text(" Initializing image processor and quantum counter...")
                processor = HistologicalImageProcessor(temp_file_path)
                quantum_counter = QuantumNucleiCounter(overlap_threshold)
                progress_bar.progress(10)
                
                # Step 2: Load image
                status_text.text(" Loading histological image...")
                if not processor.load_image():
                    st.error(" Failed to load image. Please check the image format.")
                    return
                progress_bar.progress(20)
                
                # Step 3: Extract color channels
                status_text.text(" Extracting RGB color channels...")
                if not processor.extract_color_channels():
                    st.error(" Failed to extract color channels.")
                    return
                progress_bar.progress(30)
                
                # Step 4: Find optimal channel
                status_text.text(" Analyzing channels and finding optimal one...")
                optimal_channel, stats = processor.find_optimal_channel()
                progress_bar.progress(40)
                
                # Display channel analysis results
                st.header(" Color Channel Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(" Channel Statistics")
                    if stats:
                        stats_df = pd.DataFrame(stats).T
                        st.dataframe(stats_df, use_container_width=True)
                    
                    st.success(f" **Optimal Channel**: {optimal_channel}")
                
                with col2:
                    st.subheader(" Channel Visualization")
                    # Generate and display channel comparison
                    processor.visualize_channels(save_path='temp_channel_comparison.png')
                    if os.path.exists('temp_channel_comparison.png'):
                        channel_img = Image.open('temp_channel_comparison.png')
                        st.image(channel_img, caption="RGB Channel Comparison", use_column_width=True)
                
                progress_bar.progress(50)
                
                # Step 5: Process optimal channel
                if optimal_channel:
                    status_text.text(" Processing optimal channel for nuclei segmentation...")
                    optimal_image = processor.get_optimal_channel_image(optimal_channel.lower())
                    
                    st.header(" Image Processing Pipeline")
                    
                    # Display histogram and thresholding
                    st.subheader(" Histogram Analysis and Binary Thresholding")
                    processor.create_histogram_visualization(optimal_image, threshold_value, f"{optimal_channel} Channel")
                    if os.path.exists('histogram_analysis.png'):
                            hist_img = Image.open('histogram_analysis.png')
                            st.image(hist_img, caption=f"{optimal_channel} Channel Histogram", use_column_width=True)
                 
                    progress_bar.progress(60)
                    final_binary = processor.apply_threshold(optimal_image, threshold_value)
                    
                    # Step 6: Morphological processing
                    status_text.text(" Applying morphological operations...")
                    results = apply_proposed_morphological_approach(final_binary)
                    final_nuclei_segmentation = results['final']
                    
                    st.subheader(" Morphological Processing Steps")
                    visualize_morphological_steps(results,'morphological_steps.png')
                    if os.path.exists('morphological_steps.png'):
                        morph_img = Image.open('morphological_steps.png')
                        st.image(morph_img, caption="Morphological Processing Pipeline", use_column_width=True)
                    
                    progress_bar.progress(70)
                    
                    # Step 7: Connected component analysis
                    status_text.text(" Performing connected component analysis...")
                    counting_results = count_nuclei_cells(
                        final_nuclei_segmentation,
                        connectivity=connectivity,
                        min_area=min_area,
                        max_area=max_area
                    )
                  
                    st.header(" Nuclei Counting Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Components", counting_results.get('total_components', 0))
                    with col2:
                        st.metric("Valid Nuclei", counting_results.get('nuclei_count', 0))
                    with col3:
                        st.metric("Filtered Out", counting_results.get('invalid_components', 0))
                    with col4:
                        st.metric("Average Area", f"{counting_results.get('nuclei_stats', 0):.1f}")
                    
                    # Visualize connected components
                    visualize_connected_components(
                        counting_results['labeled_image'],
                        original_image=final_nuclei_segmentation,
                        save_path='temp_connected_components.png'
                    )
                    
                    if os.path.exists('temp_connected_components.png'):
                        cc_img = Image.open('temp_connected_components.png')
                        st.image(cc_img, caption="Connected Components Analysis", use_column_width=True)
                    
                    progress_bar.progress(80)
                    
                    # Step 8: Quantum analysis
                    status_text.text(" Performing quantum-enhanced analysis...")
                    labeled_image = counting_results['labeled_image']
                    quantum_results = quantum_counter.perform_quantum_counting(labeled_image,counting_results.get('nuclei_count', 0), optimal_image)
                    
                    st.header(" Quantum-Enhanced Analysis Results")
                    
                    # Display quantum metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Initial Count",
                            value=quantum_results['initial_count'],
                            help="Standard segmentation count"
                        )
                    
                    with col2:
                        st.metric(
                            label="Detected Overlaps",
                            value=quantum_results['detected_overlaps'],
                            help="Overlapping nuclei pairs found"
                        )
                    
                    with col3:
                        st.metric(
                            label="Hidden Nuclei",
                            value=quantum_results['hidden_nuclei_estimate'],
                            delta=f"+{quantum_results['hidden_nuclei_estimate']}",
                            help="Estimated hidden nuclei"
                        )
                    
                    with col4:
                        st.metric(
                            label="Quantum Count",
                            value=quantum_results['quantum_corrected_count'],
                            delta=f"{quantum_results['correction_factor']:.3f}x",
                            help="Final quantum-corrected count"
                        )
                    
                    progress_bar.progress(90)
                    
                    # Step 9: Final visualization
                    status_text.text(" Generating comprehensive visualization...")
                    visualize_quantum_nuclei_analysis(
                        quantum_results,
                        labeled_image,
                        save_path='temp_quantum_analysis.png'
                    )
                    
                    if os.path.exists('temp_quantum_analysis.png'):
                        quantum_img = Image.open('temp_quantum_analysis.png')
                        st.image(quantum_img, caption="Complete Quantum Nuclei Analysis", use_column_width=True)
                    
                    progress_bar.progress(100)
                    status_text.text(" Analysis completed successfully!")
                    
                    # Advanced analysis tabs
                    st.header(" Detailed Analysis Results")
                    
                    tab1, tab2, tab3 = st.tabs([" Overlap Details", " Statistics", " Quantum Metrics"])
                    
                    with tab1:
                        overlap_results = quantum_results['overlap_results']
                        if overlap_results['overlapping_pairs']:
                            st.subheader("Overlapping Nuclei Pairs")
                            overlap_data = []
                            for pair in overlap_results['overlapping_pairs']:
                                overlap_data.append({
                                    'Nucleus 1': pair['nucleus_1'],
                                    'Nucleus 2': pair['nucleus_2'],
                                    'Similarity': f"{pair['similarity']:.4f}",
                                    'Confidence': f"{pair['overlap_confidence']:.4f}"
                                })
                            
                            overlap_df = pd.DataFrame(overlap_data)
                            st.dataframe(overlap_df, use_container_width=True)
                        else:
                            st.info("No overlapping nuclei detected with current threshold")
                    
                    with tab2:
                        st.subheader("Comprehensive Statistics")
                        
                        # Create statistics summary
                        stats_data = {
                            "Metric": [
                                "Image Dimensions",
                                "Optimal Channel",
                                "Threshold Value",
                                "Initial Components",
                                "Valid Nuclei (Classical)",
                                "Quantum Corrected Count",
                                "Correction Factor",
                                "Average Nuclei Area",
                                "Processing Time"
                            ],
                            "Value": [
                                f"{optimal_image.shape[1]} x {optimal_image.shape[0]}",
                                optimal_channel,
                                f"{threshold_value:.2f}",
                                counting_results.get('total_components', 0),
                                counting_results.get('valid_nuclei', 0),
                                quantum_results['quantum_corrected_count'],
                                f"{quantum_results['correction_factor']:.4f}",
                                f"{counting_results.get('average_area', 0):.2f} px²",
                                "< 30 seconds"
                            ]
                        }
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Quantum Enhancement Details")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(" **Quantum Processing Info**")
                            encoder = QuantumNucleiFeatureEncoder()
                            st.write(f"• Quantum system active: ")
                            st.write(f"• Qiskit available: {'' if encoder.use_qiskit else ''}")
                            st.write(f"• Overlap threshold: {overlap_threshold}")
                            st.write(f"• Detection method: Quantum fidelity")
                        
                        with col2:
                            st.info(" **Enhancement Results**")
                            improvement = quantum_results['quantum_corrected_count'] - quantum_results['initial_count']
                            improvement_pct = (improvement / quantum_results['initial_count'] * 100) if quantum_results['initial_count'] > 0 else 0
                            
                            st.write(f"• Additional nuclei found: {improvement}")
                            st.write(f"• Improvement percentage: {improvement_pct:.1f}%")
                            st.write(f"• Overlap pairs detected: {quantum_results['detected_overlaps']}")
                            st.write(f"• Avg quantum similarity: {quantum_results['overlap_results'].get('avg_similarity', 0):.4f}")
                    
                
                else:
                    st.error(" Could not determine optimal channel for analysis")
                
            except Exception as e:
                st.error(f" An error occurred during analysis: {str(e)}")
                st.exception(e)
            
            finally:
                # Cleanup temporary files
                for temp_file in ['temp_channel_comparison.png', 'temp_connected_components.png', 
                                 'temp_quantum_analysis.png', 'histogram_analysis.png', 
                                 'morphological_steps.png']:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                
                # Remove temporary upload
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.header(" About")
    st.sidebar.markdown("""
    **Complete Pipeline:**
    1.  RGB channel analysis
    2.  Optimal channel selection
    3. Binary thresholding
    4.  Morphological operations
    5.  Connected components
    6.  Quantum enhancement
    
    **Key Features:**
    • Automatic channel optimization
    • Advanced morphological processing
    • Quantum overlap detection
    • Hidden nuclei estimation
    """)
    
  
   
if __name__ == "__main__":
    main()