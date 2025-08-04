import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import pandas as pd
import time

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .custom-link {
        color: #1f77b4;
        font-weight: bold;
        text-decoration: none;
        margin-left: 0.5em;
    }
    .footer {
        text-align: center;
        color: gray;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

CLASS_NAMES = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary', 3: 'No Tumor'}
DETECTION_CLASSES = [0, 1, 2]

def create_square_thumbnail(img, size=200, background_color=(0, 0, 0)):
    img_copy = img.copy()
    img_copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    thumb = Image.new('RGB', (size, size), background_color)
    pos = ((size - img_copy.width) // 2, (size - img_copy.height) // 2)
    thumb.paste(img_copy, pos)
    return thumb

def process_image(image, conf_threshold):
    if model is None:
        return None, None
    
    results = model(image, conf=conf_threshold)
    
    # Get detections
    detections = []
    has_detections = False
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            
            if class_id in DETECTION_CLASSES:  # Only show bounding boxes for tumor classes
                has_detections = True
                bbox = box.xyxy.cpu().numpy()[0]
                detections.append({
                    'class': CLASS_NAMES[class_id],
                    'confidence': confidence,
                    'bbox': bbox
                })
    
    # Draw bounding boxes on image
    annotated_image = image.copy()
    if has_detections:
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            cv2.putText(annotated_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image, detections, has_detections

def process_video(video_path, conf_threshold, progress_bar):
    if model is None:
        return None, None
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, detections, has_detections = process_image(frame, conf_threshold)
        
        # Store detections for statistics
        if has_detections:
            all_detections.extend(detections)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
    
    cap.release()
    out.release()
    
    return output_path, all_detections

def get_class_statistics(detections):
    if not detections:
        return pd.DataFrame()
    
    class_counts = {}
    class_confidences = {}
    
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    stats_data = []
    for class_name in class_counts:
        stats_data.append({
            'Tumor Type': class_name,
            'Count': class_counts[class_name],
            'Avg Confidence': np.mean(class_confidences[class_name]),
            'Max Confidence': np.max(class_confidences[class_name]),
            'Min Confidence': np.min(class_confidences[class_name])
        })
    
    return pd.DataFrame(stats_data)

def export_results_csv(detections, filename="detection_results.csv"):
    if not detections:
        return None
    
    export_data = []
    for i, detection in enumerate(detections):
        export_data.append({
            'Detection_ID': i + 1,
            'Tumor_Type': detection['class'],
            'Confidence': detection['confidence'],
            'BBox_X1': detection['bbox'][0],
            'BBox_Y1': detection['bbox'][1],
            'BBox_X2': detection['bbox'][2],
            'BBox_Y2': detection['bbox'][3]
        })
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False)

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")

    st.markdown(
        """
        <div style="margin-bottom:0.2em">
            <span>Check Out the other app:</span>
            <br>
            <a href="https://brain-tumor-image-classification-richardtanjaya.streamlit.app/" target="_blank" class="custom-link">
                Brain Tumor Image Classification
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05)

    st.markdown("---")

    st.subheader("📋 Model Information")
    st.success("""
    - **Model**: YOLOv8 (Custom Trained)  
    - **mAP50-95**: 0.792  
    - **mAP50**: 0.958  
    - **Recall**: 0.919
    """)

    st.markdown("---")

    st.subheader("🎯 Class Descriptions")
    st.info("""
    - **Glioma**: Tumor ganas di jaringan otak.
    - **Meningioma**: Tumor jinak di selaput otak.
    - **Pituitary**: Tumor di kelenjar pituitari/hipofisis.
    - **No Tumor**: Tidak ada tumor.
    """)

# === Main Content ===
st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Object Detection</h1>', unsafe_allow_html=True)
st.markdown("Upload an MRI image or video to detect and classify brain tumors using YOLOv8")

tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📹 Real-time Analysis"])

# === Tab 1: Image Analysis ===
with tab1:
    st.subheader("Choose an Image")

    if 'image_to_process' not in st.session_state:
        st.session_state.image_to_process = None
        st.session_state.image_file_name = None

    # --- Sample Images Section ---
    with st.container():
        st.write("Click on any sample image below to use it for prediction:")
        sample_files = {"Glioma Sample": "samples/Glioma.jpg", 
                        "Meningioma Sample": "samples/Meningioma.jpg", 
                        "Pituitary Sample": "samples/Pituitary.jpg", 
                        "No Tumor Sample": "samples/No_Tumor.jpg"}
        
        cols = st.columns(len(sample_files))
        for i, (name, path) in enumerate(sample_files.items()):
            with cols[i]:
                if os.path.exists(path):
                    sample_image = Image.open(path)
                    
                    # Create square thumbnail
                    display_thumbnail = create_square_thumbnail(sample_image)
                    st.image(display_thumbnail, caption=name, use_container_width=True)
                    if st.button(f"Use {name}", key=f"use_{name}", use_container_width=True):
                        # Set the original (full-sized) image for processing
                        st.session_state.image_to_process = sample_image
                        st.session_state.image_file_name = os.path.basename(path)
                        st.rerun()
                else:
                    st.error(f"{os.path.basename(path)} not found.")

    st.markdown("---")

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Upload your MRI image",
        type=['jpg', 'jpeg', 'png'],
        key="image_uploader"
    )

    if uploaded_file is not None:
        st.session_state.image_to_process = Image.open(uploaded_file)
        st.session_state.image_file_name = uploaded_file.name

    # --- Processing and Display Logic ---
    if st.session_state.image_to_process is not None:
        if st.button("❌ Clear Selection", use_container_width=True):
            st.session_state.image_to_process = None
            st.session_state.image_file_name = None
            st.rerun()
        
        with st.spinner("Processing image..."):
            image = st.session_state.image_to_process
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            annotated_image, detections, has_detections = process_image(image_np, conf_threshold)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Detection Results")
                if has_detections:
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, use_container_width=True)
                    
                    st.subheader("📊 Detection Statistics")
                    stats_df = get_class_statistics(detections)
                    st.dataframe(stats_df)
                    
                    st.subheader("💾 Download Results")
                    annotated_pil = Image.fromarray(annotated_image_rgb)
                    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    annotated_pil.save(img_buffer.name)
                    with open(img_buffer.name, 'rb') as file:
                        st.download_button(
                            label="📥 Download Annotated Image", data=file.read(),
                            file_name=f"annotated_{st.session_state.image_file_name}", mime="image/jpeg")
                    
                    csv_data = export_results_csv(detections)
                    if csv_data:
                        st.download_button(
                            label="📥 Download Detection Data (CSV)", data=csv_data,
                            file_name=f"results_{os.path.splitext(st.session_state.image_file_name)[0]}.csv", mime="text/csv")
                else:
                    st.image(image, use_container_width=True)
                    st.info("⚠️ **No tumor detected** (This indicates 'No Tumor' class)")

# === Tab 2: Video Analysis ===
with tab2:
    st.subheader("Choose a Video")

    if 'video_to_process' not in st.session_state:
        st.session_state.video_to_process = None
        st.session_state.video_file_name = None
        st.session_state.is_temp_video = False

    # --- Sample Video Section ---
    sample_video_path = "samples/Video.mp4"
    if os.path.exists(sample_video_path):
        if st.button("🎬 Use Sample Video", use_container_width=True):
            st.session_state.video_to_process = sample_video_path
            st.session_state.video_file_name = os.path.basename(sample_video_path)
            st.session_state.is_temp_video = False
            st.rerun()

    st.markdown("---")

    # --- Video Uploader ---
    uploaded_video = st.file_uploader(
        "Upload your MRI video",
        type=['mp4', 'mov', 'mkv'],
        key="video_uploader"
    )

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            st.session_state.video_to_process = tmp_file.name
        st.session_state.video_file_name = uploaded_video.name
        st.session_state.is_temp_video = True
    
    # --- Processing and Display Logic ---
    if st.session_state.video_to_process:
        video_path = st.session_state.video_to_process

        if st.button("❌ Clear Selection", use_container_width=True, key="clear_video"):
            if st.session_state.is_temp_video and os.path.exists(video_path):
                os.unlink(video_path)
            st.session_state.video_to_process = None
            st.session_state.video_file_name = None
            st.session_state.is_temp_video = False
            st.rerun()
            
        st.subheader("Selected Video")
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration > 30:
            st.error("❌ Video is too long! Please upload a video shorter than 30 seconds.")
        else:
            st.success(f"Video ready for processing (Duration: {duration:.1f}s)")
            if st.button("🎬 Process Video", key="process_video", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Processing video frames...")
                
                with st.spinner("Analyzing..."):
                    output_path, all_detections = process_video(video_path, conf_threshold, progress_bar)
                    
                    if output_path and os.path.exists(output_path):
                        status_text.success("✅ Video processing complete!")
                        
                        with open(output_path, 'rb') as video_file:
                            processed_video_bytes = video_file.read()
                        
                        if all_detections:
                            st.subheader("📊 Video Detection Statistics")
                            st.dataframe(get_class_statistics(all_detections))
                        else:
                            st.info("⚠️ **No tumors detected** in the video")
                        
                        st.subheader("💾 Download Results")
                        st.download_button(
                            label="📥 Download Processed Video", data=processed_video_bytes,
                            file_name=f"processed_{st.session_state.video_file_name}", mime="video/mp4")
                        
                        if all_detections:
                            csv_data = export_results_csv(all_detections)
                            if csv_data:
                                st.download_button(
                                    label="📥 Download Detection Data (CSV)", data=csv_data,
                                    file_name=f"results_{os.path.splitext(st.session_state.video_file_name)[0]}.csv", mime="text/csv")
                        os.unlink(output_path)
        
        # Clean up temp file from upload after processing
        if st.session_state.is_temp_video and os.path.exists(video_path):
            pass


# === Tab 3: Real-time Analysis ===
with tab3:
    st.subheader("Real-time Webcam Analysis")
    st.info("📹 Use your webcam to analyze MRI images in real-time")
    
    if st.button("📹 Start Webcam", key="start_webcam"):
        frame_placeholder = st.empty()
        detection_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check your camera permissions.")
        else:
            stop_button = st.button("⏹️ Stop Webcam", key="stop_webcam")
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame from webcam")
                    break
                
                annotated_frame, detections, has_detections = process_image(frame, conf_threshold)
                
                if annotated_frame is not None:
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    if has_detections:
                        detection_info = [f"**{d['class']}**: {d['confidence']:.2f}" for d in detections]
                        detection_placeholder.markdown("🎯 **Detections**: " + " | ".join(detection_info))
                    else:
                        detection_placeholder.info("⚠️ **No tumor detected**")
                
                time.sleep(0.1)
                
                if st.session_state.get('stop_webcam'):
                    break
            cap.release()

# === Footer ===
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Richard Dean Tanjaya | Github: <a href="https://github.com/RichardDeanTan/Brain-Tumor-Object-Detection" target="_blank">@RichardDeanTan</a></p>
</div>
""", unsafe_allow_html=True)