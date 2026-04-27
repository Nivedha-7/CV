import os
import time
import requests
import streamlit as st
 
st.set_page_config(page_title="AI Detection System", layout="wide")
 
# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = 1
 
if "use_case" not in st.session_state:
    st.session_state.use_case = ""
 
if "problem_definition" not in st.session_state:
    st.session_state.problem_definition = ""
 
if "image_count" not in st.session_state:
    st.session_state.image_count = 1000
 
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
 
if "mode" not in st.session_state:
    st.session_state.mode = "Auto"
 
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
 
if "task_type" not in st.session_state:
    st.session_state.task_type = "Both"
 
if "phase" not in st.session_state:
    st.session_state.phase = "Inference"
 
# =========================
# NAVIGATION
# =========================
def go_next():
    if st.session_state.page < 6:
        st.session_state.page += 1
    st.rerun()
 
def go_back():
    if st.session_state.page > 1:
        st.session_state.page -= 1
    st.rerun()
 
# =========================
# HELPERS
# =========================
def get_dynamic_title(use_case):
    titles = {
        "Crack Detection in Manufacturing Unit": "Manufacturing Crack Detection & Analysis",
        "Surface Defect Detection": "Surface Defect Detection & Quality Analysis",
        "Predictive Maintenance": "Predictive Maintenance & Fault Detection",
        "Quality Inspection from CCTV Images": "CCTV-Based Quality Inspection System",
    }
    return titles.get(use_case, "AI Detection System")
 
def get_auto_model(task_type):
    if task_type == "Classification":
        return "ResNet18"
    elif task_type == "Segmentation":
        return "U-Net"
    return "ResNet18 + U-Net"
 
def get_message(label):
    if label == "Low":
        return """Severity of the crack detected is LOW.
 
A minor surface-level crack has been identified. This type of defect does not currently impact structural integrity or operational performance.
 
Recommendation:
No immediate action is required. However, it is advised to monitor the affected area during routine inspections to ensure the crack does not propagate over time."""
    elif label == "Medium":
        return """Severity of the crack detected is MEDIUM.
 
A noticeable structural crack has been detected, which may worsen if left unaddressed. This indicates a moderate level of risk.
 
Recommendation:
It is recommended to schedule maintenance and perform a detailed inspection within the next few days to prevent further deterioration and avoid potential impact on performance."""
    elif label == "High":
        return """Severity of the crack detected is HIGH.
 
A critical structural defect has been identified. This level of damage may significantly impact safety, reliability, and overall system performance.
 
Recommendation:
Immediate attention is required. It is strongly recommended to repair or replace the affected component within a week to prevent failure or further damage."""
    return "Unable to determine crack severity."
 
def show_workflow():
    current_step = st.session_state.page
    steps = [
        "Use Case",
        "Requirements",
        "Upload Data",
        "Model Selection",
        "Execution",
        "Output"
    ]
    cols = st.columns(len(steps))
    for i, step in enumerate(steps, start=1):
        with cols[i - 1]:
            if i < current_step:
                st.success(f"✅ {step}")
            elif i == current_step:
                st.warning(f"🔵 {step}")
            else:
                st.info(f"⬜ {step}")
 
# =========================
# HEADER
# =========================
title = get_dynamic_title(st.session_state.use_case) if st.session_state.use_case else "AI Detection System"
st.title(title)
show_workflow()
st.markdown("---")
 
# =========================
# PAGE 1 — USE CASE
# =========================
if st.session_state.page == 1:
    st.subheader("Use Case Definition")
    st.write("Please select a use case and describe your problem below to get started.")
 
    use_case = st.selectbox(
        "Select use case",
        [
            "",
            "Crack Detection in Manufacturing Unit",
            "Surface Defect Detection",
            "Predictive Maintenance",
            "Quality Inspection from CCTV Images"
        ],
        index=0 if not st.session_state.use_case else [
            "",
            "Crack Detection in Manufacturing Unit",
            "Surface Defect Detection",
            "Predictive Maintenance",
            "Quality Inspection from CCTV Images"
        ].index(st.session_state.use_case),
        format_func=lambda x: "-- Select a use case --" if x == "" else x
    )
 
    problem_definition = st.text_area(
        "Problem definition",
        value=st.session_state.problem_definition,
        placeholder="Describe your problem here. For example: I have images captured from CCTV in a manufacturing unit. I need to automatically detect cracks and understand their severity.",
        height=150
    )
 
    if st.button("Next"):
        if not use_case:
            st.error("Please select a use case before proceeding.")
        elif not problem_definition.strip():
            st.error("Please enter a problem definition before proceeding.")
        else:
            st.session_state.use_case = use_case
            st.session_state.problem_definition = problem_definition
            go_next()
 
# =========================
# PAGE 2 — REQUIREMENTS
# =========================
elif st.session_state.page == 2:
    st.subheader("Requirement Clarification")
 
    st.info(f"""
Use case selected: {st.session_state.use_case}
 
Problem summary:
The system should analyze input images, identify visible damage patterns, estimate severity, and provide predictive guidance for correction or maintenance.
 
Recommended tasks:
- Segmentation → detect exact defect region
- Classification → detect severity level
 
Recommended image count:
- Minimum: 500–1000 images
- Better: 3000+ images
- Strong model: 5000+ images
""")
 
    col1, col2 = st.columns(2)
 
    with col1:
        task_type = st.selectbox(
            "Select task type",
            ["Classification", "Segmentation", "Both"],
            index=["Classification", "Segmentation", "Both"].index(st.session_state.task_type)
        )
 
    with col2:
        image_count = st.number_input(
            "Number of images available",
            min_value=0,
            max_value=50000,
            value=st.session_state.image_count,
            step=100
        )
 
    if image_count < 500:
        st.warning("The dataset size is currently low. It is recommended to collect more images for better model performance.")
    elif image_count < 3000:
        st.info("The dataset size is sufficient for initial testing and demo purposes.")
    else:
        st.success("The dataset size is suitable for building a strong and reliable model.")
 
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go_back()
    with c2:
        if st.button("Next"):
            st.session_state.task_type = task_type
            st.session_state.image_count = image_count
            go_next()
 
# =========================
# PAGE 3 — UPLOAD
# =========================
elif st.session_state.page == 3:
    st.subheader("Upload Data")
 
    uploaded_files = st.file_uploader(
        "Upload dataset files",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "csv", "zip"]
    )
 
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} file(s) have been successfully uploaded and are ready for processing.")
    else:
        st.info("No files have been uploaded yet.")
 
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go_back()
    with c2:
        if st.button("Next"):
            go_next()
 
# =========================
# PAGE 4 — MODEL SELECTION
# =========================
elif st.session_state.page == 4:
    st.subheader("Model Selection")
 
    auto_mode = st.toggle("Enable Auto Mode", value=(st.session_state.mode == "Auto"))
 
    if auto_mode:
        mode = "Auto"
        selected_model = get_auto_model(st.session_state.task_type)
        st.info(f"Auto mode has been selected. The system has automatically chosen the most suitable model: **{selected_model}**.")
    else:
        mode = "Manual"
 
        if st.session_state.task_type == "Classification":
            selected_model = st.selectbox(
                "Choose classification model",
                ["ResNet18", "EfficientNet", "MobileNet"]
            )
        elif st.session_state.task_type == "Segmentation":
            selected_model = st.selectbox(
                "Choose segmentation model",
                ["U-Net", "DeepLabV3", "FCN"]
            )
        else:
            selected_model = st.selectbox(
                "Choose workflow",
                ["ResNet18 + U-Net", "EfficientNet + U-Net", "MobileNet + DeepLabV3"]
            )
 
        st.info(f"Manual mode has been selected. The chosen model or workflow is: **{selected_model}**.")
 
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go_back()
    with c2:
        if st.button("Next"):
            st.session_state.mode = mode
            st.session_state.selected_model = selected_model
            go_next()
 
# =========================
# PAGE 5 — EXECUTION
# =========================
elif st.session_state.page == 5:
    st.subheader("Execution")
 
    phase = st.radio(
        "Choose phase",
        ["Train / Test", "Inference"],
        index=0 if st.session_state.phase == "Train / Test" else 1
    )
 
    if st.button("Run"):
        st.session_state.phase = phase
 
        st.info(
            f"{st.session_state.mode} mode has been selected. "
            f"The system is now executing the **{st.session_state.selected_model}** model "
            f"for the selected task. Please wait while processing is in progress."
        )
 
        if phase == "Train / Test":
            # Trigger real training via backend
            with st.spinner("Triggering training..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/train")
                    if response.status_code == 200:
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
                            progress.progress(i + 1)
                        st.success("Model training started successfully in background.")
                        st.write("Training is running. Use the Predictive Output page to test predictions once done.")
                        st.write("Monitor training at: http://127.0.0.1:8000/status")
                    elif response.status_code == 409:
                        st.warning("Training is already running. Please wait for it to complete.")
                    else:
                        st.error("Training trigger failed. Please check backend connection.")
                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")
        else:
            if not st.session_state.uploaded_files:
                st.warning("No images uploaded. Please go back to Upload Data Page")
            else:
                total = len(st.session_state.uploaded_files)
                st.write(f"Running inference on **{total}** images(s).")
                progress = st.progress(0)
                for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
                    try:
                        files = {
                            "file":(uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        response = requests.post("http://127.0.0.1:8000/predict", files=files)
                        if response.status_code == 200:
                            result = response.json()
                            st.markdown("---")
                            st.success(f"Prediction Completeted for: **{result['filename']}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Severity:** {result['severity']}**")
                            with col2:
                                st.write(f"**Confidence:** {result['confidence']}**")
                            st.warning(result["message"])
                        else:
                            st.error(f"Prediction failed for {uploaded_file.name}.")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    progress.progress(int((idx + 1) / total * 100))
                    st.markdown("---")
                    st.success(f"Inference completed for all {total} image(s).")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back"):
                go_back()
        with c2:
            if st.button("Next"):
                go_next()
 
# =========================
# PAGE 6 — PREDICTIVE OUTPUT
# =========================
elif st.session_state.page == 6:
    st.subheader("Output")
   
    if not st.session_state.uploaded_files:
        st.warning("Please upload data before viewing predictive results")
        if st.button("Back"):
            go_back()
        st.stop()
   
    if st.button("Run Prediction"):
        total = len(st.session_state.uploaded_files)
        st.write(f"Running inference on **{total}** images(s).")
        progress = st.progress(0)
        for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
            try:
                files = {
                    "file":(uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.markdown("---")
                    st.success(f"Prediction Completeted for: **{result['filename']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Severity:** {result['severity']}**")
                    with col2:
                        st.write(f"**Confidence:** {result['confidence']}**")
                    st.warning(result["message"])
                else:
                    st.error(f"Prediction failed for {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            progress.progress(int((idx + 1) / total * 100))
        st.markdown("---")
        st.success(f"All predictions completed for  {total} image(s).")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go_back()
    with c2:
        if st.button("Restart"):
            for key in ["page", "use_case", "problem_definition", "image_count", "uploaded_files", "mode", "selected_model", "task_type", "phase"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()