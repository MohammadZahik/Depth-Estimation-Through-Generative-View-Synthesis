# Depth-Estimation-Through-Generative-View-Synthesis

This project aims to develop an advanced stereo image processing and depth estimation system by leveraging a combination of classical computer vision techniques and modern deep learning models. The system utilizes stereo matching algorithms to generate disparity maps, which are then used for accurate depth estimation. The project also incorporates generative view synthesis to enhance the quality of the depth maps and stereo images.

**Key Features:**
Stereo Matching & Depth Estimation: The core functionality involves generating precise depth maps from stereo image pairs by utilizing stereo correspondence and matching techniques like block matching and epipolar geometry.

Generative View Synthesis: Advanced techniques like generative view synthesis are used to improve the quality and realism of depth maps, producing more accurate 3D reconstructions from 2D stereo pairs.

Deep Learning Integration: The system integrates deep learning models using PyTorch and TensorFlow for enhanced performance in depth prediction and image processing tasks. These models help improve the accuracy and efficiency of the depth estimation process, especially when handling complex scenes or large datasets.

Preprocessing and Alignment: To ensure the quality of the depth maps, preprocessing steps like image rectification, keypoint detection, and feature matching are performed. These steps help align the stereo images and correct distortions, ensuring accurate correspondence for depth estimation.

Interactive Web Interface: The project includes a Streamlit-based web application that allows users to interact with the system, upload stereo image pairs, visualize depth maps, and analyze results in real-time. This interface provides an intuitive way to explore the outputs of the system and test its performance on various stereo image datasets.

**Libraries & Frameworks Used:**

OpenCV: For image processing, feature extraction, and stereo matching.

NumPy: For numerical operations and matrix manipulations.

PyTorch & TensorFlow: For deep learning model integration and advanced image processing tasks.

Streamlit: For creating an interactive web interface to visualize depth maps and stereo images.

Matplotlib: For plotting and visualizing results.

**How to Run:**

1. Navigate to the Project Directory by using cd command:

Example: cd filename


2. Set up the Virtual Environment:

python -m venv myen


3. Activate the Virtual Environment:

myen\Scripts\activate


4. Upgrade pip:

python.exe -m pip install --upgrade pip


5. Install Required Packages:

pip install opencv-python opencv-python-headless numpy torch torchvision num2words streamlit tensorflow matplotlib timm


6. Clear Torch Cache (if necessary):

rm -rf ~/.cache/torch/hub


7. Run the Python Script:

python cashe.py


8. Launch the Streamlit App:

streamlit run app.py

This sequence ensures the environment is correctly configured, all dependencies are installed, and the project runs without errors. Let me know if you encounter any issues!

**Conclusion:**

This project provides a comprehensive approach to depth estimation through stereo image processing, leveraging both traditional computer vision and modern deep learning techniques. By integrating PyTorch, TensorFlow, and Streamlit, the system ensures high accuracy and offers an interactive, user-friendly experience for testing and evaluating stereo vision applications.
