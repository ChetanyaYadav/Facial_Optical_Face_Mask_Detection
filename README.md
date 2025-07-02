# Facial_Optical_Face_Mask_Detection

This is my university final year project feel free for any quries,This is am open-source project you can use it for underprivileged or for better outcomes for futuristic with new tools
This Project is completely based on Surveillance of my university professors

🧠 Facial Optical Face Mask Detection
A computer vision-based system that uses deep learning to automatically detect whether a person is wearing a face mask correctly. This project was developed to support public health safety by monitoring mask usage in real-time through camera feeds.

📌 Project Overview
The Facial Optical Face Mask Detection system utilizes OpenCV, TensorFlow/Keras, and a pre-trained CNN (Convolutional Neural Network) model to:

Detect human faces in live video streams.

Classify each detected face as:

Mask (properly worn)

No Mask

Improper Mask (nose uncovered, etc.)

This project can be deployed in public areas such as offices, hospitals, schools, and malls to promote mask compliance.

🧰 Tech Stack
🐍 Python

💻 OpenCV

🤖 TensorFlow / Keras

📦 NumPy, Matplotlib

🗂 Pre-trained Haar Cascade / DNN for face detection

📂 Project Structure
bash
Copy
Edit
facial-mask-detection/
│
├── dataset/                 # Training dataset (mask, no-mask, improper)
├── model/                   # Trained model and saved weights
├── src/
│   ├── train_model.py       # Model training script
│   ├── detect_mask_video.py# Real-time mask detection with webcam
│   └── utils.py             # Utility functions for preprocessing, etc.
├── requirements.txt         # List of required Python libraries
└── README.md                # Project documentation
🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/facial-mask-detection.git
cd facial-mask-detection
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Train the Model (Optional)
bash
Copy
Edit
python src/train_model.py
Pre-trained models are also provided in the /model directory.

4. Run Real-time Detection
bash
Copy
Edit
python src/detect_mask_video.py
Make sure your webcam is connected.

📊 Results
The model achieves high accuracy on a diverse test set and can run in real time with minimal latency. Below are some sample outputs:

✅ Person wearing a mask

❌ Person not wearing a mask

⚠️ Person wearing a mask incorrectly

🧪 Dataset
We used a custom dataset combined with open datasets like:

MaskedFace-Net

RMFD (Real-World Masked Face Dataset)

📌 Use Cases
Hospitals

Airports

Office Spaces

Educational Institutions

Public Transport Stations

📎 Future Improvements
Add YOLOv8 for faster face detection.

Improve classification of "improperly worn" masks.


👤 Author
Chetanya Yadav
📧 yadavchetanya111@gmail.com
🔗 LinkedIn:https://www.linkedin.com/in/chetanya-yadav-07a048207/

📄 License
This project is open-source under the MIT License.

