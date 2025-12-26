# Project Title

### IMAGE BASED BREED RECOGNITION FOR CATTLE OF INDIA

# 📖 About the Project
This project implements an automated cattle breed classification system using Deep Learning techniques.

A pre-trained ResNet18 Convolutional Neural Network (CNN) is fine-tuned to classify cattle images into different breeds.

The system performs end-to-end processing, including dataset preparation, model training, evaluation, visualization, and deployment through a Flask-based web application.

This project is developed as an Academic Mini Project.

## 🎯 Objectives

To classify cattle images into different breeds accurately.

To apply transfer learning for efficient training.

To analyze model performance using graphs and metrics

To deploy the trained model using a web application

## Cattle Breeds Supported
Red Dane Cattle

Jersey Cattle

Holstein Friesian Cattle

Brown Swiss Cattle

Ayrshire Cattle

# ✨ Features

📁 Automatic dataset folder creation (train, val, test)

🔀 Dataset splitting (70% training, 15% validation, 15% testing)

🧠 Transfer Learning using ResNet18

📊 Training & Validation Loss Graphs

📈 Training & Validation Accuracy Graphs

📉 Class distribution bar graph

🧩 Confusion Matrix visualization

📋 Precision, Recall, and F1-score analysis

💾 Trained model saving (.pth)

🌐 Flask-based web application for prediction

⚡ GPU support (CUDA if available)

# 🗂️ Project Structure

<img width="904" height="562" alt="image" src="https://github.com/user-attachments/assets/caecd46f-bb0c-4a83-a4f4-17c89502afe8" />


#  🛠️ Technologies Used

Programming Language: Python

Deep Learning Framework: PyTorch

Model Architecture: ResNet18

Visualization: Matplotlib, Seaborn

Evaluation: Scikit-learn

Web Framework: Flask

Version Control: Git & GitHub

# 📦 Requirements
Software

Python 3.8+

Git

Visual Studio Code (or any IDE)

Python Libraries
```
pip install torch torchvision numpy matplotlib seaborn scikit-learn flask pillow tqdm
```
#  🚀 How to Run the Project

1️⃣ Create Dataset Structure
```
 python folder.py
```
2️⃣ Split Dataset
```
python split.py
```
3️⃣ Train the Model
```
python train.py
```
4️⃣ Evaluate the Model

python predict.py

5️⃣ Run Web Application

```
python app.py

```
Open browser:

http://127.0.0.1:5000

# system Architecture

<img width="723" height="475" alt="image" src="https://github.com/user-attachments/assets/2e7b4750-ba0f-4578-a288-b891b61c0a90" />

# output
```
python folder.py

```
<img width="593" height="111" alt="image" src="https://github.com/user-attachments/assets/8f4fae13-ffd8-485f-951a-8a60d0868ff9" />

```
 python split.py

```
<img width="746" height="362" alt="image" src="https://github.com/user-attachments/assets/9cd41418-b5a8-4eb1-98b7-716d19a14c1b" />

```
python train.py

```

<img width="828" height="718" alt="image" src="https://github.com/user-attachments/assets/41e40f3e-7a37-4af9-bba2-62eacd3d230a" />

<img width="803" height="689" alt="image" src="https://github.com/user-attachments/assets/08d7301d-d1b4-4558-a286-c3c1cd37b840" />

<img width="791" height="685" alt="image" src="https://github.com/user-attachments/assets/ca49acb1-4f4c-4081-ab87-e3aeef97dd70" />

<img width="754" height="722" alt="image" src="https://github.com/user-attachments/assets/df9594da-cd4d-4e6e-ac6a-5c51143a6af3" />

<img width="1310" height="267" alt="image" src="https://github.com/user-attachments/assets/5abe33e0-acda-4f1c-8531-9b14b51865c6" />

```
python predict.py
```
<img width="513" height="674" alt="image" src="https://github.com/user-attachments/assets/345e4fb2-9214-4b3a-b70f-4d382557e80c" />

```
python app.py
```
<img width="1188" height="208" alt="image" src="https://github.com/user-attachments/assets/5044aa2e-b4f5-438c-83dd-d7fb14035d40" />

<img width="1900" height="1017" alt="image" src="https://github.com/user-attachments/assets/d9d2a102-ffec-4848-9fb3-ee055323fa24" />

# 📊 Results and Impact

The cattle breed classification model based on ResNet18 achieved good accuracy and stable performance.
Training and validation loss curves showed proper convergence, while accuracy graphs indicated minimal overfitting.
The confusion matrix and evaluation metrics (Accuracy, Precision, Recall, and F1-score) confirmed reliable classification across all cattle breeds.

This project demonstrates the effective use of deep learning in agriculture, enabling automated cattle breed identification.
It reduces manual effort, supports farmers and veterinarians in decision-making, and serves as a practical educational project in AI and Computer Vision.
The system is scalable and can be extended to additional breeds or deployed on web and mobile platforms.

# Articles published / References
[8] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778. (ResNet fundamentals for breed recognition)

[9] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017. (Object detection basis).

[10] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified Real‑Time Object Detection,” Proc. IEEE CVPR, 2016, pp. 779–788. (Original YOLO paper).

[11] O. Bezsonov et al., “Breed Recognition and Estimation of Live Weight of Cattle Based on Methods of Machine Learning and Computer Vision,” Eastern‑European Journal of Enterprise Technologies, 2021.

[12] A. Vijayalakshmi, P. Shanmugavadivu, S. Vijayalakshmi, S. Padarha, and R. Sivaranjani, “Ensemble Learning Algorithm for Cattle Breed Identification using Computer Vision Techniques,” in Proc. 1st Int. Conf. on AI, Communication, IoT, Data Engineering and Security (IACIDS), 2024. 

[13] U. Ali and W. Muhammad, “Cow Face Detection for Precision Livestock Management using YOLOv8,” International Journal of Innovations in Science & Technology, 2025. 












