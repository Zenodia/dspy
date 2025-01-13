#pip install . &&
pip install openpyxl &&
pip install colorama &&
pip install pyarrow==15.0.2 &&
pip install opencv-fixer==0.2.5 &&
python -c "from opencv_fixer import AutoFix; AutoFix()" &&
pip install dspy &&
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y &&
cp -R  /workspace/dspy/datasets/alfworld/ /usr/local/lib/python3.10/dist-packages/dspy/datasets/

