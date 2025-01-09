pip install . &&
pip install openpyxl &&
pip install colorama &&
pip install pyarrow==15.0.2 &&
pip install opencv-fixer==0.2.5 &&
python -c "from opencv_fixer import AutoFix; AutoFix()" &&
pip install textworld==1.6.1