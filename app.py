import os
os.system(
    "streamlit run src/dashboard/app.py "
    "--server.port 7860 "
    "--server.address 0.0.0.0 "
    "--server.headless true"
)