import subprocess, sys, os, time, threading

def start_api():
    subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

t = threading.Thread(target=start_api)
t.daemon = True
t.start()
time.sleep(3)

os.system(
    f"streamlit run src/dashboard/app.py "
    "--server.port 7860 "
    "--server.address 0.0.0.0 "
    "--server.headless true"
)