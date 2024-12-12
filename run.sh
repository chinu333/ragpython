python3.10 -m pip install --upgrade pip
pip uninstall pydantic
pip uninstall httpx
pip install -r requirements.txt
python3.10 -m streamlit run copilot_agents.py --server.port 8000 --server.address 0.0.0.0