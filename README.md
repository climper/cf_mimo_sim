# Cell-Free Massive MIMO Simulator

This project simulates a cell-free massive MIMO system with flexible clustering, pilot, and resource block allocation. It provides an interactive Streamlit-based UI for visualizing UE-AP clustering, monitoring resource allocation, and exploring user mobility scenarios.

## Features
- **Flexible simulation parameters**: Set number of APs, UEs, pilots, area size, RBs, and more.
- **Two simulation modes**:
  - **Random Walk Mode**: All UEs move randomly at each step.
  - **Single Probe Mode**: All UEs are fixed except one probe UE, which moves and is highlighted.
- **Resource allocation**: Ensures all APs in a UE's cluster use the same RB for that UE, with strict per-RB UE limits.
- **Pilot allocation**: Assigns pilots to UEs with spatial reuse constraints.
- **Interactive visualization**: See AP/UE positions, cluster connections, and step through time. Highlight new connections and monitor detailed allocations.
- **Monitoring panel**: Inspect which APs serve each UE, pilot/RB assignments, and AP resource usage.

## How to Run Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd cellfree-mimo-simulator
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

- The app will open in your browser. Use the sidebar to set parameters and select simulation mode.
- Use the "Next Step" and "Prev Step" buttons to step through the simulation.
- Use the monitoring panel to inspect UE/AP allocations.

## Requirements
- Python 3.8+
- See `requirements.txt` for Python package dependencies.

## Notes
- In **Random Walk Mode**, all UEs move randomly at each step.
- In **Single Probe Mode**, only the selected probe UE moves (highlighted in pink), and its new connections can be highlighted in green.
- The simulation enforces that all APs in a UE's cluster use the same RB for that UE, and each AP can serve at most one UE per RB.

---

For questions or contributions, please open an issue or pull request. 