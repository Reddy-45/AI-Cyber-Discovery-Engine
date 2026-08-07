# AI Cyber Discovery Engine

> **Capstone Project** — AI-powered threat correlation, risk scoring, and explainable security intelligence.

---

## What It Does

The engine ingests security logs, normalizes them, enriches with threat intelligence, and then runs a six-stage AI reasoning pipeline:

| Stage | What Happens |
|-------|-------------|
| **① Ingest + Normalize** | Load JSON logs → Canonical Event Model |
| **② Enrich** | Extract IOCs, check reputation, map MITRE ATT&CK |
| **③ Correlate** | Group related events using NetworkX entity graphs |
| **④ Score** | Isolation Forest anomaly detection + weighted risk formula (0–100) |
| **⑤ Explain** | Generate plain-English "why this is suspicious" summaries |
| **⑥ Mitigate** | Produce actionable mitigation recommendations |

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd ai-cyber-discovery-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

That's it. No database setup, no API keys, no Docker required.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ai-cyber-discovery-engine/
├── app.py                      # Streamlit dashboard (entry point)
├── config.yaml                 # Runtime configuration
├── requirements.txt
├── data/
│   ├── sample_logs/            # Simulated attack scenarios
│   ├── mitre_attack.json       # Local MITRE ATT&CK database
│   └── threat_intel.json       # Known-bad IPs/hashes
├── engine/
│   ├── models.py               # All data models (Pydantic)
│   ├── config.py               # Config loader
│   ├── ingest.py               # ① Ingest + Normalize
│   ├── enrich.py               # ② Enrich (IOC + reputation + MITRE)
│   ├── analyze.py              # ③–⑥ AI Reasoning Engine
│   └── store.py                # SQLite persistence
└── tests/
    └── test_engine.py
```

---

## Demo Scenarios

The sample data includes three realistic attack narratives:

1. **SSH Brute Force + Privilege Escalation** — 9 failed logins, 1 success, sudo escalation
2. **Network Reconnaissance + Data Exfiltration** — Port scan, open port discovery, 847 MB outbound transfer
3. **Ransomware Infection + C2 Beacon** — WannaCry lateral movement, file encryption, Tor C2

---

## Technology Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.11+ |
| Data Models | Pydantic v2 |
| ML | scikit-learn (Isolation Forest) |
| Graph Correlation | NetworkX |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Storage | SQLite |
| Threat Intel | Local MITRE ATT&CK JSON |
