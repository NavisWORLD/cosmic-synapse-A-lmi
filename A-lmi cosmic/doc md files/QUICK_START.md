# A-LMI Quick Start Guide

**Davis Unified Intelligence System v2.0**

This guide will help you get the A-LMI system up and running.

---

## Prerequisites

Before starting, ensure you have:

1. **Python 3.9+** installed
2. **Docker Desktop** installed and running
3. **Unity 2022.3+ LTS** (optional, for VLCL simulation)
4. **CUDA-capable GPU** (recommended, for CLIP encoding)
5. **Microphone access** (for audio perception)

---

## Step 1: Install Python Dependencies

```bash
cd a_lmi_system
pip install -r requirements.txt
```

**Note**: If you encounter issues with specific packages:
- `vosk`: Requires Vosk model download (see Step 3)
- `torch`: Install from pytorch.org for CUDA support
- `selenium`: Requires ChromeDriver

---

## Step 2: Start Infrastructure Services

```bash
# Navigate to project root
cd ..

# Start all services with Docker Compose
docker-compose up -d

# Verify services are running
docker ps
```

Expected services:
- ✅ Kafka (port 9092)
- ✅ MinIO (port 9000)
- ✅ Milvus (port 19530)
- ✅ Neo4j (port 7687)
- ✅ Vault (port 8200)

---

## Step 3: Download Required Models

### Vosk Model (Required for STT)

```bash
# Create model directory
mkdir -p D:/CST/model

# Download Vosk model (small English)
# Visit: https://alphacephei.com/vosk/models
# Download: vosk-model-small-en-us-0.15.zip
# Extract to: D:/CST/model/vosk-model-small-en-us-0.15/
```

### CLIP Model (Auto-downloaded)

The CLIP model will be automatically downloaded on first use via the `transformers` library.

---

## Step 4: Configure System

Edit `a_lmi_system/config.yaml`:

```yaml
# Update these paths if needed
paths:
  ai_data: "ai/data"
  ai_models: "ai/models"

# Update Vosk model path
services:
  audio_processor:
    vosk_model_path: "D:/CST/model/vosk-model-small-en-us-0.15"

# Set your OpenAI API key (for math reasoning)
services:
  reasoning_engine:
    math_model: "openai/o1-mini"  # or "gpt-4o-mini"
```

Create a `.env` file in `a_lmi_system/`:

```bash
# .env
OPENAI_API_KEY=your_key_here
VAULT_TOKEN=your_vault_token
```

---

## Step 5: Initialize Databases

```bash
cd a_lmi_system
python main.py
```

This will:
1. Connect to all services (Kafka, MinIO, Milvus, Neo4j)
2. Create necessary indexes and collections
3. Start the autonomous agent loop

**First run notes:**
- Milvus will create `semantic_embeddings` and `spectral_signatures` collections
- Neo4j will create indexes on `entity_type` and `token_id`
- MinIO will create `raw-data-archive` bucket

---

## Step 6: Test Individual Components

### Test Crawler

```python
# In a Python shell
from a_lmi_system.services.crawler.spider import run_crawler
from a_lmi_system.core.event_bus import EventBus

event_bus = EventBus(bootstrap_servers=["localhost:9092"], topics={})
event_bus.connect()

run_crawler(
    start_urls=["https://example.com"],
    kafka_producer=event_bus
)
```

### Test Audio Processor

```python
from a_lmi_system.services.audio_processor.processor import AuditoryCortex

processor = AuditoryCortex(
    vosk_model_path="D:/CST/model/vosk-model-small-en-us-0.15",
    kafka_producer=None
)

processor.start()  # Will capture microphone input
```

---

## Step 7: Launch VLCL Simulation (Unity)

1. Open Unity Hub
2. Add project: `vlcl_simulation`
3. Open in Unity 2022.3+
4. Install NetMQ package (Window → Package Manager)
5. Press Play

**Expected behavior:**
- Terrain generates with procedural noise
- Microphone input modulates particle physics
- Particles orbit and swirl according to CST v2 physics

---

## Step 8: Launch Conversational UI

```python
from a_lmi_system.interface.conversational_ui import ConversationalUI
from a_lmi_system.memory.vector_db_client import VectorDBClient
from a_lmi_system.memory.tkg_client import TKGClient

# Initialize clients
vector_db = VectorDBClient()
vector_db.connect()

tkg_client = TKGClient("bolt://localhost:7687", "neo4j", "neo4j_password")
await tkg_client.connect()

# Launch UI
ui = ConversationalUI(vector_db, tkg_client)
ui.launch(share=True)
```

---

## Troubleshooting

### "Cannot connect to Kafka"

```bash
# Check Docker is running
docker ps

# Restart Kafka
docker-compose restart kafka

# Check logs
docker-compose logs kafka
```

### "Vosk model not found"

```bash
# Verify model path in config.yaml
# Ensure model is extracted (not in .zip)
dir D:/CST/model/vosk-model-small-en-us-0.15
```

### "CUDA out of memory"

```yaml
# In config.yaml
gpu:
  enabled: false  # Use CPU instead
```

### "Microphone access denied"

- **Windows**: Settings → Privacy → Microphone → Allow apps to access
- **Mac**: System Preferences → Security & Privacy → Microphone

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│           PYTHON BACKEND (A-LMI)                 │
├─────────────────────────────────────────────────┤
│  Crawler → Audio → Processing → Memory Tiers   │
│     ↓        ↓         ↓            ↓           │
│   Kafka   Kafka    Kafka    MinIO/Milvus/Neo4j │
└─────────────────────────────────────────────────┘
                    ↕ (ZeroMQ IPC)
┌─────────────────────────────────────────────────┐
│      UNITY SIMULATION (VLCL)                   │
├─────────────────────────────────────────────────┤
│  AudioManager → CosmosManager → Particles     │
│      ↓              ↓                           │
│    PSD         CST Physics                     │
└─────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Run integration tests**: See `IMPLEMENTATION_STATUS.md`
2. **Configure monitoring**: Add Prometheus/Grafana
3. **Scale services**: Use Kubernetes for production
4. **Run predictions**: Implement validation experiments

---

## Support

For detailed architecture and theory, see:
- `README.md` - Overview
- `IMPLEMENTATION_STATUS.md` - Current status
- Comprehensive blueprint document - Full theory

---

**Happy Learning!** 🚀

The A-LMI system will now autonomously perceive, learn, and evolve according to fundamental vibrational patterns of reality.

