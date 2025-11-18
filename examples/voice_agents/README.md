# 🗣️ LiveKit Voice Agent — Intelligent Filler Word Interruption Handler
## NSUT Step-2 Assignment Submission

This project implements an intelligent real-time interruption handler for LiveKit voice agents.

It filters conversational filler words like “umm”, “uh”, “hmm”, and “haan” while the agent is speaking, and only stops TTS when the user speaks a meaningful interruption.

This folder includes:

  - `filler_handler.py` – interruption logic
  - `voice_agent.py` – fully integrated LiveKit agent
  - `test_suite.py` – full unit & integration test suite
The solution meets all assignment requirements and includes optional bonus features.

## ⭐ Features Implemented

### ✅ Core Requirements
  - Ignores fillers when agent is speaking
  - Accepts fillers when agent is quiet
  - Stops TTS immediately on real interruptions
  - Clean classification into:
    - `FILLER_ONLY`
    - `REAL_INTERRUPTION`
    - `MIXED`
    - `UNKNOWN`
  - Fully async and thread-safe
  - No modification to LiveKit VAD or SDK internals
  - Works naturally within LiveKit's transcription stream

### 🎁 Bonus Features
  - Dynamic filler-word updates (`add`, `remove`, `update`)
  - Confidence thresholding for noisy ASR
  - Unicode & multilingual support (e.g., "हाँ", "acha")
  - Statistics tracking (ignored fillers, interruptions, event counts)
  - Complete automated test suite (unit + integration)

## Project Structure
``` 
  voice_agents/ ├── filler_handler.py ├── voice_agent.py ├── test_suite.py └── README.md 
  ``` 

## ⚙️ Installation & Setup

### 1️⃣ Navigate to the correct folder
```
  cd agents/python/examples/voice_agents
```

### 2️⃣ (Optional) Install dependencies
```
pip install -r requirements.txt 
``` 

### If missing, install manually:
``` 
pip install livekit-agents openai
``` 

## 🔧 Environment Variables (Optional)

### Add these to .env:
```
LIVEKIT_URL=wss://your-livekit-server.com LIVEKIT_API_KEY=your_api_key LIVEKIT_API_SECRET=your_secret IGNORED_FILLER_WORDS=uh,um,umm,hmm,haan,mhmm,ah,er CONFIDENCE_THRESHOLD=0.6 
``` 
These are optional — defaults are applied without them.

## 🧪 Running Tests

### ▶ Run Unit Tests
<pre> ``` python test_suite.py --unit ``` </pre>

### ▶ Run Integration Tests
<pre> ``` python test_suite.py --integration ``` </pre>

### ▶ Run All Tests
<pre> ``` python test_suite.py ``` </pre>

#### Integration Scenarios Covered
  - Filler during agent speech
  - Real interruption
  - Filler when agent quiet
  - Mixed filler + command
  - Low confidence murmurs
  - Rapid turn-taking
  - Multiple filler sequences

## 🎙️ Running the LiveKit Voice Agent

### ▶ Start LiveKit Worker
<pre> ``` python voice_agent.py start ``` </pre>

### ▶ Test Mode (no LiveKit needed)
<pre> ``` python voice_agent.py --test ``` </pre>

## 🧠 Interruption Logic Overview

<pre> ``` if agent_is_speaking: if confidence < threshold: ignore if filler_only: ignore if mixed or real: interrupt (stop TTS) else: accept all input (filler or real) ``` </pre>

## 📊 Example Test Output

<pre> ``` Scenario 1: PASSED Scenario 2: PASSED Scenario 3: PASSED Scenario 4: PASSED Scenario 5: PASSED Scenario 6: PASSED Scenario 7: PASSED Handler Statistics: total_events: 9 fillers_ignored: 3 real_interruptions: 3 low_confidence_ignored: 1 ignore_rate: 33.33% ``` </pre>
---

## 🏁 Submission Notes

This branch contains:
  - ✔ FillerWordHandler implementation
  - ✔ LiveKit voice agent with proper integration
  - ✔ Automated tests (unit + integration)
  - ✔ Logging, confidence filtering, and warning messages
  - ✔ Assignment documentation (README)

All required + bonus tasks have been implemented successfully.

## 👤 Author

**Tejas Joshi**

Netaji Subhas University of Technology

LiveKit Intelligent Interruption Handler — Step-2 Assignment
