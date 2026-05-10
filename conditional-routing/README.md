# Conditional Routing Agent

An example agentic workflow demonstrating conditional routing in LangGraph. It asks the user how they are feeling, utilizes a gemini-2.0-flash-lite LLM model to assess whether the user's energy level is "low", "medium", or "high", and routes the workflow to one of three corresponding handler nodes to provide activity suggestions.

---

## ⚙️ Setup & Installation

### 1. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
uv venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```
*(If blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating)*

**Windows (Command Prompt):**
```cmd
uv venv
.\.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
uv sync
# Or manually with pip:
# pip install -e .
```

### 3. Environment Variables Setup

Copy the example configuration file:

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows:**
```cmd
copy .env.example .env
```

Add your API keys to the `.env` file:
```ini
GEMINI_API_KEY=your_google_ai_studio_key
```

- **Gemini API Key:** Get it from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. IDE Setup (VS Code Recommended)

To select the virtual environment, press Ctrl/Cmd + Shift + P, select Python: Select Interpreter and pick the one from .venv

---

## 🚀 Running the Application

To run the application, ensure your virtual environment is activated and execute:
```bash
python app.py
```
