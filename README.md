# **Assistant for Ordering Food in Restaurant Using a Multimodal Rag**

## **Prerequisites**
**Python 3.9 or higher** must be installed on your system.

## **Steps to Run the Project**

### 0. Clone the repository

```bash
git clone https://github.com/cesarsiuu2316/Order-QA-Assistant.git
```

### 1. Create and Activate a Virtual Environment
It is recommended to use a virtual environment to install dependencies. Follow these steps or use Conda:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. **Install Dependencies**
Once the virtual environment is active, use the `requirements.txt` file from the repository to install the necessary dependencies. Run the following command:

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and download pretrained model "ollama:gemma3:4b": 

[Download Ollama](https://ollama.com/download/windows)

```bash
# Install multimodal model
ollama pull gemma3:4b
```

### 4. Finally, run the program:

```bash
# Run main program file main.py
streamlit run main.py
```
