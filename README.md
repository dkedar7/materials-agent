# Materials Tool with LLMs

## Get started

#### 1. Create the environment
```
conda create -n llm-materials python==3.10 -y
conda activate llm-materials
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name=llm-materials
```

#### 2. Set environment variables
Add the following script to src/.env:

```
OPENAI_API_KEY="sk-......."
```

#### 3. Run the app

```
python -m src.app
```

### Debug the app
OR, run the app in the VS code debug mode with the keyboard shortcut found under `Run/Start Debugging` in VS code. 
This is usually `Fn+5` for Mac.