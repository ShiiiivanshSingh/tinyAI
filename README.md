<div align="center">
 

![tinyAI](https://capsule-render.vercel.app/api?type=transparent&height=100&color=gradient&text=tinyAI&animation=fadeIn&textBg=false)

[![Check out the live site](https://img.shields.io/badge/Check%20Out-Live%20Site-blue?style=for-the-badge)](https://tinyai-production.up.railway.app/)


hii thx for visiting this repo, if u are curious about this project, its an AI chat application built with **Flask, LangGraph, and Google Gemini** or to put simply its a Gemini + LangGraph Chat Agent!! Built this to experiment with **LangGraph agents, Gemini, embeddings, caching, streaming, and persistent state** yada yada yada.

<img width="800" alt="Gemini LangGraph Chat Agent" src="https://github.com/user-attachments/assets/91d81b09-aa1f-4951-bb46-191859350cd1" />
</div>





<br> <br>

Features:

* Gemini ReAct agent with tool calling
* Streaming responses using SSE
* Semantic caching with Gemini embeddings
* Automatic fallback model on rate limits
* SQLite-based conversation checkpoints
* Simple weather tool

## u want to build it yourself ??

or u can [visit this](https://tinyai-production.up.railway.app/)

```bash
python -m venv venv
source venv/bin/activate
pip install flask python-dotenv langgraph langchain-google-genai
```

Add your API key to `.env`:

```env
GOOGLE_API_KEY=your_api_key
```

## Run

```bash
python app.py
```

Runs on `http://localhost:5000`.

## Models

* Primary: `gemini-3.1-flash-lite`
* Fallback: `gemini-3.6-flash`
* Embeddings: `gemini-embedding-001`

## Architecture

```text
User
 ↓
Semantic Cache
 ↓
Gemini Agent
 ↓
Tool Calls
 ↓
Streaming Response

Rate Limit → Fallback Model
```


Font used in the website is [Balsamiq Sans](https://fonts.google.com/specimen/Balsamiq+Sans).

> [!WARNING]
> **Be aware of Gemini API rate limits.** The primary model may hit quota limits, especially during frequent testing. The app automatically switches to the fallback model when a rate limit is detected.
