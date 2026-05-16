# 🤖 Task 1 — Chatbot with Rule-Based Responses

> **CodSoft AI Internship | Sreeja Pollreddy | BY25RY287818**

A smart, interactive chatbot that uses **regex pattern matching** and **predefined rules** to carry natural conversations with users. Built with a beautiful **Streamlit** web interface.

---

## 🎯 Features

- ✅ **25+ conversation categories** — greetings, jokes, facts, AI concepts, math, and more
- ✅ **Regex-powered pattern matching** — flexible, case-insensitive matching
- ✅ **Math calculations** — e.g., "What is 25 + 37?"
- ✅ **Dynamic responses** — multiple responses per category for variety
- ✅ **Name recognition** — remembers your name during the session
- ✅ **Date/Time awareness** — knows the current date and time
- ✅ **Beautiful Streamlit UI** — modern chat interface with typing animation
- ✅ **Quick action buttons** — one-click prompt suggestions
- ✅ **Chat statistics** — message counter and user tracking
- ✅ **Conversation history** — maintains full chat history

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **`re` (regex)** | Pattern matching for user inputs |
| **`random`** | Response variety |
| **`datetime`** | Date/time queries |
| **Streamlit** | Interactive web-based chat UI |

---

## 🚀 How to Run

### Option 1: Streamlit Web App (Recommended)
```bash
# Navigate to this directory
cd "Task 1 - Chatbot"

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
The app will open at `http://localhost:8501`

### Option 2: Command Line
```bash
python chatbot.py
```

---

## 💬 Sample Conversations

```
You: Hello!
CodBot: Hey there! Nice to see you! What's on your mind?

You: My name is Sreeja
CodBot: Nice to meet you, Sreeja! 😊 How can I help you today?

You: Tell me a joke
CodBot: Why do programmers prefer dark mode? Because light attracts bugs! 😂

You: What is 42 * 58?
CodBot: The answer is 42 * 58 = 2436! 🔢

You: What is artificial intelligence?
CodBot: AI is the field of computer science focused on creating systems that 
        can perform tasks requiring human intelligence! 🧠

You: Give me motivation
CodBot: 💪 "The only way to do great work is to love what you do." — Steve Jobs

You: Bye!
CodBot: Goodbye! 👋 It was great chatting with you. Have a wonderful day!
```

---

## 📁 Project Structure

```
Task 1 - Chatbot/
├── chatbot.py          # Core chatbot engine (RuleBasedChatbot class)
├── app.py              # Streamlit web interface
├── requirements.txt    # Python dependencies
├── screenshots/        # App screenshots for documentation
└── README.md           # This file
```

---

## 🧠 How It Works

1. **User input** is received and converted to lowercase
2. **Regex patterns** are matched against the input in priority order
3. **First matching pattern** triggers a response from its response pool
4. **Template variables** (e.g., `{datetime}`, `{math_result}`) are replaced dynamically
5. If no pattern matches, a **fallback response** is returned

```
User Input → Lowercase → Regex Match → Select Random Response → Process Templates → Output
```

---

## 📚 Concepts Demonstrated

- **Natural Language Processing (NLP)** — Basic text understanding via pattern matching
- **Regular Expressions** — Flexible pattern recognition
- **Conversation Flow** — Context-aware responses (name memory)
- **Object-Oriented Programming** — Clean, modular `RuleBasedChatbot` class
- **Web Development** — Interactive Streamlit UI with custom CSS

---

## 📸 Screenshots

*Screenshots will be added after running the application.*

---

## 📜 License

Part of the CodSoft AI Virtual Internship Program.

---
