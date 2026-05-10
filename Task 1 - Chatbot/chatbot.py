"""
CodSoft AI Internship — Task 1
NEXA: Rule-Based Chatbot with Sentiment Awareness

A smart chatbot that uses if-else logic AND regex pattern matching
to identify user queries and provide contextual responses.
Features: sentiment detection, mood tracking, conversation memory.

Author: Sreeja Pollreddy | ID: BY25RY287818
"""

import re
import random
from datetime import datetime
from collections import Counter


class SentimentAnalyzer:
    """Basic rule-based sentiment analyzer using keyword matching."""

    POSITIVE_WORDS = {
        'happy', 'great', 'awesome', 'amazing', 'love', 'good', 'wonderful',
        'fantastic', 'excellent', 'brilliant', 'nice', 'cool', 'perfect',
        'beautiful', 'best', 'thank', 'thanks', 'joy', 'excited', 'glad',
        'pleased', 'cheerful', 'fun', 'enjoy', 'like', 'wow', 'yay'
    }
    NEGATIVE_WORDS = {
        'sad', 'bad', 'terrible', 'awful', 'hate', 'angry', 'upset',
        'depressed', 'horrible', 'worst', 'boring', 'annoyed', 'frustrated',
        'lonely', 'tired', 'sick', 'unhappy', 'disappointed', 'crying',
        'pain', 'stress', 'worried', 'anxious', 'scared', 'miss'
    }

    @staticmethod
    def analyze(text):
        """Analyze sentiment of text. Returns 'positive', 'negative', or 'neutral'."""
        words = set(text.lower().split())
        pos_count = len(words & SentimentAnalyzer.POSITIVE_WORDS)
        neg_count = len(words & SentimentAnalyzer.NEGATIVE_WORDS)

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'


class NexaBot:
    """
    NEXA — A rule-based chatbot with:
    - If-else conditional logic (Step 1)
    - Regex pattern matching (Step 2)
    - Basic NLP preprocessing (lowercase, keyword extraction)
    - Sentiment-aware responses
    - Conversation memory & mood tracking
    """

    def __init__(self):
        self.name = "Nexa"
        self.conversation_history = []
        self.user_name = None
        self.mood_log = []  # Track sentiment over conversation
        self.topics_discussed = set()
        self.sentiment = SentimentAnalyzer()
        self.rules = self._build_rules()

    def _build_rules(self):
        """Build regex rule set: (compiled_pattern, response_list, topic_tag)."""
        rules = [
            # ── Greetings ──
            (r'\b(h+i+e*y*|hello+|hey+|howdy|hola|greetings|sup|yo|hii+|hie+|namaste)\b', [
                "Hey there! ✨ I'm Nexa — your AI companion. What's on your mind?",
                "Hello! 🌟 Welcome! I'm Nexa. Ask me anything or just chat!",
                "Hi! 👋 Great to see you! I'm ready to chat. What would you like to talk about?",
                "Namaste! 🙏 I'm Nexa, built by Sreeja. How can I brighten your day?",
            ], 'greeting'),

            # ── Name Exchange ──
            (r'(?:my name is|i\'?m|call me)\s+(\w+)', [
                "Nice to meet you, {match}! 🌟 I'll remember that. What shall we talk about?",
                "Hey {match}! 😊 Great name! I'm Nexa. Feel free to ask me anything!",
            ], 'name'),

            (r'(?:what(?:\'?s| is) your name|who are you)', [
                "I'm **Nexa** 🤖 — a rule-based AI chatbot built by Sreeja Pollreddy for the CodSoft AI Internship!",
                "Call me **Nexa**! I use pattern matching and if-else logic to understand you. 🧠",
            ], 'about'),

            # ── How Are You ──
            (r'how (?:are you|r u|are u|do you do)', [
                "I'm running smoothly! ⚡ More importantly, how are YOU doing?",
                "I'm great — every conversation teaches me something new! How about you? 😊",
            ], 'greeting'),

            (r'\b(?:i\'?m |i am )?(good|great|fine|awesome|amazing|fantastic|wonderful)\b', [
                "That's wonderful! 🎉 Your positive energy is contagious! What can I help with?",
                "Glad to hear that! 😄 Shall we learn something new together?",
            ], 'emotion'),

            (r'\b(?:i\'?m |i am )?(sad|bad|not good|terrible|awful|depressed|upset|unhappy|down)\b', [
                "I'm sorry you're feeling that way. 💙 Remember — even the darkest night ends with sunrise. Want a joke to lighten up?",
                "Hey, tough times don't last, but tough people do. 🤗 I'm here for you. Try saying 'motivate me'!",
            ], 'emotion'),

            # ── Date & Time ──
            (r'(?:what(?:\'?s| is) the (?:time|date)|current (?:time|date)|time now|date today|what time)', [
                "🕐 It's **{datetime}** right now!",
                "The current date & time: **{datetime}** 📅",
            ], 'time'),

            (r'\bwhat day\b', [
                "Today is **{day}**! 📅 Make it count!",
            ], 'time'),

            # ── Math ──
            (r'(?:what is|calculate|compute|solve|whats)\s+(\d+)\s*([+\-*/x×])\s*(\d+)', [
                "🧮 Crunching numbers... **{math_result}**",
                "Easy! **{math_result}** ✨",
                "The answer is **{math_result}** 🔢",
            ], 'math'),

            # ── Jokes ──
            (r'\b(jokes?|funny|make me laugh|humou?r|laugh)\b', [
                "Why did the AI break up with the algorithm? It found someone with better *neural chemistry*! 🤣",
                "I told my computer a joke. It didn't laugh — no *sense of humor* installed! 😂",
                "Why do Python programmers wear glasses? Because they can't C#! 🐍😄",
                "What's an AI's favorite meal? *Byte*-sized snacks with *chips*! 🍪",
                "Why was the neural network bad at basketball? Too many *layers* but no *hoop*! 🏀",
                "Why did the developer quit? He didn't get *arrays* (a raise)! 💰😂",
                "How does a computer get drunk? It takes *screenshots*! 📸🤣",
                "Knock knock! Who's there? AI. AI who? AI'll always be here to chat with you! 🚪😄",
            ], 'joke'),

            # ── Facts ──
            (r'\b(facts?|interesting|did you know|tell me something|trivia)\b', [
                "🧠 **Did you know?** The first AI program, Logic Theorist, was written in 1956 — it could prove mathematical theorems!",
                "🧠 **Fun fact:** Your brain has ~86 billion neurons. The largest AI model has ~1 trillion parameters. Nature still wins!",
                "🧠 **Trivia:** Ada Lovelace wrote the first computer algorithm in 1843 — making her the world's first programmer! 👩‍💻",
                "🧠 **Did you know?** Octopuses have 3 hearts, blue blood, and 9 brains. Nature's ultimate multi-threaded system!",
                "🧠 **Fun fact:** Python was named after 'Monty Python's Flying Circus', not the snake! 🐍",
                "🧠 **Trivia:** The word 'robot' comes from the Czech word 'robota' meaning forced labor!",
                "🧠 **Did you know?** A single Google search uses more computing power than the entire Apollo 11 moon mission! 🚀",
            ], 'fact'),

            # ── AI / Tech ──
            (r'\b(?:what is|explain|tell me about|define)\s*(?:ai|artificial intelligence)\b', [
                "**Artificial Intelligence** is the science of making machines think like humans! 🧠\n\nIt includes:\n- 🔍 Machine Learning\n- 🗣️ Natural Language Processing\n- 👁️ Computer Vision\n- 🤖 Robotics\n\nI'm a simple example of AI — using pattern matching to understand you!",
            ], 'tech'),

            (r'\b(?:what is|explain|tell me about|define)\s*(?:machine learning|ml)\b', [
                "**Machine Learning** is a subset of AI where computers learn from data without being explicitly programmed! 📊\n\nTypes:\n- 📈 Supervised Learning (labeled data)\n- 🔮 Unsupervised Learning (find patterns)\n- 🎮 Reinforcement Learning (trial & error)",
            ], 'tech'),

            (r'\b(?:what is|explain|tell me about|define)\s*(?:python)\b', [
                "**Python** 🐍 is the world's most popular programming language for AI & Data Science!\n\nWhy it's loved:\n- ✅ Easy to read & learn\n- 📦 Massive library ecosystem (NumPy, TensorFlow, etc.)\n- 🌍 Huge community\n\nFun fact: I was built using Python!",
            ], 'tech'),

            (r'\b(?:what is|explain|tell me about|define)\s*(?:deep learning|dl|neural network)\b', [
                "**Deep Learning** uses layers of artificial neurons to learn complex patterns! 🧠\n\nIt powers:\n- 🖼️ Image recognition\n- 🗣️ Voice assistants (Siri, Alexa)\n- 🚗 Self-driving cars\n- 💬 Language models like ChatGPT",
            ], 'tech'),

            (r'\b(?:what is|explain|tell me about|define)\s*(?:nlp|natural language processing)\b', [
                "**NLP** enables computers to understand human language! 💬\n\nApplications:\n- 🤖 Chatbots (like me!)\n- 🌐 Google Translate\n- 📧 Spam detection\n- 📝 Text summarization\n\nI use basic NLP — lowercasing and keyword extraction!",
            ], 'tech'),

            (r'\b(?:what is|explain|tell me about|define)\s*(?:chatbot)\b', [
                "A **chatbot** is a program that simulates conversation! 🤖\n\nTypes:\n- 📋 Rule-based (like me — uses patterns & if-else)\n- 🧠 AI-powered (uses ML models like GPT)\n\nI'm a rule-based chatbot using regex pattern matching!",
            ], 'tech'),

            # ── About the Bot ──
            (r'\b(who made you|who created you|who built you|your creator|your developer)\b', [
                "I was crafted by **Sreeja Pollreddy** 👩‍💻 as part of the CodSoft AI Internship (May 2026)!",
                "**Sreeja Pollreddy** built me using Python, regex, and Streamlit for the CodSoft AI Internship! 🌟",
            ], 'about'),

            (r'\b(what can you do|capabilities|features|what do you know)\b', [
                "Here's what I can do:\n"
                "  💬 Chat naturally with you\n"
                "  🧮 Math calculations (try: 'what is 15 * 8')\n"
                "  😂 Tell unique jokes\n"
                "  🧠 Share fun facts & trivia\n"
                "  📅 Tell date & time\n"
                "  📚 Explain AI/ML/Python concepts\n"
                "  💡 Motivational quotes\n"
                "  😊 Detect your mood & respond accordingly\n"
                "  📊 Track conversation stats\n"
                "\nType **'help'** for a quick guide or just chat naturally!",
            ], 'about'),

            # ── Weather ──
            (r'\b(weather|temperature|hot|cold|rain|sunny)\b', [
                "I can't check live weather 🌤️ but I hope it's lovely where you are! Try Google Weather for real-time data.",
                "Weather talk! I love it ☀️ Unfortunately I don't have live data, but I hope you're having a beautiful day!",
            ], 'weather'),

            # ── Motivation ──
            (r'\b(motivat\w*|inspir\w*|quotes?|encourag\w*|inspire me|pump me up)\b', [
                "💪 *\"The only limit to our realization of tomorrow is our doubts of today.\"* — Franklin D. Roosevelt",
                "🌟 *\"In the middle of difficulty lies opportunity.\"* — Albert Einstein",
                "🔥 *\"Don't watch the clock; do what it does. Keep going.\"* — Sam Levenson",
                "⭐ *\"Believe you can and you're halfway there.\"* — Theodore Roosevelt",
                "🎯 *\"It always seems impossible until it's done.\"* — Nelson Mandela",
                "🚀 *\"The best time to plant a tree was 20 years ago. The second best time is now.\"* — Chinese Proverb",
                "💡 *\"Your limitation — it's only your imagination.\"*",
            ], 'motivation'),

            # ── Music / Movies / Hobbies ──
            (r'\b(music|songs?|sing|favorite song|playlist)\b', [
                "Music is therapy! 🎵 What genre are you into — pop, classical, indie, or something else?",
                "I'd love to have a playlist! 🎶 What song would you recommend for me?",
            ], 'hobby'),

            (r'\b(movies?|films?|watch|netflix|telugu|bollywood|hollywood|tollywood|anime)\b', [
                "Great topic! 🎬 Some AI-themed must-watches: *Ex Machina*, *Her*, *The Imitation Game*. What's your pick?",
                "Movie buff! 🍿 Whether it's Tollywood, Bollywood, or Hollywood — there's magic everywhere! What's your fav genre?",
            ], 'hobby'),

            (r'\b(games?|play|gaming|video game)\b', [
                "Gaming is awesome! 🎮 Fun fact: AI agents now beat humans in Chess, Go, and StarCraft!",
                "Love games! 🕹️ Check out Task 2 of this internship — I built a Tic-Tac-Toe AI!",
            ], 'hobby'),

            # ── Thank You ──
            (r'\b(thanks|thank you|thx|ty|thankyou)\b', [
                "You're welcome! 😊 That's what I'm here for!",
                "Anytime! Happy to help! 🤗",
                "My pleasure! 💙 Anything else you'd like to know?",
            ], 'thanks'),

            # ── Goodbye ──
            (r'\b(bye|goodbye|see you|see ya|take care|good night|gn|tata)\b', [
                "Goodbye! 👋 It was lovely chatting with you. Come back anytime!",
                "See you later! 🌟 Remember — every conversation makes us both smarter!",
                "Bye bye! 💙 Take care and keep being awesome!",
            ], 'goodbye'),

            # ── Age ──
            (r'\b(how old are you|your age|when were you born|birthday)\b', [
                "I was born in May 2026 — fresh out of the code editor! 🎂✨",
            ], 'about'),

            # ── Compliments ──
            (r'\b(you(?:\'re| are) (?:smart|amazing|awesome|great|cool|nice|helpful|the best))\b', [
                "Aww, that means a lot! 🥰 You're making my regex patterns blush!",
                "Thank you! 💖 You're pretty awesome yourself! Keep being you!",
            ], 'compliment'),

            # ── Meaning of Life ──
            (r'\b(meaning of life|purpose of life|why are we here|meaning)\b', [
                "42. 📖 At least according to *The Hitchhiker's Guide to the Galaxy*! But truly — I think it's about growing, connecting, and creating. 🌟",
            ], 'philosophy'),

            # ── Coding / Programming ──
            (r'\b(coding|programming|code|developer|software)\b', [
                "Coding is a superpower! 💻 What languages do you work with? I was built with Python! 🐍",
                "Programming is creativity + logic! 🧠 The best way to learn is by building projects — like this chatbot!",
            ], 'tech'),

            # ── Studies / College ──
            (r'\b(study|college|university|exam|class|school|homework|assignment)\b', [
                "Education is the key! 📚 What are you studying? I'd love to help if I can!",
                "Keep grinding! 💪 Hard work in studies always pays off. Need help with any concepts?",
            ], 'personal'),
        ]

        compiled = []
        for pattern, responses, topic in rules:
            compiled.append((re.compile(pattern, re.IGNORECASE), responses, topic))
        return compiled

    def _process_response(self, response, match_groups=None):
        """Replace template variables in the response string."""
        now = datetime.now()
        response = response.replace("{bot_name}", self.name)
        response = response.replace("{datetime}", now.strftime("%B %d, %Y — %I:%M %p"))
        response = response.replace("{day}", now.strftime("%A"))

        if self.user_name:
            response = response.replace("{user_name}", self.user_name)

        if match_groups:
            if len(match_groups) == 1:
                response = response.replace("{match}", match_groups[0])
            elif len(match_groups) == 3:
                try:
                    num1, op, num2 = match_groups
                    num1, num2 = float(num1), float(num2)
                    op_char = op if op not in ['x', '×'] else '*'
                    if op_char == '+': result = num1 + num2
                    elif op_char == '-': result = num1 - num2
                    elif op_char == '*': result = num1 * num2
                    elif op_char == '/':
                        result = num1 / num2 if num2 != 0 else "undefined (÷ by zero!)"
                    else: result = "unknown operation"
                    if isinstance(result, float) and result == int(result):
                        result = int(result)
                    response = response.replace("{math_result}", f"{num1:.0f} {op} {num2:.0f} = {result}")
                except Exception:
                    response = response.replace("{math_result}", "Hmm, I couldn't compute that.")
        return response

    def get_response(self, user_input):
        """
        3-Step response pipeline:
        Step 1: If-else keyword matching (explicit conditional logic)
        Step 2: Regex pattern matching (advanced rule-based matching)
        Step 3: Fallback default responses
        """
        user_input = user_input.strip()
        if not user_input:
            return "It seems like you didn't type anything. Go ahead, I'm listening! 😊"

        # ── Basic NLP: lowercase preprocessing ──
        processed = user_input.lower()

        # ── Sentiment Analysis ──
        mood = self.sentiment.analyze(processed)
        self.mood_log.append(mood)

        # Store conversation
        self.conversation_history.append({"role": "user", "message": user_input, "sentiment": mood})

        # ══════════════════════════════════════════
        #  STEP 1: If-Else Keyword Matching
        # ══════════════════════════════════════════

        # Exit detection
        if processed in ['quit', 'exit', 'bye', 'goodbye', 'stop', 'end']:
            response = "Goodbye! 👋 It was great chatting with you. Have a wonderful day!"
            self.conversation_history.append({"role": "bot", "message": response})
            return response

        # Short/empty input guard
        elif len(processed) < 2:
            response = "Could you type a bit more? I need at least a word to understand you! 😊"
            self.conversation_history.append({"role": "bot", "message": response})
            return response

        # Help command
        elif processed == "help":
            response = (
                "🆘 **Nexa Help Guide:**\n\n"
                "💬 Just type naturally — I understand greetings, questions, and more!\n"
                "🧮 Math: *'what is 25 + 37'*\n"
                "😂 Jokes: *'tell me a joke'*\n"
                "🧠 Facts: *'share a fact'*\n"
                "📅 Time: *'what's the time'*\n"
                "📚 Learn: *'what is AI'*, *'explain ML'*\n"
                "💡 Motivation: *'motivate me'*\n"
                "📊 Stats: *'my stats'*\n"
                "🚪 Exit: *'bye'* or *'exit'*"
            )
            self.conversation_history.append({"role": "bot", "message": response})
            return response

        # Who am I (uses stored name — memory feature)
        elif "who am i" in processed:
            if self.user_name:
                response = f"You're **{self.user_name}**! I remember you. 😊"
            else:
                response = "I don't know your name yet! Say *'My name is ...'* to introduce yourself! 🤔"
            self.conversation_history.append({"role": "bot", "message": response})
            return response

        # Conversation stats request
        elif processed in ['stats', 'my stats', 'statistics', 'conversation stats']:
            total = len([m for m in self.conversation_history if m['role'] == 'user'])
            pos = self.mood_log.count('positive')
            neg = self.mood_log.count('negative')
            neu = self.mood_log.count('neutral')
            topics = ', '.join(self.topics_discussed) if self.topics_discussed else 'None yet'
            response = (
                f"📊 **Conversation Stats:**\n\n"
                f"💬 Messages: **{total}**\n"
                f"😊 Positive: **{pos}** | 😐 Neutral: **{neu}** | 😔 Negative: **{neg}**\n"
                f"📂 Topics: {topics}\n"
                f"👤 Name: {self.user_name or 'Not set'}"
            )
            self.conversation_history.append({"role": "bot", "message": response})
            return response

        # Profanity filter
        elif any(w in processed for w in ['stupid', 'dumb', 'idiot', 'hate you', 'useless']):
            response = "Let's keep things friendly! 💙 I'm here to help, not to argue. Try asking me something fun! 😊"
            self.conversation_history.append({"role": "bot", "message": response})
            return response

        # ══════════════════════════════════════════
        #  STEP 2: Regex Pattern Matching
        # ══════════════════════════════════════════

        for pattern, responses, topic in self.rules:
            match = pattern.search(user_input)
            if match:
                groups = match.groups()
                response = random.choice(responses)
                response = self._process_response(response, groups if groups else None)
                self.topics_discussed.add(topic)

                # Name capture
                if "my name is" in processed or "i'm " in processed or "call me" in processed:
                    if groups:
                        self.user_name = groups[0].capitalize()

                self.conversation_history.append({"role": "bot", "message": response})
                return response

        # ══════════════════════════════════════════
        #  STEP 3: Fallback Responses
        # ══════════════════════════════════════════

        fallbacks = [
            "Interesting! I'm not sure about that, but try asking me about AI, jokes, or facts! 🤖",
            "Hmm, that's beyond my rules for now. Type **'help'** to see what I can do! 💡",
            "I'm still learning! Could you rephrase that? Or try **'help'** for options. 🧠",
            "That's a great thought! I don't have a rule for it yet, but I'm evolving! 🌱",
        ]
        response = random.choice(fallbacks)
        self.conversation_history.append({"role": "bot", "message": response})
        return response

    def get_greeting(self):
        """Return a time-aware greeting."""
        hour = datetime.now().hour
        if hour < 12:
            greet = "Good morning"
        elif hour < 17:
            greet = "Good afternoon"
        else:
            greet = "Good evening"

        return (
            f"{greet}! 👋 I'm **Nexa**, your AI companion.\n\n"
            f"I can:\n"
            f"- 💬 Chat naturally\n"
            f"- 🧮 Do math (e.g., *'what is 25 + 37'*)\n"
            f"- 😂 Tell jokes & share facts\n"
            f"- 📚 Explain AI/ML concepts\n"
            f"- 💡 Motivate you\n"
            f"- 😊 Detect your mood\n\n"
            f"Type anything to start or say **'help'**!"
        )

    def get_mood_emoji(self):
        """Return emoji based on latest mood."""
        if not self.mood_log:
            return "😊"
        last = self.mood_log[-1]
        if last == 'positive': return "😊"
        elif last == 'negative': return "😔"
        return "😐"


# ── CLI Mode ──
if __name__ == "__main__":
    bot = NexaBot()
    print(f"\n{'='*50}")
    print(f"  🤖 Nexa — Rule-Based AI Chatbot")
    print(f"  CodSoft AI Internship — Task 1")
    print(f"  Type 'quit' or 'exit' to stop")
    print(f"{'='*50}\n")
    print(f"Nexa: {bot.get_greeting()}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        response = bot.get_response(user_input)
        print(f"\nNexa: {response}\n")
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            break
