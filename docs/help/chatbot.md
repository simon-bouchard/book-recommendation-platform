# Chatbot

The site includes a **virtual librarian chatbot** that can answer questions, search the web, look up help documentation, and recommend books from the internal catalogue.

## Access
- You **do not need to be logged in** to use the chatbot.
- Anonymous users have stricter rate limits (3 requests/min, 10/day) compared to logged-in users (5 requests/min, 40/day).
- Logging in also unlocks **personalized recommendations** through the chatbot (ALS-based and subject-based), since those require your rating history.
- Conversations are stored temporarily in Redis (tied to your browser cookie) and expire after 2 days.

## How It Works

The chatbot uses a multi-agent system. Each message is automatically routed to the most appropriate agent:

- **Recommendation agent:** Handles requests for book suggestions. Runs a four-stage pipeline — strategy planning, candidate retrieval, selection, and a personalized written response. Uses ALS collaborative filtering (for logged-in users with 10+ ratings), semantic search, subject-based retrieval, and Bayesian popularity as fallback. Recommends books as clickable cards inline in the chat.
- **Web agent:** Handles questions about current events, author biographies, or anything that benefits from live information. Searches the web and synthesizes results with source citations.
- **Docs agent:** Handles questions about the website itself — features, how recommendations work, troubleshooting, etc. Searches internal help documentation to answer.
- **Response agent:** Handles general small talk and conversational messages that don't fit any of the above.

The routing happens automatically — you don't need to choose a mode.

## Profile Access

When you are logged in, the chatbot can optionally access your profile (favorite subjects and recent interactions) to improve recommendation quality. This is controlled by the **"Use my profile"** toggle in the chat interface.

## Conversation Memory
- A short rolling history (default: 3 turns) is sent to the agent at each request.
- Up to 50 turns are stored in Redis, scoped to your browser session via a cookie.
- Memory expires after 2 days.
- If Redis is unavailable or cookies are blocked, the chatbot still works but will not remember earlier messages.
- You can clear your conversation history at any time using the **"Clear history"** button.
