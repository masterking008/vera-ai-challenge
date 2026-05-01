# Vera Bot — Lean v3 (magicpin AI Challenge)

A minimal, adaptive WhatsApp merchant AI assistant that grounds every output in actual context rather than pattern-matching. Designed to score 45+/50 on the magicpin AI Challenge judge.

## Architecture

**Lean Bot Advantage**: No pre-computed signals or specialized prompts. Every message is composed by:
1. Extracting actual context at compose time (merchant stats, trigger payload, customer data)
2. Grounding output in that context via a single flexible system prompt
3. Adapting automatically when judge injects fresh data (new digest items, performance shifts, customer scopes)

This makes the bot robust to injected context changes during judging — it doesn't pattern-match the 30 canonical test pairs, it adapts to unseen data.

## Scoring Dimensions (0-10 each, total 50)

1. **Specificity (9-10/10)**: Cite TWO concrete facts — a number (CTR%, views, calls, price, count, days) + a source (JIDA, DCI, peer median, date, merchant name)
2. **Category Fit (9-10/10)**: Match voice exactly — Dentists: "Dr. [name]" + clinical vocab. Salons: warm + first-name. Restaurants: operator-to-operator. Gyms: coaching. Pharmacies: trustworthy.
3. **Merchant Fit (9-10/10)**: Owner/customer name in first 3 words + one specific merchant stat + if Hindi: 4-5 Hindi phrases throughout
4. **Trigger Relevance (9-10/10)**: WHY NOW signal in first 10 words — "Dr. X — JIDA just published:", "6 mahine ho gaye:", "Competitor 1.3 km away:", "Festival season:", "Tomorrow appointment:"
5. **Engagement (9-10/10)**: Binary CTA + consequence — "Reply YES — [benefit] by [date]" or "Reply 1 for [slot], 2 for [slot]"

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone and install**:
   ```bash
   git clone <repo>
   cd submission
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Run bot**:
   ```bash
   uvicorn bot:app --host 0.0.0.0 --port 8080
   ```

4. **Test health**:
   ```bash
   curl http://localhost:8080/v1/healthz
   ```

### Docker Deployment

**Local**:
```bash
docker-compose up --build
```

**Production** (Render, Railway, etc.):
```bash
docker build -t vera-bot .
docker run -p 8080:8080 -e OPENAI_API_KEY=<key> vera-bot
```

## API Endpoints

### Health Check
```bash
GET /v1/healthz
```
Returns bot status and context counts.

### Metadata
```bash
GET /v1/metadata
```
Returns team info, model version, approach.

### Push Context
```bash
POST /v1/context
{
  "scope": "category|merchant|trigger|customer",
  "context_id": "...",
  "version": 1,
  "payload": {...},
  "delivered_at": "2024-01-01T00:00:00Z"
}
```

### Tick (Compose Messages)
```bash
POST /v1/tick
{
  "now": "2024-01-01T00:00:00Z",
  "available_triggers": ["trg_001", "trg_002", ...]
}
```
Returns list of actions (messages to send).

### Reply (Handle Merchant Responses)
```bash
POST /v1/reply
{
  "conversation_id": "conv_...",
  "merchant_id": "m_001",
  "customer_id": "c_001",
  "from_role": "merchant",
  "message": "...",
  "received_at": "2024-01-01T00:00:00Z",
  "turn_number": 1
}
```

### Teardown
```bash
POST /v1/teardown
```
Clears all contexts and conversations (for testing).

## Key Features

- **Fast LLM calls**: 8-second timeout, max_tokens=300, compact prompts
- **Adaptive context extraction**: Dynamically extracts merchant stats, performance metrics, customer data, trigger payloads
- **Hindi support**: Automatically weaves 4-5 Hindi phrases for Hindi-speaking merchants
- **Auto-reply handling**: Detects and suppresses auto-replies (3x threshold)
- **Opt-out detection**: Respects customer opt-outs
- **Suppression keys**: Prevents duplicate messages via trigger-level suppression

## Performance

- **Average score**: 39-41/50 on canonical 30 test pairs
- **Target score**: 45+/50 with aggressive specificity, trigger relevance, engagement
- **Timeout handling**: Completes 30 test pairs in ~60 seconds without timeouts

## Testing

Run against judge simulator:
```bash
python judge_lean.py
```

## Deployment Checklist

- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Test `/v1/healthz` endpoint
- [ ] Verify `/v1/metadata` returns correct team info
- [ ] Run judge simulator locally
- [ ] Deploy to production (Render, Railway, etc.)
- [ ] Monitor logs for LLM errors or timeouts

## File Structure

```
submission/
├── bot.py                 # Main bot (lean v3)
├── requirements.txt       # Python dependencies
├── .env                   # Environment config (add OPENAI_API_KEY)
├── .gitignore            # Git ignore rules
├── Dockerfile            # Container image
├── docker-compose.yml    # Local dev environment
└── README.md             # This file
```

## Troubleshooting

**LLM timeout**: Reduce `max_tokens` or increase timeout in `compose_message()`.

**High latency**: Check OpenAI API status or reduce context extraction overhead.

**Auto-reply loops**: Increase `auto_reply_counts` threshold or add more opt-out phrases.

## License

Proprietary — magicpin AI Challenge submission.

## Contact

Team: Vera Challenger  
Email: dinesh@dins.in
# vera-ai-challenge
