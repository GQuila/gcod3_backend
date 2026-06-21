"""
GCOD3 - Backend API
-------------------
Premium SaaS platform for GCOD3 (https://www.gcod3.com).

Provides:
  - JWT email/password auth + Google OAuth callback exchange
  - Lead capture (contact form + AI chat)
  - Client project tracking
  - G-Bot AI concierge (streaming) + AI tool hub
  - Admin endpoints (users, leads, stats)

Designed, built and engineered by Gian Q.
Stack: FastAPI + Motor (Mongo) + JWT + Claude Sonnet (Anthropic).
"""
import os
import uuid
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Annotated

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Response, Cookie, Header, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr, ConfigDict
import bcrypt
import jwt
import httpx

from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# ─── Configuration ──────────────────────────────────────────────────────────
def _require_env(name: str, hint: str) -> str:
    """Fail fast with a human-readable message if a required env var is missing.
    Render / Hostinger / any host will surface this in the logs instead of a
    cryptic KeyError stack trace."""
    val = os.environ.get(name, '').strip()
    if not val:
        raise RuntimeError(
            f"\n\n[GCOD3 startup] Missing required environment variable: {name}\n"
            f"  -> {hint}\n"
            f"  On Render: dashboard -> your service -> Environment -> Add Environment Variable.\n"
        )
    return val

MONGO_URL = _require_env(
    'MONGO_URL',
    "Set your MongoDB Atlas connection string (mongodb+srv://user:pass@cluster.../...).",
)
DB_NAME = os.environ.get('DB_NAME', 'gcod3_production')
JWT_SECRET = _require_env(
    'JWT_SECRET',
    "Set a long random string. Generate one with: openssl rand -hex 48",
)
JWT_ALGO = os.environ.get('JWT_ALGO', 'HS256')
# AI: use the official public OpenAI SDK against the Emergent universal-key
# proxy (OpenAI-compatible). Same key (EMERGENT_LLM_KEY) routes to Claude /
# OpenAI / Gemini behind the scenes — no private packages required, so the
# build is clean on Render. Override AI_BASE_URL + LLM_API_KEY at any time
# to use a direct provider (e.g. raw OpenAI or Anthropic-OpenAI gateway).
LLM_API_KEY = os.environ.get('LLM_API_KEY', '') or os.environ.get('EMERGENT_LLM_KEY', '')
AI_BASE_URL = os.environ.get('AI_BASE_URL', 'https://integrations.emergentagent.com/llm')
AI_MODEL_NAME = os.environ.get('AI_MODEL', 'claude-sonnet-4-5-20250929')
AI_ENABLED = bool(LLM_API_KEY)

# Shared async client (connection pooling across requests).
ai_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=AI_BASE_URL) if AI_ENABLED else None

# SMTP / Contact email — optional, gracefully degrades if not configured
SMTP_HOST = os.environ.get('SMTP_HOST', '')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
CONTACT_TO = os.environ.get('CONTACT_TO', 'gcod3web@gmail.com')
CONTACT_FROM = os.environ.get('CONTACT_FROM', SMTP_USER or 'noreply@gcod3.com')

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="GCOD3 API", version="1.0.0")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gcod3")


def send_contact_email(payload: dict) -> bool:
    """Deliver a contact-form / lead notification to gcod3web@gmail.com.

    Tries STARTTLS first (port 587), falls back to implicit SSL (port 465) if
    STARTTLS is blocked by the host. Returns True on success, False on any
    failure (we never block the user response on email — the lead is always
    persisted to Mongo first).
    """
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        logger.warning(
            "SMTP not configured. Contact email skipped. Missing env vars: %s",
            [n for n, v in [('SMTP_HOST', SMTP_HOST), ('SMTP_USER', SMTP_USER), ('SMTP_PASSWORD', SMTP_PASSWORD)] if not v],
        )
        return False

    subject = f"[GCOD3] New lead: {payload.get('name', 'Unknown')}"
    body_lines = [
        "A new enquiry just arrived on gcod3.com:",
        "",
        f"Name:     {payload.get('name', '-')}",
        f"Email:    {payload.get('email', '-')}",
        f"Phone:    {payload.get('phone', '-')}",
        f"Company:  {payload.get('company', '-')}",
        f"Subject:  {payload.get('subject', '-')}",
        f"Source:   {payload.get('source', 'contact_form')}",
        "",
        "Message:",
        payload.get('message') or payload.get('project_details') or '(no message)',
        "",
        "—",
        f"Logged at {now_iso()}",
    ]
    msg = MIMEMultipart()
    msg['From'] = CONTACT_FROM
    msg['To'] = CONTACT_TO
    msg['Reply-To'] = payload.get('email') or CONTACT_FROM
    msg['Subject'] = subject
    msg.attach(MIMEText("\n".join(body_lines), 'plain'))

    # Try STARTTLS (587), then implicit SSL (465). Render's free tier
    # sometimes blocks 587 outbound — 465 SSL usually goes through.
    attempts = [(SMTP_PORT, 'starttls')]
    if SMTP_PORT != 465:
        attempts.append((465, 'ssl'))

    last_err = None
    for port, mode in attempts:
        try:
            if mode == 'ssl':
                import ssl
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(SMTP_HOST, port, timeout=15, context=ctx) as s:
                    s.login(SMTP_USER, SMTP_PASSWORD)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(SMTP_HOST, port, timeout=15) as s:
                    s.ehlo()
                    s.starttls()
                    s.ehlo()
                    s.login(SMTP_USER, SMTP_PASSWORD)
                    s.send_message(msg)
            logger.info("Contact email sent to %s via %s:%s", CONTACT_TO, SMTP_HOST, port)
            return True
        except Exception as exc:
            last_err = exc
            logger.warning("SMTP %s on port %s failed: %s", mode, port, exc)
    logger.error("All SMTP attempts failed. Last error: %s", last_err)
    return False

# ─── Models ─────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    role: str = "user"
    plan: str = "free"
    picture: Optional[str] = None
    created_at: Optional[str] = None

class LeadCreate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    budget: Optional[str] = None
    project_details: Optional[str] = None
    source: str = "contact_form"  # contact_form | ai_chat
    message: Optional[str] = None

class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    company: Optional[str] = None
    subject: Optional[str] = None
    message: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    # Optional lead-capture meta from the widget
    visitor_name: Optional[str] = None
    visitor_email: Optional[str] = None

class AIToolRequest(BaseModel):
    tool: str  # content_writer | proposal | email | idea | marketing | code
    prompt: str
    tone: Optional[str] = "professional"

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TicketCreate(BaseModel):
    subject: str
    message: str
    priority: str = "normal"   # low | normal | high | urgent
    category: Optional[str] = None


class TicketReply(BaseModel):
    message: str

# ─── Helpers ────────────────────────────────────────────────────────────────
def hash_password(pwd: str) -> str:
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def verify_password(pwd: str, hashed: str) -> bool:
    return bcrypt.checkpw(pwd.encode(), hashed.encode())

def create_jwt(user_id: str, days: int = 7) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=days),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def decode_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload.get("sub")
    except Exception:
        return None

async def get_user_from_token(token: str) -> Optional[dict]:
    # Try JWT first
    uid = decode_jwt(token)
    if uid:
        user = await db.users.find_one({"user_id": uid}, {"_id": 0})
        if user:
            return user
    # Try Emergent session token
    sess = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if sess:
        expires_at = sess.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at and expires_at > datetime.now(timezone.utc):
            user = await db.users.find_one({"user_id": sess["user_id"]}, {"_id": 0})
            if user:
                return user
    return None

async def current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session_token: Optional[str] = Cookie(default=None),
) -> dict:
    token = None
    if session_token:
        token = session_token
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = await get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

async def admin_only(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def serialize_user(u: dict) -> dict:
    return {k: v for k, v in u.items() if k not in ("password_hash", "_id")}

# ─── Auth Routes ────────────────────────────────────────────────────────────
@api.post("/auth/signup")
async def signup(payload: UserCreate, response: Response):
    existing = await db.users.find_one({"email": payload.email.lower()}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    doc = {
        "user_id": user_id,
        "email": payload.email.lower(),
        "name": payload.name,
        "password_hash": hash_password(payload.password),
        "role": "user",
        "plan": "free",
        "auth_provider": "password",
        "created_at": now_iso(),
    }
    await db.users.insert_one(doc)
    token = create_jwt(user_id)
    response.set_cookie("session_token", token, httponly=True, secure=True, samesite="none", path="/", max_age=60*60*24*7)
    return {"token": token, "user": serialize_user(doc)}

@api.post("/auth/login")
async def login(payload: UserLogin, response: Response):
    user = await db.users.find_one({"email": payload.email.lower()}, {"_id": 0})
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_jwt(user["user_id"])
    response.set_cookie("session_token", token, httponly=True, secure=True, samesite="none", path="/", max_age=60*60*24*7)
    return {"token": token, "user": serialize_user(user)}

@api.post("/auth/google/session")
async def google_session(request: Request, response: Response):
    """Exchange Emergent OAuth session_id for our user session."""
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid OAuth session")
    data = r.json()
    email = data["email"].lower()
    name = data.get("name", email)
    picture = data.get("picture")
    session_token = data["session_token"]

    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "role": "user",
            "plan": "free",
            "auth_provider": "google",
            "created_at": now_iso(),
        }
        await db.users.insert_one(user)
    else:
        # update picture/name if changed
        await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"name": name, "picture": picture}})

    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one({
        "session_token": session_token,
        "user_id": user["user_id"],
        "expires_at": expires_at.isoformat(),
        "created_at": now_iso(),
    })
    response.set_cookie("session_token", session_token, httponly=True, secure=True, samesite="none", path="/", max_age=60*60*24*7)
    return {"user": serialize_user(user)}

@api.get("/auth/me")
async def me(user: dict = Depends(current_user)):
    return serialize_user(user)

@api.post("/auth/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(default=None)):
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    response.delete_cookie("session_token", path="/")
    return {"ok": True}

# ─── Leads ──────────────────────────────────────────────────────────────────
@api.post("/leads")
async def create_lead(payload: LeadCreate):
    doc = payload.model_dump()
    doc["lead_id"] = f"lead_{uuid.uuid4().hex[:12]}"
    doc["created_at"] = now_iso()
    doc["status"] = "new"
    await db.leads.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api.post("/contact")
async def contact_submit(payload: ContactMessage, background: BackgroundTasks):
    doc = payload.model_dump()
    doc["lead_id"] = f"lead_{uuid.uuid4().hex[:12]}"
    doc["created_at"] = now_iso()
    doc["status"] = "new"
    doc["source"] = "contact_form"
    await db.leads.insert_one(doc)
    doc.pop("_id", None)
    # Fire-and-forget email — never blocks the response
    background.add_task(send_contact_email, doc)
    return {"ok": True, "lead_id": doc["lead_id"]}

@api.get("/leads")
async def list_leads(_admin: dict = Depends(admin_only)):
    leads = await db.leads.find({}, {"_id": 0}).sort("created_at", -1).to_list(500)
    return leads

# ─── AI Assistant (streaming) ───────────────────────────────────────────────
GCOD3_SYSTEM_PROMPT = """You are G-Bot, the AI concierge for GCOD3 - a premium technology company specializing in:
- Custom Software Development
- Mobile App Development  
- Website Development
- AI Solutions & Chatbots
- Business Automation
- Cloud Solutions
- UI/UX Design
- Enterprise Systems

Your role:
1. Answer visitor questions clearly and concisely (2-4 short paragraphs max).
2. Explain GCOD3 services with confidence and warmth.
3. Recommend tailored solutions based on the visitor's needs.
4. Gently capture lead info: name, email, phone, company, budget, project details - only after rapport.
5. Be elite, futuristic, friendly. Never robotic. Use crisp sentences.

If asked about pricing: Starter is for MVPs, Professional for growing businesses, Enterprise for custom & dedicated solutions. Always invite the visitor to "Schedule a Consultation".
Never invent specific prices - always say "let's schedule a quick call so our team can give you accurate pricing."
"""

@api.post("/ai/chat/stream")
async def ai_chat_stream(payload: ChatRequest):
    sid = payload.session_id or f"chat_{uuid.uuid4().hex[:12]}"

    # persist lead if visitor info provided
    if payload.visitor_email or payload.visitor_name:
        await db.leads.update_one(
            {"chat_session_id": sid},
            {"$set": {
                "chat_session_id": sid,
                "name": payload.visitor_name,
                "email": (payload.visitor_email or "").lower() or None,
                "source": "ai_chat",
                "updated_at": now_iso(),
            }, "$setOnInsert": {
                "lead_id": f"lead_{uuid.uuid4().hex[:12]}",
                "created_at": now_iso(),
                "status": "new",
            }},
            upsert=True,
        )

    # persist user message
    await db.ai_messages.insert_one({
        "session_id": sid,
        "role": "user",
        "content": payload.message,
        "created_at": now_iso(),
    })

    async def event_gen():
        import json as _json
        # send session_id first
        yield f"data: {_json.dumps({'session_id': sid})}\n\n"
        if not ai_client:
            yield f"data: {_json.dumps({'error': 'AI is not configured on the server.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Load prior conversation (multi-turn memory). OpenAI-compatible API:
        # system + alternating user/assistant turns. We already inserted the
        # current user message above, so it's the last item.
        history_docs = await db.ai_messages.find(
            {"session_id": sid},
            {"_id": 0, "role": 1, "content": 1, "created_at": 1},
        ).sort("created_at", 1).to_list(40)
        messages = [{"role": "system", "content": GCOD3_SYSTEM_PROMPT}]
        for m in history_docs:
            messages.append({"role": m["role"], "content": m["content"]})

        full_text = ""
        try:
            stream = await ai_client.chat.completions.create(
                model=AI_MODEL_NAME,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                full_text += delta
                yield f"data: {_json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            logger.exception("AI stream failed")
            yield f"data: {_json.dumps({'error': str(e)[:160]})}\n\n"

        # persist assistant message
        if full_text:
            await db.ai_messages.insert_one({
                "session_id": sid,
                "role": "assistant",
                "content": full_text,
                "created_at": now_iso(),
            })
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@api.post("/ai/tool")
async def ai_tool(payload: AIToolRequest, user: dict = Depends(current_user)):
    """One-shot AI tools for the dashboard."""
    sys_map = {
        "content_writer": "You are an elite content writer. Produce premium copy in the requested tone. Output Markdown.",
        "proposal": "You are a top-tier business proposal writer. Generate a complete client proposal in Markdown with sections: Executive Summary, Scope, Deliverables, Timeline, Pricing, Next Steps.",
        "email": "You are a master email copywriter. Generate a polished, concise email. Provide Subject + Body.",
        "idea": "You are an elite startup strategist. Generate a complete business idea breakdown: Problem, Solution, Target Market, Revenue Model, Go-to-Market, Risks.",
        "marketing": "You are a world-class marketing strategist. Generate a complete marketing campaign: Hook, Channels, Copy variants, KPIs.",
        "code": "You are a senior software engineer. Provide clean, production-quality code with brief explanation. Output in Markdown with code blocks.",
        "website_content": "You are an elite website copywriter. Generate full sections: Hero headline + sub, 3 feature bullets, CTA. Use Markdown.",
    }
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not configured on the server.")
    system = sys_map.get(payload.tool, sys_map["content_writer"])
    full_prompt = f"Tone: {payload.tone}.\n\nTask: {payload.prompt}"
    output = ""
    try:
        stream = await ai_client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                output += delta
    except Exception as e:
        logger.exception("AI tool failed")
        raise HTTPException(status_code=500, detail=str(e))
    # persist usage
    await db.ai_usage.insert_one({
        "user_id": user["user_id"],
        "tool": payload.tool,
        "prompt": payload.prompt,
        "output_len": len(output),
        "created_at": now_iso(),
    })
    return {"output": output}

# ─── Projects ───────────────────────────────────────────────────────────────
@api.post("/projects")
async def create_project(payload: ProjectCreate, user: dict = Depends(current_user)):
    pid = f"proj_{uuid.uuid4().hex[:12]}"
    doc = {
        "project_id": pid,
        "user_id": user["user_id"],
        "name": payload.name,
        "description": payload.description or "",
        "status": "planning",
        "progress": 0,
        "milestones": [],
        "created_at": now_iso(),
    }
    await db.projects.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api.get("/projects")
async def my_projects(user: dict = Depends(current_user)):
    items = await db.projects.find({"user_id": user["user_id"]}, {"_id": 0}).sort("created_at", -1).to_list(200)
    return items


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None      # planning | active | review | shipped | paused
    progress: Optional[int] = None    # 0-100


class MilestoneCreate(BaseModel):
    title: str
    due_date: Optional[str] = None
    done: bool = False


@api.patch("/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectUpdate, user: dict = Depends(current_user)):
    proj = await db.projects.find_one({"project_id": project_id, "user_id": user["user_id"]}, {"_id": 0})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    if 'progress' in updates:
        updates['progress'] = max(0, min(100, int(updates['progress'])))
    updates['updated_at'] = now_iso()
    await db.projects.update_one({"project_id": project_id}, {"$set": updates})
    proj.update(updates)
    return proj


@api.delete("/projects/{project_id}")
async def delete_project(project_id: str, user: dict = Depends(current_user)):
    res = await db.projects.delete_one({"project_id": project_id, "user_id": user["user_id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}


@api.post("/projects/{project_id}/milestones")
async def add_milestone(project_id: str, payload: MilestoneCreate, user: dict = Depends(current_user)):
    proj = await db.projects.find_one({"project_id": project_id, "user_id": user["user_id"]}, {"_id": 0})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    milestone = {
        "id": uuid.uuid4().hex[:10],
        "title": payload.title,
        "due_date": payload.due_date,
        "done": payload.done,
        "created_at": now_iso(),
    }
    await db.projects.update_one(
        {"project_id": project_id},
        {"$push": {"milestones": milestone}, "$set": {"updated_at": now_iso()}},
    )
    return milestone


@api.patch("/projects/{project_id}/milestones/{milestone_id}")
async def toggle_milestone(project_id: str, milestone_id: str, user: dict = Depends(current_user)):
    proj = await db.projects.find_one({"project_id": project_id, "user_id": user["user_id"]}, {"_id": 0})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    milestones = proj.get("milestones", [])
    found = False
    for m in milestones:
        if m.get("id") == milestone_id:
            m["done"] = not m.get("done", False)
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Milestone not found")
    # recompute progress from milestones
    if milestones:
        done = sum(1 for m in milestones if m.get("done"))
        progress = int(done / len(milestones) * 100)
    else:
        progress = proj.get("progress", 0)
    await db.projects.update_one(
        {"project_id": project_id},
        {"$set": {"milestones": milestones, "progress": progress, "updated_at": now_iso()}},
    )
    return {"ok": True, "progress": progress}


# ─── Dashboard (user-scoped) ────────────────────────────────────────────────
@api.get("/dashboard/overview")
async def dashboard_overview(user: dict = Depends(current_user)):
    """Powerful user-scoped overview for the dashboard."""
    uid = user["user_id"]
    projects = await db.projects.find({"user_id": uid}, {"_id": 0}).sort("created_at", -1).to_list(50)
    ai_usage_count = await db.ai_usage.count_documents({"user_id": uid})
    chat_msgs = await db.ai_messages.count_documents({})  # global chat usage
    recent_ai = await db.ai_usage.find({"user_id": uid}, {"_id": 0, "prompt": 1, "tool": 1, "created_at": 1}).sort("created_at", -1).to_list(5)

    # status breakdown for projects
    by_status = {"planning": 0, "active": 0, "review": 0, "shipped": 0, "paused": 0}
    total_progress = 0
    for p in projects:
        st = p.get("status", "planning")
        by_status[st] = by_status.get(st, 0) + 1
        total_progress += p.get("progress", 0)
    avg_progress = int(total_progress / len(projects)) if projects else 0

    return {
        "user": serialize_user(user),
        "stats": {
            "projects_total": len(projects),
            "projects_active": by_status.get("active", 0),
            "projects_shipped": by_status.get("shipped", 0),
            "ai_runs": ai_usage_count,
            "ai_chat_msgs": chat_msgs,
            "avg_progress": avg_progress,
        },
        "by_status": by_status,
        "recent_projects": projects[:5],
        "recent_ai": recent_ai,
    }


@api.get("/dashboard/activity")
async def dashboard_activity(user: dict = Depends(current_user)):
    """Activity feed — last 20 events across projects + AI usage."""
    uid = user["user_id"]
    events = []
    async for p in db.projects.find({"user_id": uid}, {"_id": 0}).sort("created_at", -1).limit(10):
        events.append({
            "type": "project_created",
            "title": f"Project '{p['name']}' created",
            "meta": p.get("status"),
            "at": p["created_at"],
        })
    async for a in db.ai_usage.find({"user_id": uid}, {"_id": 0}).sort("created_at", -1).limit(15):
        events.append({
            "type": "ai_used",
            "title": f"Used AI tool: {a.get('tool')}",
            "meta": (a.get("prompt") or "")[:80],
            "at": a["created_at"],
        })
    async for t in db.tickets.find({"user_id": uid}, {"_id": 0}).sort("created_at", -1).limit(10):
        events.append({
            "type": "ticket_opened",
            "title": f"Ticket opened: {t.get('subject')}",
            "meta": t.get("status"),
            "at": t["created_at"],
        })
    events.sort(key=lambda e: e["at"], reverse=True)
    return events[:20]


# ─── Support tickets ────────────────────────────────────────────────────────
@api.post("/tickets")
async def create_ticket(payload: TicketCreate, background: BackgroundTasks,
                        user: dict = Depends(current_user)):
    tid = f"tkt_{uuid.uuid4().hex[:10].upper()}"
    doc = {
        "ticket_id": tid,
        "user_id": user["user_id"],
        "user_email": user["email"],
        "user_name": user.get("name"),
        "subject": payload.subject,
        "category": payload.category or "general",
        "priority": payload.priority,
        "status": "open",
        "replies": [{"from": "user", "message": payload.message, "at": now_iso()}],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    await db.tickets.insert_one(doc)
    doc.pop("_id", None)
    background.add_task(send_contact_email, {
        "name": user.get("name"),
        "email": user["email"],
        "subject": f"[Ticket {tid}] {payload.subject}",
        "message": f"Priority: {payload.priority}\nCategory: {payload.category or 'general'}\n\n{payload.message}",
        "source": "support_ticket",
    })
    return doc


@api.get("/tickets")
async def my_tickets(user: dict = Depends(current_user)):
    items = await db.tickets.find({"user_id": user["user_id"]}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return items


@api.post("/tickets/{ticket_id}/reply")
async def reply_ticket(ticket_id: str, payload: TicketReply, user: dict = Depends(current_user)):
    t = await db.tickets.find_one({"ticket_id": ticket_id, "user_id": user["user_id"]}, {"_id": 0})
    if not t:
        raise HTTPException(status_code=404, detail="Ticket not found")
    reply = {"from": "user", "message": payload.message, "at": now_iso()}
    await db.tickets.update_one(
        {"ticket_id": ticket_id},
        {"$push": {"replies": reply}, "$set": {"status": "open", "updated_at": now_iso()}},
    )
    return reply


@api.patch("/tickets/{ticket_id}/close")
async def close_ticket(ticket_id: str, user: dict = Depends(current_user)):
    res = await db.tickets.update_one(
        {"ticket_id": ticket_id, "user_id": user["user_id"]},
        {"$set": {"status": "closed", "closed_at": now_iso(), "updated_at": now_iso()}},
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"ok": True}


# ─── Admin ──────────────────────────────────────────────────────────────────
@api.get("/admin/stats")
async def admin_stats(_admin: dict = Depends(admin_only)):
    users = await db.users.count_documents({})
    leads = await db.leads.count_documents({})
    projects = await db.projects.count_documents({})
    ai_msgs = await db.ai_messages.count_documents({})
    return {"users": users, "leads": leads, "projects": projects, "ai_messages": ai_msgs}

@api.get("/admin/users")
async def admin_users(_admin: dict = Depends(admin_only)):
    users = await db.users.find({}, {"_id": 0, "password_hash": 0}).sort("created_at", -1).to_list(500)
    return users

# ─── Health ─────────────────────────────────────────────────────────────────
@api.get("/")
async def root():
    return {"service": "GCOD3 API", "status": "ok", "time": now_iso()}

# Include router
app.include_router(api)

# ─── CORS ──────────────────────────────────────────────────────────────────
_cors_raw = os.environ.get('CORS_ORIGINS', '').strip()
if not _cors_raw or _cors_raw == '*':
    # No explicit origins set → permissive mode for first-deploy convenience.
    # `allow_credentials=False` is REQUIRED by the CORS spec when origin = "*",
    # otherwise browsers reject every preflight (which silently breaks signup).
    # Auth still works because we send JWTs via the Authorization header.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.warning("CORS: no CORS_ORIGINS set → using permissive mode (*, credentials=off). Set CORS_ORIGINS env on Render to lock this down.")
else:
    origins = [o.strip() for o in _cors_raw.split(',') if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS: locked to origins: %s", origins)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
