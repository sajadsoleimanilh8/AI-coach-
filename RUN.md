# How to Run This Project

A complete, start-from-nothing guide for Windows. Follow it top to bottom and the
website will open in your browser.

You do **not** need Docker, an NVIDIA graphics card, or CUDA to run the website.

**Time:** about 20–30 minutes, most of it waiting for one download.
**Disk space:** about 7 GB free.

---

## 1. Install these three programs

Install all three before doing anything else. Accept the default options unless
noted.

| # | Program | Why | Download |
|---|---|---|---|
| 1 | **Git** | Downloads the project | <https://git-scm.com/downloads/win> |
| 2 | **Python 3.11** (64-bit) | Runs the backend and the AI code | <https://www.python.org/downloads/windows/> |
| 3 | **Node.js 20 LTS or newer** | Runs the website itself | <https://nodejs.org/en/download> |

**Python: one option matters.** On the first installer screen, tick
**"Add python.exe to PATH"** before clicking Install. If you miss it, re-run the
installer and choose Modify.

> Use **Python 3.11**, not 3.13. One dependency (`mediapipe 0.10.14`) has no
> installer for the newest Python versions. On the download page pick any
> **3.11.x** release, then "Windows installer (64-bit)".

**Not needed:** NVIDIA drivers, CUDA, Postgres, or Docker. The project detects
whether a graphics card is available and falls back to the CPU on its own.

### Check they worked

Open **PowerShell** (press `Windows`, type `powershell`, press Enter) and run:

```powershell
git --version
py -3.11 --version
node --version
npm --version
```

Four version numbers should appear. If any command says "not recognized", close
PowerShell, open it again, and retry. If it still fails, reinstall that program.

---

## 2. Download the project

> **Use a short folder path**, like `C:\ssc`. One of the packages installed in
> step 3 (PyTorch) contains deeply nested files, and Windows refuses to create
> them if the folder you started from is already long. This is the single most
> common installation failure.

```powershell
cd C:\
git clone https://github.com/sajadsoleimanilh8/AI-coach-.git ssc
cd C:\ssc
```

You are now inside the project folder. **Every command in this guide is run from
`C:\ssc`** unless it says otherwise.

---

## 3. Set up the backend

### 3.1 Create the Python environment

```powershell
cd C:\ssc
py -3.11 -m venv venv
```

This creates a `venv` folder that holds the project's Python packages. If a
`venv` folder already exists, skip this command.

### 3.2 Install the dependencies

```powershell
cd C:\ssc
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r backend\requirements-full.txt
```

**This downloads about 2.5 GB and takes 5–20 minutes** (roughly 5.4 GB once
installed). That is normal — it includes PyTorch. Let it finish. It ends with a
long `Successfully installed ...` line.

> Every command in this guide spells out `venv\Scripts\python.exe` on purpose, so
> nothing depends on "activating" the environment. You may activate it with
> `venv\Scripts\Activate.ps1` if you prefer, but it is not required and can be
> blocked by Windows security settings.

> Use `backend\requirements-full.txt`, **not** `backend\requirements.txt`. The
> plain one is missing packages the backend needs just to start.

### 3.3 Configuration

**There is nothing to configure.** No `.env` file is needed. The backend creates
its own database on first start and already defaults to the right ports.

(`backend\.env.example` exists as a reference list of optional settings. The
backend does not read it automatically — ignore it.)

### 3.4 Start the backend

```powershell
cd C:\ssc
venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000
```

Wait for:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Leave this window open.** The backend runs for as long as this window is open.

---

## 4. Set up the frontend

Open a **second** PowerShell window (leave the backend running in the first).

```powershell
cd C:\ssc\frontend\web
npm install
npm run dev
```

`npm install` takes under a minute. Then you should see:

```
VITE v6.4.3  ready in 800 ms
➜  Local:   http://127.0.0.1:3000/
```

Use the address Vite prints — **`127.0.0.1`, not `localhost`**. Section 9
explains why; the short version is that on Windows the name `localhost` points
at IPv6 first, while the backend listens on IPv4 only.

**Leave this window open too.**

---

## 5. Run everything — the short version

This is the recommended path. Two windows, that is all.

**PowerShell window 1 — backend:**

```powershell
cd C:\ssc
venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000
```

**PowerShell window 2 — frontend:**

```powershell
cd C:\ssc\frontend\web
npm run dev
```

(The very first time only, run `npm install` once before `npm run dev`.)

Then open this address in your browser:

> ## 👉 **http://127.0.0.1:3000**

Type it exactly — the digits, not the word `localhost`. `http://localhost:3000`
also opens, but every request stalls for about two seconds first, and the site
may report *"Core backend unreachable"*. Section 9 has the reason.

That is the whole system. The database is a file that is created automatically —
there is nothing to install or start for it.

---

## 6. Check that everything is working

Run this in a **third** PowerShell window:

```powershell
cd C:\ssc
venv\Scripts\python.exe -m scripts.dev_environment
```

It prints one line per service. You want to see:

```
SERVICES
    OK   core backend   http://localhost:8000
    OK   frontend       http://localhost:3000
```

Checklist:

```text
✓ Backend running      http://127.0.0.1:8000/health   shows {"status":"ok",...}
✓ API reachable        http://127.0.0.1:8000/docs     shows a documentation page
✓ Frontend running     http://127.0.0.1:3000          the website opens
✓ Database             created automatically at backend\database\sports_strategy.db
```

(The `dev_environment` output above prints these as `localhost` — that is the
script's own wording. Open them as `127.0.0.1` in a browser.)

`WARN` next to **redis**, **celery worker** or **NEXUS** is expected and fine —
those are optional (see section 8).

`FAIL  no matches in the database` is also expected on a fresh install: the
project ships without any match data.

---

## 7. Files that are not included in the download

The trained AI models and the sample match videos are too large for GitHub, so
they are **not** in the download. The website starts and runs without them.

If the project owner gives you these files, put them here:

```text
C:\ssc\models\yolo\player_v1\weights\best.pt
C:\ssc\models\yolo\ball_v1\weights\best.pt
C:\ssc\models\yolo\field_v1\weights\best.pt
C:\ssc\models\yolo\calibration_v1\weights\best.pt
C:\ssc\models\yolo\goalpost_v1\weights\best.pt

C:\ssc\samples\           <- put the sample .mp4 videos in this folder
```

Create any folder that does not exist. Restart the backend afterwards.

---

## 8. Optional extras

Skip this section entirely if you only need the website running.

### 8.1 Processing a newly uploaded video

The website runs without this. It is only needed to process a **new** video
upload automatically.

Install **Docker Desktop** — <https://www.docker.com/products/docker-desktop/> —
start it, and wait until its whale icon stops animating. Check it works:

```powershell
docker ps
```

Start the Redis service (first time only):

```powershell
docker run -d --name ssc-redis -p 6379:6379 --restart unless-stopped redis:7-alpine
```

Every time after that, just:

```powershell
docker start ssc-redis
```

Then, in a **new** PowerShell window, start the worker:

```powershell
cd C:\ssc
venv\Scripts\python.exe -m celery -A backend.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` is required on Windows. Wait for `celery@<your-pc> ready.`

**Without Docker**, you can process the most recent upload by hand instead:

```powershell
cd C:\ssc
venv\Scripts\python.exe -m scripts.run_job_local --latest
```

### 8.2 The Coach Chat tab

Every other part of the website works without this.

```powershell
cd C:\ssc
venv\Scripts\python.exe -m pip install -r nexus\requirements.txt
venv\Scripts\python.exe -m uvicorn nexus.api.main:app --port 8100
```

Written answers come from [Ollama](https://ollama.com/download) running locally.
Without Ollama, Coach Chat reports that it is unreachable and nothing else is
affected.

---

## 9. Common problems

| Problem | Cause | Fix |
|---|---|---|
| `pip install` ends with `OSError ... No such file or directory` and mentions **long paths** | The project folder path is too long | Move the project to a short path such as `C:\ssc` and run the install again |
| `py` or `python` **is not recognized** | Python was installed without "Add python.exe to PATH" | Re-run the Python installer, choose **Modify**, tick that box |
| `npm` **is not recognized** | Node.js not installed, or PowerShell was open during install | Close and reopen PowerShell; if it persists, reinstall Node.js |
| Backend: `ModuleNotFoundError: No module named 'fastapi'` (or `numpy`, `cv2`) | Dependencies not installed, or the wrong Python was used | Re-run step 3.2, using the full `venv\Scripts\python.exe` path |
| Backend: `ModuleNotFoundError: No module named 'configs'` | Started from the wrong folder | `cd C:\ssc` first, then start the backend |
| `ERROR: [Errno 10048] address already in use` (port 8000) | Something is already on port 8000 | Close the other backend window, or run `netstat -ano \| findstr :8000` and then `taskkill /PID <number> /F` |
| Vite **exits instead of starting** | Port 3000 is busy — the port is fixed on purpose | Free port 3000 (same commands as above, with `:3000`), then `npm run dev` |
| Website shows **"Failed to fetch"** everywhere | Backend is not running, or the site is not on port 3000 | Make sure window 1 shows `Application startup complete`, and use `http://127.0.0.1:3000` exactly |
| Website shows **"Core backend unreachable ... confirm this origin is listed in `CORS_ALLOWED_ORIGINS`"**, sometimes only on some tabs, sometimes after a delay | **Not a CORS problem.** You opened the site as `localhost:3000`. Windows resolves `localhost` to IPv6 `::1` first, but the backend listens on IPv4 `127.0.0.1` only — so the browser waits ~2 s for `::1` to fail, and sometimes gives up rather than retrying on IPv4. The site reports every failed connection with that one message, so it names CORS as a guess | Open **`http://127.0.0.1:3000`** instead. To confirm the cause: `http://127.0.0.1:8000/health` answers instantly while `http://[::1]:8000/health` fails after ~2 s |
| Website opens but every tab is **empty** | Normal — a fresh install has no match data | See section 7 |
| Upload stays at `queued`, or says `Processing queue unavailable` | Redis / Celery worker not running | Optional feature — see section 8.1 |
| `docker: ... container name "/ssc-redis" is already in use` | You ran the first-time command twice | Not an error — run `docker start ssc-redis` instead |

---

## 10. Stop everything

Click each PowerShell window and press **`Ctrl` + `C`**, then close it:

1. Frontend window
2. Backend window
3. Celery worker window, if you started one
4. NEXUS window, if you started one

If you started Docker in section 8.1:

```powershell
docker stop ssc-redis
```

Nothing else needs to be shut down. Your data stays in
`C:\ssc\backend\database\sports_strategy.db`.
