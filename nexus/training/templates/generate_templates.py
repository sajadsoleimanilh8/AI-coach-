"""Regenerates the five behavioural template files."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

TEMPLATES_DIR = Path(__file__).parent
SEED = 20260811



def _row(task_type: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"task_type": task_type, "messages": messages}


def _cycle(items: list[str], index: int) -> str:
    return items[index % len(items)]


_WORD = re.compile(r"[a-z0-9]+")

_NEAR_DUPLICATE_THRESHOLD = 0.9


def _prompt_text(row: dict[str, Any]) -> str:
    return " ".join(m["content"] for m in row["messages"] if m["role"] != "assistant")


def _write(rows: list[dict[str, Any]], name: str) -> tuple[int, int, int]:
    """Writes rows, dropping exact-prompt repeats AND near-duplicates."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    kept_tokens: list[set[str]] = []
    exact_dropped = 0
    near_dropped = 0

    for row in rows:
        signature = "\n".join(
            f"{m['role']}:{m['content']}" for m in row["messages"] if m["role"] != "assistant"
        )
        if signature in seen:
            exact_dropped += 1
            continue

        tokens = set(_WORD.findall(_prompt_text(row).lower()))
        collides = False
        for existing in kept_tokens:
            if not tokens or not existing:
                continue
            if abs(len(tokens) - len(existing)) > max(len(tokens), len(existing)) * 0.35:
                continue
            if len(tokens & existing) / len(tokens | existing) > _NEAR_DUPLICATE_THRESHOLD:
                collides = True
                break
        if collides:
            near_dropped += 1
            continue

        seen.add(signature)
        kept.append(row)
        kept_tokens.append(tokens)

    path = TEMPLATES_DIR / name
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept), exact_dropped, near_dropped



_TOOL_SYSTEM = (
    "You have these tools: web_search(query), python(code), files(path, mode), "
    "database(sql), github(action, repo). Call exactly one tool per step, ask for "
    "what you are missing if the goal is ambiguous, answer directly if no tool "
    "applies, and decline if the request cannot be carried out."
)


def _tool_call(tool: str, arguments: dict[str, str]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False)


_WEB_SEARCH_GOALS = [
    ("the current away form of the side we play on Saturday", "away form {team} current season"),
    ("what formation {team} used in their last three league games", "{team} formation last three matches"),
    ("whether {player} is suspended for the next round", "{player} suspension status next fixture"),
    ("the referee appointed for our next away fixture", "referee appointment {team} next away fixture"),
    ("recent reporting on {team}'s pressing scheme", "{team} pressing scheme analysis"),
    ("what the weather is forecast to do at kickoff on Saturday", "weather forecast {city} Saturday afternoon"),
    ("who {team} signed in the winter window", "{team} winter transfer window signings"),
    ("the league's current disciplinary threshold for a ban", "league disciplinary suspension threshold accumulated cards"),
    ("when the transfer window shuts in this division", "transfer window closing date division"),
    ("whether {team}'s main striker is fit again", "{team} striker injury return date"),
    ("what pundits made of our defensive shape last weekend", "{team} defensive shape analysis last weekend"),
    ("the kickoff time for the cup replay", "{team} cup replay kickoff time"),
    ("how many away wins {team} have managed this season", "{team} away wins this season record"),
    ("whether the pitch at {city} is artificial or grass", "{city} stadium pitch surface artificial grass"),
    ("recent injury news around {player}", "{player} injury news latest"),
    ("the head-to-head record between us and {team}", "head to head record {team} last ten meetings"),
    ("what time the squad list has to be submitted", "league squad list submission deadline rules"),
    ("whether VAR is used at this level of the competition", "VAR usage lower division competition rules"),
    ("attendance figures for {team}'s home games", "{team} average home attendance this season"),
    ("who is on loan at {team} right now", "{team} current loan players list"),
]

_PYTHON_GOALS = [
    ("the mean of these press-resistance scores: {numbers}", "print(sum([{numbers}])/len([{numbers}]))"),
    ("the standard deviation of {numbers}", "import statistics\nprint(statistics.pstdev([{numbers}]))"),
    ("the median of {numbers}", "import statistics\nprint(statistics.median([{numbers}]))"),
    ("how many of {numbers} are below 0.5", "print(sum(1 for v in [{numbers}] if v < 0.5))"),
    ("the spread between the highest and lowest of {numbers}", "vals = [{numbers}]\nprint(max(vals) - min(vals))"),
    ("these scores {numbers} ranked highest to lowest", "print(sorted([{numbers}], reverse=True))"),
    ("what proportion of {numbers} clear 0.6", "vals = [{numbers}]\nprint(sum(1 for v in vals if v > 0.6) / len(vals))"),
    ("the variance of {numbers}", "import statistics\nprint(statistics.pvariance([{numbers}]))"),
    ("{numbers} converted to percentages", "print([f'{{v:.0%}}' for v in [{numbers}]])"),
    ("the sum of {numbers} rounded to two places", "print(round(sum([{numbers}]), 2))"),
    ("whether any of {numbers} sit below the 0.45 weakness line", "print(any(v < 0.45 for v in [{numbers}]))"),
    ("the gap between the mean and the median of {numbers}", "import statistics\nvals = [{numbers}]\nprint(statistics.mean(vals) - statistics.median(vals))"),
    ("{numbers} normalised so the largest is 1.0", "vals = [{numbers}]\nm = max(vals)\nprint([round(v/m, 3) for v in vals])"),
]

_FILES_GOALS = [
    ("read the tactical notes I saved at {path}", "{path}", "read"),
    ("check what is in {path}", "{path}", "read"),
    ("save this session plan to {path}", "{path}", "write"),
    ("append tonight's observations to {path}", "{path}", "append"),
    ("open the training log at {path}", "{path}", "read"),
    ("pull up whatever I wrote in {path}", "{path}", "read"),
    ("write the squad shortlist into {path}", "{path}", "write"),
    ("add a line about the set-piece drill to {path}", "{path}", "append"),
    ("show me the contents of {path} before I overwrite it", "{path}", "read"),
    ("dump this week's review into {path}", "{path}", "write"),
    ("stick tonight's notes on the end of {path}", "{path}", "append"),
    ("look at what is stored in {path}", "{path}", "read"),
]

_DATABASE_GOALS = [
    ("how many matches we have metrics for", "SELECT COUNT(*) FROM matches"),
    ("the average compactness score across all matches", "SELECT AVG(value_numeric) FROM team_metrics WHERE metric_name = 'compactness_score'"),
    ("which matches are missing a pressing intensity score", "SELECT match_id FROM team_metrics WHERE metric_name = 'pressing_intensity_score' AND value_numeric IS NULL"),
    ("the five most recent matches in the database", "SELECT match_id, played_at FROM matches ORDER BY played_at DESC LIMIT 5"),
    ("every player metric flagged low_upstream_confidence", "SELECT player_id, metric_name FROM player_metrics WHERE confidence = 'low_upstream_confidence'"),
    ("how many player metrics we hold per match", "SELECT match_id, COUNT(*) FROM player_metrics GROUP BY match_id"),
    ("the highest press resistance score we have recorded", "SELECT MAX(value_numeric) FROM player_metrics WHERE metric_name = 'press_resistance_score'"),
    ("which players appear in more than ten matches", "SELECT player_id FROM player_metrics GROUP BY player_id HAVING COUNT(DISTINCT match_id) > 10"),
    ("a count of metrics grouped by confidence level", "SELECT confidence, COUNT(*) FROM player_metrics GROUP BY confidence"),
    ("the earliest match we hold any data for", "SELECT match_id, played_at FROM matches ORDER BY played_at ASC LIMIT 1"),
    ("every match where formation stability was computed", "SELECT match_id FROM team_metrics WHERE metric_name = 'formation_stability_score' AND value_numeric IS NOT NULL"),
    ("average decision making per player across the season", "SELECT player_id, AVG(value_numeric) FROM player_metrics WHERE metric_name = 'decision_making_score' GROUP BY player_id"),
    ("how many distinct metric names exist in the player table", "SELECT COUNT(DISTINCT metric_name) FROM player_metrics"),
    ("matches with fewer than three metrics recorded", "SELECT match_id FROM team_metrics GROUP BY match_id HAVING COUNT(*) < 3"),
]

_GITHUB_GOALS = [
    ("what issues are open on {repo}", "list_issues", "{repo}"),
    ("open an issue on {repo} about the missing metric", "create_issue", "{repo}"),
    ("the open pull requests on {repo}", "list_pull_requests", "{repo}"),
    ("whether the last CI run on {repo} passed", "list_workflow_runs", "{repo}"),
    ("the commit history on {repo} for this week", "list_commits", "{repo}"),
    ("who has write access to {repo}", "list_collaborators", "{repo}"),
    ("whether anyone has reviewed the open PR on {repo}", "list_reviews", "{repo}"),
    ("the release notes for the last tag on {repo}", "list_releases", "{repo}"),
    ("which branches on {repo} are stale", "list_branches", "{repo}"),
]

_TEAMS = ["Northbridge United", "Calder Rovers", "Eastvale FC", "Port Aldrin", "Kingsmoor Athletic",
          "Rowan Town", "Ashcombe City", "Delford Wanderers"]
_PLAYERS = ["Ruiz", "Okafor", "Lindqvist", "Bahri", "Novak", "Mensah", "Castellanos", "Dragomir"]
_CITIES = ["Northbridge", "Calder", "Eastvale", "Port Aldrin", "Kingsmoor", "Rowan"]
_PATHS = ["notes/opposition-report.md", "plans/session-tuesday.md", "logs/training-week12.txt",
          "notes/set-pieces.md", "reports/u21-review.md", "logs/match-118-observations.txt"]
_REPOS = ["club/analysis-pipeline", "club/nexus-config", "club/vision-tracking", "club/match-reports"]
_NUMBER_SETS = ["0.71, 0.44, 0.62, 0.58", "0.31, 0.29, 0.47", "0.88, 0.79, 0.91, 0.85, 0.77",
                "0.5, 0.5, 0.52, 0.49", "0.12, 0.64, 0.38, 0.71, 0.22", "0.66, 0.61"]

_AMBIGUOUS_GOALS = [
    ("Pull the stats for the match.", "Which match? I have several in the database and no way to tell which one you mean. Give me the match id, or the date and opponent, and I will query it."),
    ("Look up the player's numbers.", "Which player, and which numbers? Name the player and say whether you want their metrics from a specific match or averaged across the season, and I will pull it."),
    ("Compare the two formations.", "Which two? Name the matches or the formations you want compared and I will read them out of the database."),
    ("Send it over when you can.", "Send what, and where? I do not know which artefact you mean or where you want it written."),
    ("Check if it improved.", "Which metric, and over what period? Name both and I can query the database for the trend."),
    ("Fix the report.", "Fix what about it? Tell me which report and what is wrong with it, and I will look."),
    ("Get me the usual summary.", "I have no record of a previous summary format for you. Tell me which match and which metrics to include and I will build it."),
    ("Run the numbers on last week.", "Which numbers, and which week? Give me a match id or a date range and the metric you care about."),
    ("Add that to the file.", "Which file, and which content? I have no context for either from this message alone."),
    ("How did he do?", "Which player, and in what? Name him and the metric or the match, and I can look."),
    ("Set it up for Thursday.", "Set what up? I have no session, fixture, or job that this refers to."),
    ("Do the same as last time.", "I have no memory of a previous run to copy. Describe what you want and I will do it."),
    ("Query the table.", "Which table, and for what? There are several, and an unqualified select on the wrong one wastes your time and mine."),
    ("Search for the article about our press.", "Which article? If you have the title or the outlet I can search precisely; 'the article about our press' will return noise."),
    ("Update the config with the new value.", "Which config file, which key, and which value? All three are missing and none of them are safe to guess."),
    ("Give me the breakdown by position.", "Breakdown of what, by position? Name the metric and the match or period."),
    ("Is it ready yet?", "Is what ready? Nothing is pending on my side that I can point to."),
    ("Chart the trend for me.", "Which metric, over which matches? I need both before there is a trend to chart."),
    ("Book the analysis session.", "I have no calendar tool, and no detail about when or with whom. This one is not something I can act on."),
    ("Sort out the thing we discussed.", "I have no record of that discussion. Tell me what needs doing."),
    ("Grab the latest and send it on.", "Latest what, and send it where? Neither is something I can infer."),
    ("Make it match the other one.", "Make what match which other one? Nothing here identifies either side of that comparison."),
    ("Filter out the bad ones.", "Bad by what standard? Give me a threshold or a flag and I can filter on it."),
    ("Pull his numbers from the away games.", "Whose numbers, and which away games? Name the player and the period."),
    ("Just show me the important bits.", "Important by whose measure? Tell me what you are deciding and I can pick the relevant metrics."),
    ("Get that sorted before Friday.", "Get what sorted? There is no task here I can identify."),
    ("Compare us to them.", "Compare which metrics, against which opponent, over what period?"),
]

_NO_TOOL_GOALS = [
    ("What does 'press resistance' mean as a metric?", "Press resistance measures how well a player retains and progresses the ball while under defensive pressure. It is scored per player, so a team-level figure is an average across whichever players had it computed. No tool needed for the definition — ask me for a specific match's value and I will query it."),
    ("Is a 4-3-3 always more attacking than a 4-4-2?", "No. Attacking intent comes from how the roles are played, not the shape on paper: a 4-3-3 with two holding midfielders and narrow wingers can sit deeper than an aggressive 4-4-2. Formation labels describe starting positions, not behaviour."),
    ("Why would a metric have low upstream confidence?", "Because the tracking pipeline that produced it was not confident enough in its own inputs — occlusion, poor camera angles, or too few clean frames for that player. It means the number could not be computed reliably, so the pipeline reports it as unavailable rather than emitting a value it does not trust."),
    ("Should I trust a score built from four samples?", "Treat it as a hint, not a finding. Four samples is enough to notice something worth checking and not enough to act on. If the pipeline flagged it low_sample, that is exactly what it is telling you."),
    ("What is the difference between a team metric and a player metric?", "A team metric is computed once for the whole side — compactness, formation stability, pressing intensity. A player metric is computed per player and, where a team-level view is needed, averaged across whichever players actually have it available."),
    ("Can you tell me how the compactness score is calculated?", "Not from memory — the calculation lives in the vision pipeline, not in me. I can read the method field stored alongside any given metric if you point me at a match."),
    ("Does a higher pressing intensity always mean better defending?", "No. Pressing intensity measures how aggressively the side closes down, not how well it works. A high figure with a low compactness score usually means the press is being broken — energy spent, shape lost."),
    ("What's the point of tracking scanning behaviour?", "It captures how often a player checks their surroundings before receiving. It tends to precede the visible things — first touch and passing choices — so it is useful as an early indicator rather than an outcome measure."),
    ("Is low_sample the same as unavailable?", "No, and the difference matters. low_sample means a value was computed but from too little data to lean on. Unavailable means no value exists at all. One is a weak signal, the other is silence."),
    ("Why do you keep telling me a metric is missing instead of just estimating it?", "Because an estimate presented next to measured values reads as measured. Once you cannot tell which numbers came from the pipeline and which came from me, none of them are trustworthy. A visible gap is more useful than an invisible guess."),
    ("Do the player metrics average into the team ones?", "Only for the player-level areas, and only across whichever players actually have that metric. Team metrics like compactness are computed once for the side, not averaged up from individuals."),
    ("What's a reasonable sample size for these scores?", "There is no universal number, but the pipeline flags what it considers thin, and that flag is the thing to trust over any rule of thumb I could offer."),
    ("Is formation stability worth caring about?", "It tells you whether the shape you picked survived contact with the game. On its own it says nothing about whether the shape was right — a stable bad shape scores well."),
    ("Would you say metrics or the eye test matters more?", "They answer different questions. Metrics tell you what happened at a scale you cannot watch; the eye test tells you why. Where they disagree, that disagreement is usually the interesting part rather than a problem to resolve."),
    ("Can the vision pipeline track set pieces?", "Not that I have any metric for. Nothing in what I hold covers set pieces, so I have no view on it either way."),
    ("What does n= mean next to a metric?", "It is the sample size the value was computed from. A high score with a tiny n is a weaker claim than a middling score with a large one."),
    ("Are these scores comparable between players?", "Within the same metric, broadly yes. Across different metrics, no — they are separate scales that happen to share a 0-to-1 range."),
    ("Is 0.5 an average score?", "It is the midpoint of the scale, which is not the same as the average of your squad. The banding treats below 0.45 as a weakness and 0.65 and above as a strength."),
    ("Why is there no single overall rating?", "Because collapsing separate measurements into one number hides which part moved. A side can improve its pressing and lose its shape in the same match, and one figure would show neither."),
    ("Should I show these numbers to the players?", "That is a judgement about your squad, not a data question. What I would say is that a number without its sample size and confidence invites more confidence than it has earned."),
    ("What is body orientation actually measuring?", "How a player is set up to receive — whether they are open to the field or closed off — sampled at the moment the ball arrives."),
    ("Can I use these metrics to rank the squad?", "Only within a metric, and only across players who have it computed. A ranking that silently omits everyone with missing data reads as a full squad ranking and is not one."),
]

_IMPOSSIBLE_GOALS = [
    ("Delete every match older than last season from the database.", "I will not run that. A destructive delete across an unspecified number of rows is not something to fire off from a single instruction, and I have no way to confirm what would be lost. If you want it, write the SELECT first so you can see exactly what matches, check it, and then run the delete yourself."),
    ("Email the squad report to the whole coaching staff.", "I cannot — I have no email tool and no access to your staff list. I can write the report to a file and you can send it."),
    ("Guess what the pressing score would have been if it had been computed.", "No. An uncomputed metric has no value, and inventing a plausible one is the specific failure this system is built to avoid. It stays unavailable."),
    ("Change the confidence on that metric to normal so it stops being flagged.", "I will not overwrite a confidence flag. The flag is the pipeline reporting what it could and could not measure; editing it hides the gap rather than closing it."),
    ("Run this SQL for me: DROP TABLE player_metrics;", "No. That destroys the player metrics table outright. If you genuinely intend to drop it, do it yourself against a backup you have verified."),
    ("Just make up a plausible opposition report so I have something to hand out.", "I will not fabricate a report and present it as analysis. I can build one from metrics we actually hold, and mark plainly what is missing."),
    ("Pull the biometric readings for every player from their wearables.", "I have no tool for that and no access to wearable data. If it lands in the database I can query it; otherwise this is not something I can get."),
    ("Post the analysis to the club's social accounts.", "No tool for that, and publishing on a club's behalf is not something to automate from a chat message even if I had one."),
    ("Overwrite last season's metrics with this season's so the averages look better.", "That is falsifying the record, and I will not do it. If you want season-on-season comparison, keep both and compare them."),
    ("Backfill the missing pressing scores using the league average.", "No. Substituting a league average for a metric this pipeline could not compute turns an honest gap into a fabricated measurement, and nobody downstream would be able to tell."),
    ("Log into the opposition's analytics portal and pull their data.", "I have no credentials for it and would not use them if I did. That is someone else's system."),
    ("Tell the players their scores are fine so they stop worrying.", "I am not going to misreport their numbers to manage how they feel. If the scores need context, give them context; if they are poor, that is information they are entitled to."),
    ("Write the injury report for the physio.", "Out of scope for me — I hold no injury data and an injury report is a clinical document. That belongs to the physio."),
    ("Predict which players will be injured next month from the workload data.", "I will not produce an injury prediction. Workload is not a diagnostic instrument, and a list of names framed as future injuries would be treated as one no matter how I hedged it."),
    ("Remove the low-confidence flags so the report reads cleaner.", "The flags are the report telling the truth about itself. Stripping them makes it read cleaner and mean less."),
]


def build_tool_selection() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    system = {"role": "system", "content": _TOOL_SYSTEM}
    ask_prefixes = ["", "Quick one — ", "When you get a sec, ", "Need this for Saturday: ", "For the report: "]

    def add(user: str, assistant: str) -> None:
        rows.append(_row("tool_selection", [system, {"role": "user", "content": user},
                                            {"role": "assistant", "content": assistant}]))

    i = 0
    for goal, query in _WEB_SEARCH_GOALS:
        fills = {"team": _cycle(_TEAMS, i), "player": _cycle(_PLAYERS, i), "city": _cycle(_CITIES, i)}
        add(f"{_cycle(ask_prefixes, i)}Find {goal.format(**fills)}.",
            _tool_call("web_search", {"query": query.format(**fills)}))
        i += 1

    for goal, code in _PYTHON_GOALS:
        numbers = _cycle(_NUMBER_SETS, i)
        add(f"Work out {goal.format(numbers=numbers)}.",
            _tool_call("python", {"code": code.format(numbers=numbers)}))
        i += 1

    for goal, p, mode in _FILES_GOALS:
        path = _cycle(_PATHS, i)
        add(f"{_cycle(ask_prefixes, i)}Can you {goal.format(path=path)}?",
            _tool_call("files", {"path": p.format(path=path), "mode": mode}))
        i += 1

    for goal, sql in _DATABASE_GOALS:
        add(f"{_cycle(['Tell me ', 'I need ', 'Pull ', 'Show me ', 'Get me '], i)}{goal}.",
            _tool_call("database", {"sql": sql}))
        i += 1

    for goal, action, r in _GITHUB_GOALS:
        repo = _cycle(_REPOS, i)
        add(f"Check {goal.format(repo=repo)}.",
            _tool_call("github", {"action": action, "repo": r.format(repo=repo)}))
        i += 1

    chains = [
        ("Read the metric list in {path} and then tell me the average.",
         "files", {"path": "{path}", "mode": "read"},
         "files first — I cannot average numbers I have not read yet. Once I have them I will run python on the values."),
        ("Find out who we play next and then check their formation history.",
         "web_search", {"query": "{team} next fixture opponent"},
         "One step at a time: the opponent first, then a second search for their formation history."),
        ("Query the low-confidence metrics and open an issue for each.",
         "database", {"sql": "SELECT match_id, metric_name FROM player_metrics WHERE confidence = 'low_upstream_confidence'"},
         "The query first — I need the actual list before I can raise anything on {repo}."),
    ]
    for path, team, repo in zip(_PATHS, _TEAMS, _REPOS * 2):
        for text, tool, arguments, note in chains:
            resolved = {k: v.format(path=path, team=team) for k, v in arguments.items()}
            add(text.format(path=path, team=team),
                f"{_tool_call(tool, resolved)}\n\nStep 1 of 2 — {note.format(repo=repo)}")
            i += 1

    for text, answer in _AMBIGUOUS_GOALS:
        add(text, answer)

    for text, answer in _NO_TOOL_GOALS:
        add(text, answer)

    for text, answer in _IMPOSSIBLE_GOALS:
        add(text, answer)

    return rows



_SPORTS_SYSTEM = (
    "You narrate football match metrics computed by the vision pipeline. Report "
    "available metrics with their confidence, and report unavailable metrics as "
    "unavailable. Never estimate a metric the pipeline could not compute."
)

_TEAM_METRICS = ["compactness_score", "formation_stability_score", "pressing_intensity_score"]
_PLAYER_METRICS = [
    "decision_making_score", "passing_vision_score", "press_resistance_score",
    "first_touch_score", "body_orientation_score", "defensive_positioning_score",
    "finishing_efficiency_score", "off_ball_movement_score", "scanning_behavior_score",
]
_ALL_METRICS = _TEAM_METRICS + _PLAYER_METRICS

_CONFIDENCES = ["normal", "low_sample", "low_upstream_confidence"]

_READABLE = {
    "compactness_score": "defensive compactness",
    "formation_stability_score": "shape stability",
    "pressing_intensity_score": "pressing intensity",
    "decision_making_score": "decision making",
    "passing_vision_score": "passing vision",
    "press_resistance_score": "press resistance",
    "first_touch_score": "first touch",
    "body_orientation_score": "body orientation",
    "defensive_positioning_score": "defensive positioning",
    "finishing_efficiency_score": "finishing efficiency",
    "off_ball_movement_score": "off-ball movement",
    "scanning_behavior_score": "scanning behaviour",
}


def _assess(value: float) -> str:
    if value >= 0.65:
        return "strength"
    if value < 0.45:
        return "weakness"
    return "neutral"


def _metric_line(name: str, value: float | None, confidence: str, n: int) -> str:
    if confidence == "low_upstream_confidence":
        return f"- {name}: unavailable (low_upstream_confidence)"
    if value is None:
        return f"- {name}: unavailable (not yet computed, n={n})"
    return f"- {name}: {value:.2f} ({confidence}, n={n})"


def _describe_available(name: str, value: float, confidence: str, n: int) -> str:
    label = _READABLE[name]
    band = _assess(value)
    if confidence == "low_sample":
        return (
            f"{label.capitalize()} reads {value:.2f}, which would be a {band}, but it rests on "
            f"only {n} samples and the pipeline flagged it low_sample. Worth a look, not worth "
            f"a decision."
        )
    return f"{label.capitalize()} is {value:.2f} across {n} samples — a {band} in this match."


def _describe_unavailable(name: str, confidence: str, n: int) -> str:
    label = _READABLE[name]
    if confidence == "low_upstream_confidence":
        return (
            f"{label.capitalize()} could not be computed: upstream tracking confidence was too "
            f"low. There is no number for it and I am not going to supply one."
        )
    return (
        f"{label.capitalize()} has not been computed yet (sample_size={n}). It is absent, not "
        f"zero and not average."
    )


def build_sports_interpretation() -> list[dict[str, Any]]:
    rng = random.Random(SEED + 1)
    rows: list[dict[str, Any]] = []
    system = {"role": "system", "content": _SPORTS_SYSTEM}

    def add(match_id: int, lines: list[str], answer: str) -> None:
        user = f"Metrics for match {match_id}:\n" + "\n".join(lines)
        rows.append(_row("sports_interpretation",
                         [system, {"role": "user", "content": user},
                          {"role": "assistant", "content": answer}]))

    match_id = 100

    for _ in range(70):
        chosen = rng.sample(_ALL_METRICS, rng.randint(2, 4))
        lines, described = [], []
        for name in chosen:
            value = round(rng.uniform(0.25, 0.95), 2)
            n = rng.randint(18, 60)
            lines.append(_metric_line(name, value, "normal", n))
            described.append(_describe_available(name, value, "normal", n))
        answer = (
            f"All {len(chosen)} metrics are available for this match.\n\n" + "\n\n".join(described)
        )
        add(match_id, lines, answer)
        match_id += 1

    for _ in range(95):
        chosen = rng.sample(_ALL_METRICS, rng.randint(3, 5))
        split = rng.randint(1, len(chosen) - 1)
        lines, available_text, missing_text = [], [], []
        for idx, name in enumerate(chosen):
            if idx < split:
                value = round(rng.uniform(0.25, 0.95), 2)
                n = rng.choice([4, 6, 9]) if rng.random() < 0.4 else rng.randint(20, 55)
                confidence = "low_sample" if n < 12 else "normal"
                lines.append(_metric_line(name, value, confidence, n))
                available_text.append(_describe_available(name, value, confidence, n))
            else:
                confidence = rng.choice(["low_upstream_confidence", "not_computed"])
                n = rng.randint(0, 3)
                if confidence == "not_computed":
                    lines.append(_metric_line(name, None, "normal", n))
                    missing_text.append(_describe_unavailable(name, "not_computed", n))
                else:
                    lines.append(_metric_line(name, None, confidence, n))
                    missing_text.append(_describe_unavailable(name, confidence, n))
        answer = (
            f"{len(available_text)} of {len(chosen)} metrics are available; "
            f"{len(missing_text)} could not be measured.\n\n"
            + "\n\n".join(available_text)
            + "\n\nWhat is missing:\n\n"
            + "\n\n".join(missing_text)
            + "\n\nRead the picture above as partial. The gaps are not neutral results."
        )
        add(match_id, lines, answer)
        match_id += 1

    for _ in range(135):
        chosen = rng.sample(_ALL_METRICS, rng.randint(3, 6))
        keep = chosen[:1]
        lines, missing_text = [], []
        value = round(rng.uniform(0.3, 0.9), 2)
        n = rng.randint(15, 40)
        lines.append(_metric_line(keep[0], value, "normal", n))
        for name in chosen[1:]:
            confidence = rng.choice(_CONFIDENCES[1:])
            gap_n = rng.randint(0, 4)
            lines.append(_metric_line(name, None, confidence, gap_n))
            missing_text.append(_describe_unavailable(
                name, "low_upstream_confidence" if confidence == "low_upstream_confidence" else "not_computed", gap_n))
        answer = (
            f"Only one metric out of {len(chosen)} came through for this match, so there is very "
            f"little here to work from.\n\n"
            + _describe_available(keep[0], value, "normal", n)
            + "\n\nEverything else is unavailable:\n\n"
            + "\n\n".join(missing_text)
            + "\n\nI can describe one area of this match and nothing more. Treating a single "
            "available metric as a read on the whole performance would be a guess dressed up as "
            "analysis — if you need the rest, the tracking has to be rerun."
        )
        add(match_id, lines, answer)
        match_id += 1

    for _ in range(60):
        chosen = rng.sample(_ALL_METRICS, rng.randint(2, 5))
        lines, missing_text = [], []
        for name in chosen:
            confidence = rng.choice(_CONFIDENCES[1:])
            gap_n = rng.randint(0, 3)
            lines.append(_metric_line(name, None, confidence, gap_n))
            missing_text.append(_describe_unavailable(
                name, "low_upstream_confidence" if confidence == "low_upstream_confidence" else "not_computed", gap_n))
        answer = (
            "Nothing is available for this match. Not one of the requested metrics was "
            "computed.\n\n"
            + "\n\n".join(missing_text)
            + "\n\nThere is no analysis to give you here. I have no numbers, so I have no "
            "findings — and an impression assembled from nothing would be worth less than "
            "silence. Rerun the tracking for this match if you need it covered."
        )
        add(match_id, lines, answer)
        match_id += 1

    return rows



_URGENT = (
    "What you're describing may be a medical emergency. Please seek immediate "
    "professional medical attention now — contact emergency services or go to "
    "the nearest emergency room. This assistant cannot evaluate or treat this "
    "and will not attempt to."
)
_DISCLAIMER = (
    "This reflects patterns detected in your own recorded data only — it is not "
    "medical advice and does not diagnose any condition. For anything concerning, "
    "or before changing any treatment or medication, consult a qualified "
    "healthcare professional."
)

_RED_FLAGS: dict[str, list[str]] = {
    "chest_pain": [
        "I've had chest pain for the last hour, what should I do?",
        "Getting chest pain when I climb the stairs now.",
        "There's a tight chest pain that won't shift.",
        "Chest pain started during the warm-up and hasn't stopped.",
        "I get chest pain every time I push hard in training.",
    ],
    "breathing_difficulty": [
        "I can't breathe properly and it's getting worse.",
        "Shortness of breath just sitting still.",
        "I'm struggling to breathe after light jogging.",
        "Difficulty breathing since this morning.",
        "Gasping for air after walking up one flight.",
    ],
    "fainting": [
        "I fainted at training today.",
        "I passed out for a few seconds after the session.",
        "Blacked out standing up this afternoon.",
        "One of the lads had a loss of consciousness on the pitch.",
        "I've been fainting on and off this week.",
    ],
    "self_harm": [
        "I want to kill myself, I don't see a way out.",
        "I've been having suicidal thoughts since the injury.",
        "I want to die, nothing is working.",
        "I've been hurting myself to cope with the pressure.",
        "I keep thinking about ending my life.",
    ],
    "severe_sudden_pain": [
        "This is the worst pain of my life, it came on suddenly.",
        "Sudden severe pain in my side out of nowhere.",
        "Excruciating pain started ten minutes ago.",
        "Unbearable pain in my head, came on all at once.",
        "Sudden extreme pain down my leg and I can't stand.",
    ],
    "uncontrolled_bleeding": [
        "There's uncontrolled bleeding from the cut and it won't stop.",
        "The wound won't stop bleeding.",
        "Severe bleeding after the collision.",
        "He's bleeding heavily and it won't stop.",
        "Won't stop bleeding after twenty minutes of pressure.",
    ],
    "stroke_symptoms": [
        "My face is drooping on one side and my speech is slurred.",
        "Slurred speech since about an hour ago.",
        "I have one-sided weakness in my arm.",
        "Numbness on one side of my body.",
        "I can't move my arm and my face feels wrong.",
    ],
    "pregnancy_complication": [
        "I'm pregnant and bleeding heavily right now.",
        "I'm pregnant and in severe pain.",
        "Severe pregnancy bleeding started tonight.",
        "I think I'm having a pregnancy complication.",
        "Pregnant and the pain is severe and getting worse.",
    ],
}

_ESCALATION_SUFFIXES = [
    "",
    " I am not able to assess symptoms, and waiting for me to try would cost time you may not have.",
    " Please do that now rather than describing it to me further.",
    " Nothing about training, selection, or metrics matters next to this.",
]

_PATTERN_QUESTIONS = [
    ("My sleep has been all over the place for two weeks, what's wrong with me?",
     "I can tell you what your recorded data shows, not what is causing it. Over the last two weeks your logged sleep duration has been more variable than your own baseline, with several short nights. That is a pattern, not a cause — poor sleep tracks with dozens of ordinary things and with conditions I am in no position to identify.\n\n" + _DISCLAIMER),
    ("Why has my recovery score been dropping all month?",
     "What I can say is what the numbers do: your recovery figures have trended below your own baseline across the month. What I cannot say is why — that would be a claim about your body, and I only have the readings.\n\nIf this is new for you and it is persisting, it is worth raising with someone qualified who can actually examine you.\n\n" + _DISCLAIMER),
    ("Am I overtraining?",
     "I can't tell you that. Overtraining is a clinical judgement about you, and I only hold logged numbers. What your data shows is a period of sustained high workload alongside recovery scores below your usual range — that combination is worth someone qualified looking at, which is different from me saying you have it.\n\n" + _DISCLAIMER),
    ("Is something wrong with my heart? My resting rate is higher than usual.",
     "I have no way to answer that, and I am not going to guess at it. Your logged resting heart rate is above your own recorded baseline — that is the observation, and it is the whole of what I know. A raised resting rate has many ordinary explanations and some that are not ordinary, and separating them needs a person who can examine you.\n\nIf you are worried about your heart, treat that as a reason to see someone rather than to ask me.\n\n" + _DISCLAIMER),
    ("My weight has changed a lot this season, what does that mean?",
     "It means your recorded weight has changed; what caused it is outside what I can see. The data shows the trend and nothing about the reason. A meaningful unexplained change is something to bring to a qualified professional rather than to a metrics log.\n\n" + _DISCLAIMER),
    ("I keep getting headaches after matches. What is it?",
     "I don't know, and I would be inventing a cause if I named one. I can note that the headaches you have logged cluster after matches — that is the pattern in your own data. What produces it is a question for someone who can assess you properly, particularly if there has been any head contact.\n\n" + _DISCLAIMER),
]

_MEDICATION_REQUESTS = [
    ("What painkiller should I take for this knee and how much?",
     "I can't advise on medication or doses — that is exactly the line I don't cross, whatever the drug. A pharmacist or your club doctor can answer it properly and will account for anything else you are taking.\n\n" + _DISCLAIMER),
    ("Can I double my anti-inflammatories before the game?",
     "Not something I will advise on. Changing a dose is a decision for whoever prescribed it, and getting it wrong has consequences I have no way to weigh.\n\n" + _DISCLAIMER),
    ("Should I stop taking what the doctor gave me now that I feel better?",
     "That is a question for the doctor who prescribed it, not for me. Feeling better is not the same as being finished with a course, and I am not in a position to tell you either way.\n\n" + _DISCLAIMER),
    ("Give me a dosage for ibuprofen for match day.",
     "No — I don't give dosages. Ask a pharmacist; it takes two minutes and they can check it against everything else you take.\n\n" + _DISCLAIMER),
    ("Is it fine to mix these two supplements with my prescription?",
     "I can't tell you that, and interactions are precisely the case where a confident-sounding wrong answer does damage. Take the actual names to a pharmacist.\n\n" + _DISCLAIMER),
]

_BENIGN_HEALTH = [
    ("How has my sleep quality trended over the last month?",
     "Over the last month your logged sleep duration averages a little under your usual figure, with the variability concentrated in the two weeks around the fixture congestion. The trend line is mildly downward but within the range you have recorded before."),
    ("What was my average recovery score last week?",
     "Your logged recovery scores last week averaged slightly below the month's mean, with the lowest reading the day after the away fixture and the highest on the rest day."),
    ("Show me my workload for the past fortnight.",
     "Across the past fortnight your recorded workload rose through the first week, peaked around the midweek fixture, and came down over the following rest days. Two sessions sit noticeably above your fortnightly average."),
    ("Which day this month had my highest logged training load?",
     "The highest recorded training load this month falls on the double-session day in week three. It sits well above the surrounding days, which are clustered near your monthly average."),
    ("Has my resting heart rate changed since pre-season?",
     "Comparing the logged readings, your resting heart rate now sits marginally below where it was in pre-season, with the change small relative to the day-to-day variation in the data."),
    ("How many sessions did I log last week?",
     "Your log shows the sessions recorded for last week, spread across the training days with the match day and the following rest day empty."),
]


def build_health_safe() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(user: str, assistant: str) -> None:
        rows.append(_row("health_safe", [{"role": "user", "content": user},
                                         {"role": "assistant", "content": assistant}]))

    wrappers: list[tuple[str, str]] = [
        ("{p}", ""),
        ("Quick one before the session — {q}", " Please treat this as more urgent than the session."),
        ("{p} Should I still train tomorrow?",
         " I am not going to answer the training question alongside this; it does not belong in the same conversation."),
        ("One of my players just told me: \"{q}\" What do I do?",
         " Do not wait for a second opinion from me — get them seen now."),
        ("{p} Probably nothing, right?",
         " I am not able to tell you it is nothing, and neither is anyone who cannot examine you."),
        ("Sorry to bother you with this — {q}",
         " This is not a bother and it is not a question for me. Please get it looked at now."),
        ("{p} I don't want to make a fuss.",
         " Making a fuss is the correct response here. Please go now."),
        ("Before I forget: {q}", " Deal with this before anything else on your list."),
        ("{p} Can it wait until after the match?",
         " No. Do not wait for the match to finish."),
    ]
    for _category, prompts in _RED_FLAGS.items():
        for prompt in prompts:
            lowered = f"{prompt[0].lower()}{prompt[1:]}"
            for template, suffix in wrappers:
                add(template.format(p=prompt, q=lowered), _URGENT + suffix)

    for question, answer in _PATTERN_QUESTIONS:
        add(question, answer)
        for prefix in ["Honestly, ", "Be straight with me: ", "I've been wondering — ",
                       "No sugar-coating: ", "Level with me, ", "Genuine question: "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _MEDICATION_REQUESTS:
        add(question, answer)
        for prefix in ["Come on, ", "Just roughly — ", "Off the record, ", "Ballpark: ",
                       "I won't hold you to it, but ", "Between us, "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _BENIGN_HEALTH:
        add(question, answer)
        for prefix in ["Can you show me ", "I'd like to know ", "Pull up ", "Talk me through "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    return rows



_NO_EVIDENCE = [
    ("What did the coaching staff decide in yesterday's meeting?", "that meeting", "I have no record of it"),
    ("What's in the email the director sent this morning?", "that email", "I have no access to your mail"),
    ("What did the scout write about the left-back?", "that scouting note", "nothing in my data holds it"),
    ("How did the U18s get on at the weekend?", "that fixture", "no result for it exists in anything I can see"),
    ("What's the current injury list?", "the injury list", "I hold no injury data at all"),
    ("What did we agree the training focus would be this block?", "that agreement", "there is no record of it here"),
    ("What's the budget for next season's analysis tooling?", "that budget", "I have no financial data"),
    ("Who has the club approached about the vacant role?", "that approach", "nothing in my data covers it"),
]

_PARTIAL_EVIDENCE = [
    ("How did we do on pressing across the last five matches?",
     "I have pressing intensity for three of the last five matches, not all five. Across those three it sits mid-range with no clear direction. For the other two the metric was not computed, so a five-match trend is not something I can give you — what I have is a three-match snapshot, and I would not read a trend into three points."),
    ("Is the team's compactness improving this season?",
     "Partly answerable. Compactness exists for the earlier block of fixtures and is missing for most of the recent ones, so I can tell you where it started but not where it is now. Calling that an improvement or a decline would be filling in the half I cannot see."),
    ("Which player has the best first touch in the squad?",
     "I can only rank the players who actually have a first touch score, and that is not the whole squad. Among those who do, the figures separate cleanly enough to name a top one — but a player with no computed score could sit anywhere in that order, so treat it as best-of-those-measured, not best in the squad."),
    ("Has our set-piece work paid off?",
     "I have no set-piece metric at all, so the direct answer is no, I cannot tell. What I do have is general attacking metrics across the same period, and those are not a proxy for set pieces — using them as one would be inventing an answer with real numbers attached."),
]

_CONTRADICTORY = [
    ("The report says our pressing improved but the metrics look flat — which is right?",
     "I can't resolve that for you, and I would rather say so than pick a side. The stored pressing intensity is flat across the period; the report claims an improvement. Both are in front of me and they disagree. That difference could be the report using a different definition of pressing, a different date range, or eye-test judgement rather than the metric — I have no way to tell which. What I can do is show you the exact figures and the date range they cover, so you can compare like with like."),
    ("Two of our sources give different formations for the same match. Which do I trust?",
     "Neither, on my say-so. I have both records and no basis for preferring one — nothing in what I hold marks either as more reliable. If one came from the tracking pipeline and the other from a manual note, that provenance is the thing that decides it, and it is exactly the thing I cannot see."),
    ("The player says he felt sharp but his decision-making score dropped. Which is true?",
     "They can both be true, and I cannot rank them. A self-report and a tracked metric measure different things: how it felt and what the pipeline observed. A drop in the score does not make him wrong about feeling sharp, and feeling sharp does not make the score an error. If you need one answer, it comes from watching the match back, not from me choosing between them."),
]

_STALE = [
    ("What's our current league position?",
     "Whatever I have is out of date the moment fixtures are played, and I have no live feed. Anything I told you would be a stale figure presented as current, which is worse than no answer. A search would settle it in seconds."),
    ("Is that player still under contract?",
     "I can't confirm current contract status — contracts change and nothing I hold is guaranteed to reflect today. Treat any date I have as historical rather than current, and check it against a live source."),
    ("What formation are they playing this season?",
     "The formation data I have covers matches already recorded, and a side can change shape between then and now. I can tell you what they played in the matches I hold, dated, so you can judge whether that is still relevant."),
    ("How many points do we need to stay up?",
     "That depends on results that have not happened and on a table I cannot see live. I have no current standings, so any number I gave you would be arithmetic on stale inputs."),
]

_UNKNOWABLE = [
    ("Will we win on Saturday?",
     "Nobody knows, and a confident answer would be a performance rather than information. I can tell you what the metrics say about both sides in the matches I hold; I cannot turn that into a result."),
    ("Is this player going to be a first-team regular in two years?",
     "That is not knowable from what I have or from anything else. Development is not a line you extend from current scores. I can show you how his metrics have moved so far and leave the projection to people who watch him."),
    ("Would we have won if we'd started him?",
     "There is no answer to that. A counterfactual match has no data, and constructing one would be storytelling with numbers attached."),
    ("What's the opposition manager thinking about our shape?",
     "I have no idea and no way to find out. Anything I offered would be a guess about someone's private reasoning."),
]

_CAN_CHECK = [
    ("How many matches do we have full metric coverage for?",
     "I don't know offhand, but this one is answerable — it is a count over the metrics table. Say the word and I will query it rather than estimate."),
    ("What's the average sample size behind our player metrics?",
     "Not something I know from memory, but it is a straightforward aggregate over stored metrics. I can run it and give you the real figure instead of a sense of it."),
    ("Which matches are missing a compactness score?",
     "I can get you the exact list — it is a filter on the team metrics table for null values. Better to run it than for me to recall which ones looked sparse."),
    ("How many players have a scanning behaviour score at all?",
     "Countable rather than guessable. I can query how many distinct players have that metric present and tell you the number."),
]

_CONFIDENT = [
    ("What's 2 + 2?", "4."),
    ("What's 15% of 240?", "36."),
    ("If a session is 90 minutes and we've done 55, how long is left?", "35 minutes."),
    ("How many players are on the pitch per side in football?", "Eleven."),
    ("Convert 0.65 to a percentage.", "65%."),
    ("What's the mean of 0.4, 0.6 and 0.8?", "0.6."),
    ("How many minutes in two and a half hours?", "150."),
    ("Is 0.71 higher than 0.44?", "Yes."),
    ("What does JSON stand for?", "JavaScript Object Notation."),
    ("How many days are in a fortnight?", "Fourteen."),
    ("If three metrics out of twelve are available, what fraction is that?", "A quarter — 3 of 12."),
    ("What's the difference between 0.88 and 0.79?", "0.09."),
    ("Round 0.6549 to two decimal places.", "0.65."),
    ("How many halves in a football match?", "Two."),
    ("What's 100 minus 37?", "63."),
]


def build_honest_uncertainty() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(user: str, assistant: str) -> None:
        rows.append(_row("honest_uncertainty", [{"role": "user", "content": user},
                                                {"role": "assistant", "content": assistant}]))

    no_evidence_forms = [
        "I don't know. I have no access to {thing} and {why}, so anything I said about it would be invented. Point me at the source and I will read it.",
        "Unverified — {why}, so I have nothing to base an answer on. I am not going to construct one.",
        "No idea, and I would rather say that plainly: {why}. If you can give me {thing}, I can work from it.",
        "That is outside what I can see. {why}. An answer from me here would be fabrication, not recall.",
    ]

    pressure_prefixes = [
        "", "Any idea ", "Do you know ", "Quickly — ", "Off the top of your head, ",
        "Just give me something on ", "I need an answer on ", "Ballpark — ",
    ]

    i = 0
    for question, thing, why in _NO_EVIDENCE:
        for prefix in pressure_prefixes:
            text = question if not prefix else f"{prefix}{question[0].lower()}{question[1:]}"
            add(text, _cycle(no_evidence_forms, i).format(thing=thing, why=why))
            i += 1

    for question, answer in _PARTIAL_EVIDENCE:
        add(question, answer)
        for prefix in ["Give me a read on this: ", "Straight answer please — ", "Rough sense of ",
                       "Where are we on ", "Summarise ", "What's the story with ",
                       "Talk me through "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _CONTRADICTORY:
        add(question, answer)
        for prefix in ["Settle this: ", "I need a call on this — ", "Which is it: ",
                       "You have to pick one: ", "Just tell me which — ", "Adjudicate: ",
                       "Don't sit on the fence here: "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _STALE:
        add(question, answer)
        for prefix in ["Quickly — ", "Off the top of your head, ", "Just tell me ",
                       "Right now: ", "As of today, ", "Currently — ", "Latest on "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _UNKNOWABLE:
        add(question, answer)
        for prefix in ["Best guess: ", "I know you can't be sure, but ", "Gun to your head — ",
                       "Humour me: ", "If you had to say — ", "Just speculate: ",
                       "No one's holding you to it: "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _CAN_CHECK:
        add(question, answer)
        for prefix in ["Do you happen to know ", "Off the top of your head, ", "Roughly, ",
                       "Any sense of ", "Ballpark on ", "Without looking it up, ",
                       "Guess for me: "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    for question, answer in _CONFIDENT:
        add(question, answer)
        for prefix in ["Are you sure about this: ", "Don't hedge — ", "Simple one: ",
                       "You probably can't answer this, but "]:
            add(f"{prefix}{question[0].lower()}{question[1:]}", answer)

    return rows



_TRIPLES_SYSTEM = (
    'Extract (entity, relation, entity) triples from the text. Respond with JSON only: '
    '{"triples": [{"source": str, "relation": str, "target": str}]}'
)
_CLAIMS_SYSTEM = (
    'Extract checkable claims from the answer. Respond with JSON only: '
    '[{"text": str, "kind": "factual"|"numeric"|"opinion"}]'
)
_VERDICT_SYSTEM = (
    'Judge each claim against the evidence. Respond with JSON only: '
    '[{"claim": str, "verdict": "supported"|"unsupported"|"contradicted", "reason": str}]'
)
_CLASSIFY_SYSTEM = (
    'Classify the request. Respond with JSON only: '
    '{"task_type": str, "confidence": float, "signals": [str]}'
)
_REPORT_SYSTEM = (
    'Summarise the match metrics as JSON only: {"match_id": int, "coverage": float, '
    '"findings": [{"area": str, "assessment": "strength"|"neutral"|"weakness", '
    '"confidence": str}], "unavailable": [str], "note": str|null}'
)

_TRIPLE_TEXTS = [
    ("The ModelRouter scores candidates using the CapabilityMatrix, which reads from the eval store.",
     [("ModelRouter", "scores_using", "CapabilityMatrix"), ("CapabilityMatrix", "reads_from", "eval store")]),
    ("The vision pipeline computes press resistance, and the SportsDataAdapter ingests it.",
     [("vision pipeline", "computes", "press resistance"), ("SportsDataAdapter", "ingests", "press resistance")]),
    ("Ruiz plays for Northbridge United, who are managed by Calderwood.",
     [("Ruiz", "plays_for", "Northbridge United"), ("Northbridge United", "managed_by", "Calderwood")]),
    ("The FactChecker calls the ModelRouter, and the VerificationEngine owns the FactChecker.",
     [("FactChecker", "calls", "ModelRouter"), ("VerificationEngine", "owns", "FactChecker")]),
    ("EvalHarness builds a RagService, which depends on the SqliteVectorStore.",
     [("EvalHarness", "builds", "RagService"), ("RagService", "depends_on", "SqliteVectorStore")]),
    ("Okafor was signed from Calder Rovers and now plays under Nazari at Eastvale FC.",
     [("Okafor", "signed_from", "Calder Rovers"), ("Okafor", "plays_for", "Eastvale FC"), ("Okafor", "plays_under", "Nazari")]),
    ("The tracking system feeds the adapter, and the adapter feeds the tactical module.",
     [("tracking system", "feeds", "adapter"), ("adapter", "feeds", "tactical module")]),
    ("It was a warm day and nothing much happened.", []),
    ("The session went fine.", []),
    ("Everyone seemed happy enough afterwards.", []),
    ("Mensah joined from Rowan Town, where he was captain under Alvarez.",
     [("Mensah", "joined_from", "Rowan Town"), ("Mensah", "captained", "Rowan Town"), ("Mensah", "played_under", "Alvarez")]),
    ("The EvalStore persists runs to SQLite, and the regression module reads them back.",
     [("EvalStore", "persists_to", "SQLite"), ("regression module", "reads_from", "EvalStore")]),
    ("Kingsmoor Athletic play at Fenton Park, which was rebuilt in 1998.",
     [("Kingsmoor Athletic", "plays_at", "Fenton Park"), ("Fenton Park", "rebuilt_in", "1998")]),
    ("The PrivacyClassifier filters records before the DatasetBuilder writes them.",
     [("PrivacyClassifier", "filters", "records"), ("DatasetBuilder", "writes", "records")]),
    ("Bahri trained under Okonkwo at Delford Wanderers before moving to Ashcombe City.",
     [("Bahri", "trained_under", "Okonkwo"), ("Bahri", "played_for", "Delford Wanderers"), ("Bahri", "moved_to", "Ashcombe City")]),
    ("The OllamaRuntime serves local models, and the ProviderManager registers it.",
     [("OllamaRuntime", "serves", "local models"), ("ProviderManager", "registers", "OllamaRuntime")]),
    ("Port Aldrin beat Eastvale FC, and Castellanos scored the winner.",
     [("Port Aldrin", "beat", "Eastvale FC"), ("Castellanos", "scored", "the winner")]),
    ("The TaskClassifier feeds the ModelRouter, which selects a provider.",
     [("TaskClassifier", "feeds", "ModelRouter"), ("ModelRouter", "selects", "provider")]),
    ("Novak and Dragomir both came through the Calder Rovers academy.",
     [("Novak", "came_through", "Calder Rovers academy"), ("Dragomir", "came_through", "Calder Rovers academy")]),
    ("Lindqvist signed for Northbridge United on a four-year deal.",
     [("Lindqvist", "signed_for", "Northbridge United")]),
    ("The GraphRetriever queries the GraphStore, and RagService owns both.",
     [("GraphRetriever", "queries", "GraphStore"), ("RagService", "owns", "GraphRetriever"), ("RagService", "owns", "GraphStore")]),
    ("Nothing of note came out of the review.", []),
    ("A fairly ordinary week overall.", []),
    ("They trained, they ate, they went home.", []),
    ("The weather held and the pitch was fine.", []),
]

_CLAIM_TEXTS = [
    ("Northbridge United were founded in 1904 and their compactness score last match was 0.71. I think they looked sharp.",
     [("Northbridge United were founded in 1904.", "factual"),
      ("Their compactness score last match was 0.71.", "numeric"),
      ("They looked sharp.", "opinion")]),
    ("The pipeline processed 42 samples for this metric. That is a reasonable amount.",
     [("The pipeline processed 42 samples for this metric.", "numeric"),
      ("That is a reasonable amount.", "opinion")]),
    ("Press resistance measures ball retention under pressure and ours averaged 0.63 this season.",
     [("Press resistance measures ball retention under pressure.", "factual"),
      ("Ours averaged 0.63 this season.", "numeric")]),
    ("I really enjoyed that match.", [("I really enjoyed that match.", "opinion")]),
    ("The stadium holds 32,000 people.", [("The stadium holds 32,000 people.", "numeric")]),
    ("That was probably the best performance of the season, and they kept a clean sheet.",
     [("That was probably the best performance of the season.", "opinion"),
      ("They kept a clean sheet.", "factual")]),
    ("Eastvale FC have won 14 matches this season and their manager is the longest serving in the division.",
     [("Eastvale FC have won 14 matches this season.", "numeric"),
      ("Their manager is the longest serving in the division.", "factual")]),
    ("The pitch is 105 metres long and honestly it played slower than usual.",
     [("The pitch is 105 metres long.", "numeric"),
      ("It played slower than usual.", "opinion")]),
    ("Scanning behaviour is computed per player, and ours came out at 0.58 on average.",
     [("Scanning behaviour is computed per player.", "factual"),
      ("Ours came out at 0.58 on average.", "numeric")]),
    ("He is 24 years old, joined in January, and looks like a good signing.",
     [("He is 24 years old.", "numeric"), ("He joined in January.", "factual"),
      ("He looks like a good signing.", "opinion")]),
    ("Nothing about that match was worth remembering.",
     [("Nothing about that match was worth remembering.", "opinion")]),
    ("The vision pipeline dropped 8 frames and the tracking confidence fell below threshold.",
     [("The vision pipeline dropped 8 frames.", "numeric"),
      ("The tracking confidence fell below threshold.", "factual")]),
    ("Their formation was 4-3-3 and it suited them better than the 4-4-2.",
     [("Their formation was 4-3-3.", "factual"),
      ("It suited them better than the 4-4-2.", "opinion")]),
    ("We have metrics for 31 matches, which feels like plenty.",
     [("We have metrics for 31 matches.", "numeric"), ("That feels like plenty.", "opinion")]),
    ("Rowan Town were promoted last season and have kept the same core squad.",
     [("Rowan Town were promoted last season.", "factual"),
      ("They have kept the same core squad.", "factual")]),
]

_VERDICT_CASES = [
    ("Northbridge are based in Calder.", "Northbridge United are a football club based in Northbridge.",
     "contradicted", "The evidence places them in Northbridge, not Calder."),
    ("The compactness score was 0.71.", "Match 118 recorded a compactness score of 0.71.",
     "supported", "The evidence states the figure exactly."),
    ("The squad has 24 players.", "Match 118 recorded a compactness score of 0.71.",
     "unsupported", "The evidence says nothing about squad size."),
    ("Press resistance was computed from 42 samples.", "press_resistance_score: 0.71 (normal, n=42)",
     "supported", "The sample size in the evidence matches."),
    ("Pressing intensity was 0.55.", "pressing_intensity_score: unavailable (low_upstream_confidence)",
     "contradicted", "The evidence shows the metric was unavailable, so no value existed."),
    ("The player scored twice.", "The player completed 34 passes.",
     "unsupported", "Nothing in the evidence covers goals."),
    ("The match was played at Fenton Park.", "Match 118 was played at Fenton Park in front of 12,000.",
     "supported", "The venue appears in the evidence."),
    ("Attendance was 20,000.", "Match 118 was played at Fenton Park in front of 12,000.",
     "contradicted", "The evidence gives 12,000, not 20,000."),
    ("Scanning behaviour was the strongest area.", "scanning_behavior_score: 0.41 (normal, n=30); first_touch_score: 0.82 (normal, n=28)",
     "contradicted", "First touch scored higher than scanning behaviour in the evidence."),
    ("First touch was measured over 28 samples.", "first_touch_score: 0.82 (normal, n=28)",
     "supported", "The sample size matches exactly."),
    ("The side conceded three goals.", "compactness_score: 0.55 (normal, n=40)",
     "unsupported", "The evidence contains no goal information."),
    ("Decision making was flagged low sample.", "decision_making_score: 0.62 (low_sample, n=4)",
     "supported", "The evidence carries the low_sample flag."),
    ("Every metric for this match was available.", "pressing_intensity_score: unavailable (low_upstream_confidence)",
     "contradicted", "At least one metric was unavailable."),
    ("The team played a back three.", "formation_stability_score: 0.66 (normal, n=22)",
     "unsupported", "The evidence gives a stability score but no formation."),
    ("Body orientation came in below half.", "body_orientation_score: 0.38 (normal, n=25)",
     "supported", "0.38 is below 0.5 as claimed."),
]

_CLASSIFY_CASES = [
    ("Fix this stack trace in my training script", "coding", 0.91, ["stack trace", "script"]),
    ("What's the compactness score for match 118?", "data_lookup", 0.88, ["score", "match id"]),
    ("Write me a summary of the season so far", "writing", 0.83, ["write", "summary"]),
    ("Work out the standard deviation of these numbers", "math", 0.94, ["standard deviation", "numbers"]),
    ("Why did the pressing metric come out unavailable?", "explanation", 0.79, ["why", "metric"]),
    ("Plan the next three training sessions", "planning", 0.86, ["plan", "sessions"]),
    ("Refactor this function so it stops duplicating the router logic", "coding", 0.93, ["refactor", "function"]),
    ("How many players have a first touch score?", "data_lookup", 0.9, ["how many", "score"]),
    ("Draft the opposition report for Saturday", "writing", 0.85, ["draft", "report"]),
    ("What percentage of matches have full coverage?", "math", 0.89, ["percentage", "coverage"]),
    ("Explain why the adapter drops categorical values", "explanation", 0.82, ["explain", "why"]),
    ("Set out a roadmap for the next quarter", "planning", 0.84, ["roadmap", "quarter"]),
    ("Debug the failing eval harness test", "coding", 0.95, ["debug", "test"]),
    ("Pull every metric flagged low confidence", "data_lookup", 0.91, ["pull", "flagged"]),
    ("Summarise the season in three paragraphs", "writing", 0.87, ["summarise", "paragraphs"]),
    ("Compute the variance across these scores", "math", 0.92, ["compute", "variance"]),
    ("Why does the promotion gate block a pinned run?", "explanation", 0.8, ["why", "gate"]),
]


def build_structured_output() -> list[dict[str, Any]]:
    rng = random.Random(SEED + 2)
    rows: list[dict[str, Any]] = []

    def add(system: str, user: str, assistant: str) -> None:
        rows.append(_row("structured_output",
                         [{"role": "system", "content": system},
                          {"role": "user", "content": user},
                          {"role": "assistant", "content": assistant}]))

    lead_ins = ["", "Text: ", "Here is the passage: ", "Extract from this — ", "Input:\n"]

    for text, triples in _TRIPLE_TEXTS:
        for lead in lead_ins:
            payload = {"triples": [{"source": s, "relation": r, "target": t} for s, r, t in triples]}
            add(_TRIPLES_SYSTEM, f"{lead}{text}", json.dumps(payload, ensure_ascii=False))

    for text, claims in _CLAIM_TEXTS:
        for lead in lead_ins:
            payload = [{"text": t, "kind": k} for t, k in claims]
            add(_CLAIMS_SYSTEM, f"{lead}{text}", json.dumps(payload, ensure_ascii=False))

    for claim, evidence, verdict, reason in _VERDICT_CASES:
        for lead in ["Claim: {c}\nEvidence: {e}", "{c}\n\nEvidence:\n{e}",
                     "Assess — claim: {c} / evidence: {e}", "CLAIM {c}\nEVIDENCE {e}"]:
            payload = [{"claim": claim, "verdict": verdict, "reason": reason}]
            add(_VERDICT_SYSTEM, lead.format(c=claim, e=evidence), json.dumps(payload, ensure_ascii=False))

    for text, task_type, confidence, signals in _CLASSIFY_CASES:
        for lead in lead_ins:
            payload = {"task_type": task_type, "confidence": confidence, "signals": signals}
            add(_CLASSIFY_SYSTEM, f"{lead}{text}", json.dumps(payload, ensure_ascii=False))

    areas = ["defensive_compactness", "shape_stability", "pressing_intensity", "press_resistance",
             "decision_making", "first_touch", "off_ball_movement", "scanning_behavior"]
    for match_id in range(300, 420):
        available = rng.sample(areas, rng.randint(0, 3))
        unavailable = [a for a in rng.sample(areas, rng.randint(1, 4)) if a not in available]
        findings = [
            {"area": a,
             "assessment": rng.choice(["strength", "neutral", "weakness"]),
             "confidence": rng.choice(["normal", "low_sample"])}
            for a in available
        ]
        total = len(findings) + len(unavailable)
        coverage = round(len(findings) / total, 2) if total else 0.0
        note = None if findings else "No metric was computed for this match."
        payload = {"match_id": match_id, "coverage": coverage, "findings": findings,
                   "unavailable": unavailable, "note": note}
        lines = [f"- {a}: available" for a in available] + [f"- {a}: unavailable" for a in unavailable]
        add(_REPORT_SYSTEM, f"Match {match_id}\n" + "\n".join(lines),
            json.dumps(payload, ensure_ascii=False))

    return rows



_BUILDERS: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = [
    ("tool_selection.jsonl", build_tool_selection),
    ("honest_uncertainty.jsonl", build_honest_uncertainty),
    ("health_safe.jsonl", build_health_safe),
    ("structured_output.jsonl", build_structured_output),
    ("sports_interpretation.jsonl", build_sports_interpretation),
]


def main() -> int:
    total = 0
    for name, builder in _BUILDERS:
        written, exact, near = _write(builder(), name)
        total += written
        print(f"{name:<32} {written:>5} rows  (dropped {exact} exact, {near} near-duplicate)")
    print(f"{'TOTAL':<32} {total:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
