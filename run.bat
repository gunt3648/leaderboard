@echo off

SET "ROUTES=%LEADERBOARD_ROOT%\data\routes_1s.xml"
SET "REPETITIONS=1"
SET "DEBUG_CHALLENGE=2"
SET "TEAM_AGENT=%LEADERBOARD_ROOT%\leaderboard\autoagents\human_agent.py"
SET "CHECKPOINT_ENDPOINT=%LEADERBOARD_ROOT%\results.json"
SET "CHALLENGE_TRACK_CODENAME=SENSORS"

CALL ./scripts/run_evaluation.bat
pause