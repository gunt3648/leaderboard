@echo off

python "%LEADERBOARD_ROOT%\leaderboard\leaderboard_evaluator.py" "--routes=%ROUTES%" "--repetitions=%REPETITIONS%" "--track=%CHALLENGE_TRACK_CODENAME%" "--checkpoint=%CHECKPOINT_ENDPOINT%" "--agent=%TEAM_AGENT%" "--agent-config=%TEAM_CONFIG%" "--debug=%DEBUG_CHALLENGE%" "--record=%RECORD_PATH%" "--resume=%RESUME%" "--objects-detected=%OBJECT_DETECTED%" "--audio-folder=%AUDIO_FOLDER%" "--assistant=%ASSISTANT%"
