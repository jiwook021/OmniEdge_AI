# OmniEdge_AI

OmniEdge_AI is a reference implementation for a local NVIDIA GPU AI runtime.
This checkout currently contains the buildable reference surfaces for:

- Four static frontend mode directories: `conversation`, `security`, `beauty`, and `game`.
- Game-mode browser logic for MP4 chart generation and finger-pinch scoring.
- A C++ hand-tracking contract node with an explicit unavailable-backend path.
- Shared CMake helpers used by the reference build.

The full daemon, Docker profile runtime, production model services, and hardware
capture stack are not included in this reference tree.

## Current Commands

Build and run the reference tests:

```bash
bash scripts/integration/install.sh --test
```

Run only the lightweight game/reference test:

```bash
bash test.sh game
```

Show the reference surface for a mode:

```bash
bash scripts/run_mode.sh conversation
bash scripts/run_mode.sh security
bash scripts/run_mode.sh beauty
bash scripts/run_mode.sh game
```

## Reference Modes

| Mode | Frontend | Runtime status |
|:---|:---|:---|
| `conversation` | `frontend/conversation` | Static frontend reference only |
| `security` | `frontend/security` | Static frontend reference only |
| `beauty` | `frontend/beauty` | Static frontend reference only |
| `game` | `frontend/game` | Browser-local chart/scoring logic plus hand-tracking contract tests |

Mode contracts used by tests live in `tests/e2e/mode_contracts.py`.

## Game Mode

Game mode runs in the browser from `frontend/game/index.html`. It decodes a
local MP4, generates notes from beat/frequency features, renders the playfield,
and scores pinch gestures. The native C++ hand-tracking node currently exposes
the expected module contract and returns a structured unavailable-backend status
until a real MediaPipe or ONNX backend is wired in.

Relevant tests:

```bash
node --test tests/frontend/test_game_logic.js
ctest --test-dir .codex-build --output-on-failure
```

## Build Notes

The CMake project is intentionally tolerant of missing optional source trees.
Only directories with a `CMakeLists.txt` are added to the build. The current
reference build registers:

- `modules/core/hand_tracking`
- `modules/nodes/hand_tracking`
- `tests/core/hand_tracking`
- `tests/nodes/hand_tracking`

Generated caches such as `build/`, `.codex-build/`, `__pycache__/`, and
`*.egg-info/` are not source.
