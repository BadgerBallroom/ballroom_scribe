"""Desktop GUI for viewing, generating, and editing ballroom competition event timestamps.

This script supervises the editor. It runs the editor itself in a child process
(`python -m ballroom_scribe.gui.run_editor`), so that it is still there to offer a report if the editor crashes. See
`ballroom_scribe.gui.supervise_editor`.
"""

import ballroom_scribe.gui

if __name__ == "__main__":
    raise SystemExit(ballroom_scribe.gui.supervise_editor())
