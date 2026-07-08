"""
CLI entry-point for the mille-feuille Streamlit dashboard.

After ``pip install millefeuille[dashboard]`` the command
``mf-dashboard`` becomes available system-wide and can be
launched on a remote server with SSH port-forwarding.
"""

import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard.

    All extra CLI arguments are forwarded to ``streamlit run``, e.g.::

        mf-dashboard --server.port 8502

    When running on a headless server (typical for SSH-forwarded sessions)
    ``--server.headless true`` is added automatically.
    """
    from streamlit.web.cli import main as _st_main

    app_path = str(Path(__file__).with_name("app.py"))

    # Build the argv that streamlit expects
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "true",
        *sys.argv[1:],
    ]
    _st_main()
