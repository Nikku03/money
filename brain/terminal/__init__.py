"""MONEY TERMINAL: a keyboard-driven market terminal over the memory bank, served locally.

``commands`` parses a command line, ``data`` turns a command into JSON read from the
bank, ``server`` serves it on loopback (``jobs`` runs collection in the background),
and ``analyst`` answers questions through the same read-only commands. The contract
between them is docs/TERMINAL.md.
"""
