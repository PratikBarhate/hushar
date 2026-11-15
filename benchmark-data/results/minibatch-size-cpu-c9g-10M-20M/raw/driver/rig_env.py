#!/usr/bin/env python3
"""Prints `RIG` as the sweep wants it: `name=ip;name=ip` for every server in the set.

Its own file so a shell script can ask for it without an inline `python -c`, where the
quoting is a second chance to get it wrong.
"""
import hosts

print(hosts.rig_env())
