"""Contract for Scanner.new_tracker() against the real ultralytics tracker
classes (not mocked) -- new_tracker() is a @staticmethod, so this doesn't
need the SigLIP/YOLO models loaded, just the actual BYTETracker/BOTSORT
constructors it calls.

This is the thing that broke silently across an ultralytics upgrade:
BYTETracker/BOTSORT dropped their `frame_rate` constructor argument in
ultralytics>=8.4, so passing it raised TypeError on every /live-recognize
connection right after accept() -- before any frame was read, so a client
saw it as "the socket drops on the first frame" rather than a clean
startup error. tests/test_live_recognition.py's Scanner mock doesn't call
the real new_tracker(), so it didn't (and can't) catch this class of bug.
"""
from scanner import Scanner


def test_new_tracker_constructs_without_error():
    tracker = Scanner.new_tracker()
    assert hasattr(tracker, 'update')


def test_new_tracker_returns_a_fresh_instance_each_call():
    a = Scanner.new_tracker()
    b = Scanner.new_tracker()
    assert a is not b
