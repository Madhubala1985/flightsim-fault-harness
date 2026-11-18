"""LLM triage agent: alarms + sensor feature summary in, structured fault report out.

The agent never sees truth or the fault log. It sees exactly two things:
the alarm list for a flagged window (``fsfh.detectors``) and a feature summary
of the sensor channels in that window (``fsfh.triage.features``).
"""
