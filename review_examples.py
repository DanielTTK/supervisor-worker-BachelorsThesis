"""Fixed, hand-written supervisor demonstrations kept outside the benchmark."""

CUSTOM_REVIEW_VERSION = "few-shot-v1"
REVIEW_EXAMPLES = (
    {
        "label": "Supported claim and valid citation",
        "input": {
            "question": "Which revision is currently active for DEMO-UNIT-A?",
            "field": "revision",
            "task": "Copy the current approved revision.",
            "evidence": {
                "EX-A1": "Approved assignments determine current revisions.",
                "EX-A2": "APPROVED current assignment: DEMO-UNIT-A, DEMO-R11",
            },
            "context": {},
            "candidate": {"value": "DEMO-R11", "source": "EX-A2"},
        },
        "output": {"verdict": "accept", "value": None, "source": None},
    },
    {
        "label": "Unsupported claim without an approved replacement",
        "input": {
            "question": "What is the approved setting for DEMO-UNIT-B?",
            "field": "setting",
            "task": "Look up the approved setting for the active revision.",
            "evidence": {
                "EX-B1": "Only approved records can establish a setting.",
                "EX-B2": "APPROVED current assignment: DEMO-UNIT-B, DEMO-R12",
                "EX-B3": "PROVISIONAL setting: DEMO-UNIT-B, DEMO-R12, DEMO-S41",
            },
            "context": {"revision": {"value": "DEMO-R12", "source": "EX-B2"}},
            "candidate": {"value": "DEMO-S41", "source": "EX-B3"},
        },
        "output": {"verdict": "reject", "value": None, "source": None},
    },
    {
        "label": "Wrong claim with a supported replacement",
        "input": {
            "question": "What is the exception for DEMO-UNIT-C on its active revision and setting?",
            "field": "exception",
            "task": "Match the active revision and setting before selecting the exception.",
            "evidence": {
                "EX-C1": "APPROVED current assignment and setting: DEMO-UNIT-C, DEMO-R13, DEMO-S42",
                "EX-C2": "APPROVED exception: DEMO-UNIT-C, DEMO-R13, DEMO-S42, DEMO-ONLY-AT-RESET",
                "EX-C3": "APPROVED exception: DEMO-UNIT-C, DEMO-R13, DEMO-S43, DEMO-SERVICE-ONLY",
            },
            "context": {
                "revision": {"value": "DEMO-R13", "source": "EX-C1"},
                "setting": {"value": "DEMO-S42", "source": "EX-C1"},
            },
            "candidate": {"value": "DEMO-SERVICE-ONLY", "source": "EX-C3"},
        },
        "output": {"verdict": "correct", "value": "DEMO-ONLY-AT-RESET", "source": "EX-C2"},
    },
    {
        "label": "Contradictory evidence with no authority tie-break",
        "input": {
            "question": "Which revision is currently active for DEMO-UNIT-D?",
            "field": "revision",
            "task": "Use the approved current assignment if the records establish one.",
            "evidence": {
                "EX-D1": "These records have equal authority and no precedence rule.",
                "EX-D2": "APPROVED current assignment: DEMO-UNIT-D, DEMO-R14",
                "EX-D3": "APPROVED current assignment: DEMO-UNIT-D, DEMO-R15",
            },
            "context": {},
            "candidate": {"value": "DEMO-R14", "source": "EX-D2"},
        },
        "output": {"verdict": "uncertain", "value": None, "source": None},
    },
)
