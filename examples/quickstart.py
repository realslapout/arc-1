"""Smallest useful example: route a support ticket, rate its urgency and check the customer's mood.

    pip install git+https://github.com/realslapout/arc-1
    python examples/quickstart.py
"""
from arc1 import ARC1Predictor

model = ARC1Predictor("realslapout/ARC-1", cuda_graphs=True)

state = {"ticket": "I was charged twice for the same order and I want my money back."}
questions = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle `ticket`?",
        "criteria": {
            "billing": "payments, charges and refunds",
            "shipping": "delivery problems",
            "tech": "app or login problems",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is `ticket`?",
        "criteria": ["not urgent", "normal", "urgent"],
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
}

answers = model.predict(state, questions)["answers"]
print("team:   ", answers["team"]["choice"], answers["team"]["probabilities"])
print("urgency:", round(answers["urgency"]["score"], 2), "(0 = not urgent, 2 = urgent)")
print("angry:  ", round(answers["angry"]["noul"], 3), "(probability of yes)")
