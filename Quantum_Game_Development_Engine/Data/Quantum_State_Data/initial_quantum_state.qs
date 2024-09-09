{
    "quantum_states": [
        {
            "id": "q1",
            "type": "qubit",
            "state": {
                "amplitude_0": 0.6,
                "amplitude_1": 0.8
            },
            "position": [0, 0, 0],
            "properties": {
                "spin": "up",
                "charge": "neutral"
            }
        },
        {
            "id": "q2",
            "type": "qubit",
            "state": {
                "amplitude_0": 0.4,
                "amplitude_1": 0.9
            },
            "position": [1, 0, 0],
            "properties": {
                "spin": "down",
                "charge": "positive"
            }
        }
    ],
    "entanglements": [
        {
            "pair": ["q1", "q2"],
            "type": "entanglement",
            "strength": 0.7
        }
    ],
    "metadata": {
        "created_at": "2024-09-09T12:00:00Z",
        "description": "Initial quantum state setup for game simulation."
    }
}
