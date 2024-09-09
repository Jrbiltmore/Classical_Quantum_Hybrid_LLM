# saved_quantum_states.qs
{
    "saved_states": [
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
            },
            "timestamp": "2024-09-09T15:00:00Z"
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
            },
            "timestamp": "2024-09-09T15:00:00Z"
        }
    ],
    "entanglements": [
        {
            "pair": ["q1", "q2"],
            "type": "entanglement",
            "strength": 0.7,
            "timestamp": "2024-09-09T15:00:00Z"
        }
    ],
    "metadata": {
        "saved_at": "2024-09-09T15:00:00Z",
        "description": "Saved quantum states at the end of the game session."
    }
}
