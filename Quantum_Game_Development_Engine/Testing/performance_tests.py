# performance_tests.py
# Quantum_Game_Development_Engine/Testing/performance_tests.py

import unittest
import time
from Quantum_Game_Development_Engine.Game_Mechanics.player_controller import Player
from Quantum_Game_Development_Engine.Quantum_Mechanics.superposition_simulator import SuperpositionSimulator
from Quantum_Game_Development_Engine.Quantum_Logic.quantum_probability_trigger import QuantumProbabilityTrigger
from Quantum_Game_Development_Engine.Quantum_AI.entangled_npc_ai import EntangledNPC

class TestPerformance(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.player = Player(id=1, name="TestPlayer", position=(0, 0), health=100)
        self.simulator = SuperpositionSimulator()
        self.probability_trigger = QuantumProbabilityTrigger()
        self.entangled_npc = EntangledNPC(name="TestNPC", position=(5, 5))

    def test_player_move_performance(self):
        """
        Test the performance of the player move function.
        """
        start_time = time.time()
        for _ in range(1000):  # Perform 1000 moves
            self.player.move((10, 10))
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertLess(elapsed_time, 1, "Player move function is too slow")

    def test_superposition_simulator_performance(self):
        """
        Test the performance of the superposition simulator.
        """
        start_time = time.time()
        for _ in range(1000):  # Perform 1000 simulations
            self.simulator.simulate_superposition()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertLess(elapsed_time, 2, "Superposition simulator is too slow")

    def test_quantum_probability_trigger_performance(self):
        """
        Test the performance of the quantum probability trigger.
        """
        start_time = time.time()
        for _ in range(1000):  # Perform 1000 triggers
            self.probability_trigger.trigger_probability()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertLess(elapsed_time, 1.5, "Quantum probability trigger is too slow")

    def test_entangled_npc_ai_performance(self):
        """
        Test the performance of the entangled NPC AI decision-making.
        """
        start_time = time.time()
        for _ in range(1000):  # Perform 1000 decisions
            self.entangled_npc.make_decision()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertLess(elapsed_time, 2, "Entangled NPC AI decision-making is too slow")

if __name__ == "__main__":
    unittest.main()
