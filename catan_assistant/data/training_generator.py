"""
Training data generation for Catan Assistant.
Uses Catanatron simulator to generate training datasets.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
import os
from pathlib import Path

class CatanTrainingDataGenerator:
    """Generates training data for LLM fine-tuning using Catanatron."""
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_instruction_dataset(
        self, 
        num_games: int = 1000,
        players_per_game: int = 4
    ) -> List[Dict[str, str]]:
        """
        Generate instruction-tuning dataset from simulated games.
        
        Args:
            num_games: Number of games to simulate
            players_per_game: Number of players per game
            
        Returns:
            List of instruction-tuning examples
        """
        training_examples = []
        
        for game_id in range(num_games):
            print(f"Generating game {game_id + 1}/{num_games}")
            
            # Generate a mock game (replace with actual Catanatron simulation)
            game_states = self._simulate_mock_game(game_id, players_per_game)
            
            # Convert game states to training examples
            for state, action, rationale in game_states:
                example = self._create_training_example(state, action, rationale)
                training_examples.append(example)
        
        # Save dataset
        output_file = self.output_dir / "catan_instruction_dataset.jsonl"
        with open(output_file, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"✅ Generated {len(training_examples)} training examples")
        print(f"📁 Saved to {output_file}")
        
        return training_examples
    
    def _simulate_mock_game(
        self, 
        game_id: int, 
        num_players: int
    ) -> List[Tuple[Dict, str, str]]:
        """
        Simulate a mock Catan game (replace with actual Catanatron).
        
        Returns:
            List of (game_state, chosen_action, rationale) tuples
        """
        players = ["Red", "Blue", "White", "Orange"][:num_players]
        game_states = []
        
        for turn in range(1, 21):  # 20 turns per game
            for player in players:
                state = self._generate_mock_state(turn, player, players)
                action, rationale = self._generate_expert_decision(state)
                game_states.append((state, action, rationale))
        
        return game_states
    
    def _generate_mock_state(
        self, 
        turn: int, 
        current_player: str, 
        all_players: List[str]
    ) -> Dict[str, Any]:
        """Generate a realistic mock game state."""
        # Generate realistic resource counts based on turn
        base_resources = max(0, turn // 3)
        
        player_states = {}
        for player in all_players:
            is_current = player == current_player
            resources = {
                "wood": random.randint(0, base_resources + 2),
                "brick": random.randint(0, base_resources + 2),
                "wheat": random.randint(0, base_resources + 2),
                "ore": random.randint(0, base_resources + 1),
                "sheep": random.randint(0, base_resources + 2)
            }
            
            player_states[player] = {
                "resources": resources,
                "development_cards": {
                    "knight": random.randint(0, 2),
                    "victory_point": random.randint(0, 1),
                    "road_building": random.randint(0, 1)
                },
                "victory_points": min(10, random.randint(2, turn // 2 + 3)),
                "settlements": random.randint(0, min(5, turn // 3 + 2)),
                "cities": random.randint(0, min(4, turn // 5)),
                "roads": random.randint(4, min(15, turn + 4)),
                "largest_army": False,
                "longest_road": False
            }
        
        # Assign special achievements
        if random.random() < 0.1:
            random.choice(list(player_states.values()))["largest_army"] = True
        if random.random() < 0.1:
            random.choice(list(player_states.values()))["longest_road"] = True
        
        return {
            "game_info": {
                "turn": turn,
                "phase": "main_game",
                "dice_roll": [random.randint(1, 6), random.randint(1, 6)],
                "current_player": current_player,
                "players": all_players
            },
            "player_states": player_states,
            "legal_actions": self._generate_realistic_actions(player_states[current_player])
        }
    
    def _generate_realistic_actions(self, player_state: Dict) -> List[Dict]:
        """Generate realistic legal actions based on player resources."""
        actions = [{"type": "end_turn"}]
        resources = player_state["resources"]
        
        # Building actions based on resources
        if (resources["wood"] >= 1 and resources["brick"] >= 1 and 
            resources["wheat"] >= 1 and resources["sheep"] >= 1 and 
            player_state["settlements"] < 5):
            actions.append({
                "type": "build_settlement",
                "position": [random.randint(0, 5), random.randint(0, 5)],
                "cost": {"wood": 1, "brick": 1, "wheat": 1, "sheep": 1}
            })
        
        if (resources["wheat"] >= 2 and resources["ore"] >= 3 and 
            player_state["cities"] < 4):
            actions.append({
                "type": "build_city",
                "position": [random.randint(0, 5), random.randint(0, 5)],
                "cost": {"wheat": 2, "ore": 3}
            })
        
        if (resources["wood"] >= 1 and resources["brick"] >= 1 and 
            player_state["roads"] < 15):
            actions.append({
                "type": "build_road",
                "start": [random.randint(0, 5), random.randint(0, 5)],
                "end": [random.randint(0, 5), random.randint(0, 5)],
                "cost": {"wood": 1, "brick": 1}
            })
        
        if (resources["wheat"] >= 1 and resources["ore"] >= 1 and 
            resources["sheep"] >= 1):
            actions.append({
                "type": "buy_development_card",
                "cost": {"wheat": 1, "ore": 1, "sheep": 1}
            })
        
        # Trading actions
        for resource, count in resources.items():
            if count >= 4:
                other_resources = [r for r in resources.keys() if r != resource]
                if other_resources:
                    target_resource = random.choice(other_resources)
                    actions.append({
                        "type": "trade",
                        "offer": {resource: 4},
                        "request": {target_resource: 1},
                        "ratio": "4:1"
                    })
        
        return actions
    
    def _generate_expert_decision(
        self, 
        game_state: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Generate expert-level decision with rationale.
        
        Returns:
            (chosen_action, rationale) tuple
        """
        current_player = game_state["game_info"]["current_player"]
        player_state = game_state["player_states"][current_player]
        legal_actions = game_state["legal_actions"]
        
        # Strategy-based decision making
        vp = player_state["victory_points"]
        
        # Prioritize victory if close to winning
        if vp >= 8:
            building_actions = [a for a in legal_actions 
                             if a["type"] in ["build_settlement", "build_city"]]
            if building_actions:
                action = random.choice(building_actions)
                return (
                    f"{action['type']} at position {action.get('position', 'N/A')}",
                    f"Close to victory ({vp} VP), prioritizing building to secure win."
                )
        
        # Resource development strategy
        if vp < 5:
            road_actions = [a for a in legal_actions if a["type"] == "build_road"]
            settlement_actions = [a for a in legal_actions if a["type"] == "build_settlement"]
            
            if settlement_actions and random.random() < 0.7:
                action = random.choice(settlement_actions)
                return (
                    f"build_settlement at position {action.get('position', 'N/A')}",
                    "Early game focus on expansion and resource production."
                )
            elif road_actions:
                action = random.choice(road_actions)
                return (
                    f"build_road from {action.get('start', 'N/A')} to {action.get('end', 'N/A')}",
                    "Building road network for future expansion opportunities."
                )
        
        # Trade if resource imbalanced
        resources = player_state["resources"]
        total_resources = sum(resources.values())
        if total_resources > 7:
            trade_actions = [a for a in legal_actions if a["type"] == "trade"]
            if trade_actions:
                action = random.choice(trade_actions)
                return (
                    f"trade {action.get('offer', {})} for {action.get('request', {})}",
                    f"Have {total_resources} resources, need to balance hand and avoid robber."
                )
        
        # Default to ending turn
        return (
            "end_turn",
            "No optimal moves available, conserving resources for next turn."
        )
    
    def _create_training_example(
        self, 
        game_state: Dict[str, Any], 
        chosen_action: str, 
        rationale: str
    ) -> Dict[str, str]:
        """Create instruction-tuning example."""
        current_player = game_state["game_info"]["current_player"]
        player_state = game_state["player_states"][current_player]
        
        # Format game state for LLM
        state_description = f"""Player: {current_player}
Resources: {player_state['resources']}
Victory Points: {player_state['victory_points']}
Buildings: {player_state['settlements']} settlements, {player_state['cities']} cities
Turn: {game_state['game_info']['turn']}
Legal actions: {[action['type'] for action in game_state['legal_actions']]}"""
        
        legal_actions_str = "\n".join([
            f"- {action['type']}: {action.get('cost', 'No cost')}"
            for action in game_state['legal_actions']
        ])
        
        return {
            "instruction": "Suggest the best Catan move for the current player.",
            "input": f"{state_description}\n\nAvailable actions:\n{legal_actions_str}",
            "output": f"Best move: {chosen_action}\nReason: {rationale}"
        }

def generate_training_data():
    """Main function to generate training data."""
    generator = CatanTrainingDataGenerator("training_data")
    dataset = generator.generate_instruction_dataset(num_games=100)
    return dataset

if __name__ == "__main__":
    generate_training_data()