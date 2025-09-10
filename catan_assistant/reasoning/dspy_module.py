"""
DSPy module for Catan game decision making.
Provides declarative approach to prompt optimization for strategic decisions.
"""

import dspy
from typing import List, Dict, Any, Tuple
from ..core.game_state import GameState, GameAction

class CatanDecisionSignature(dspy.Signature):
    """
    Signature for Catan game decision making.
    Given current game state and legal actions, predict the best move with rationale.
    """
    game_state: str = dspy.InputField(desc="Current game state in JSON format")
    legal_actions: str = dspy.InputField(desc="List of legal actions available")
    player_name: str = dspy.InputField(desc="Current player making the decision")
    
    best_action: str = dspy.OutputField(desc="Best action to take from legal actions")
    rationale: str = dspy.OutputField(desc="Strategic reasoning for the chosen action")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1) for the decision")

class CatanStrategicAnalysis(dspy.Signature):
    """
    Signature for analyzing Catan board position and strategic opportunities.
    """
    game_state: str = dspy.InputField(desc="Current game state")
    player_perspective: str = dspy.InputField(desc="Player to analyze from")
    
    position_analysis: str = dspy.OutputField(desc="Analysis of current position strength")
    strategic_priorities: str = dspy.OutputField(desc="Key strategic priorities")
    opponent_threats: str = dspy.OutputField(desc="Analysis of opponent positions and threats")

class CatanDecisionModule(dspy.Module):
    """
    Main DSPy module for Catan decision making.
    Combines strategic analysis with action selection.
    """
    
    def __init__(self):
        super().__init__()
        self.analyze_position = dspy.Predict(CatanStrategicAnalysis)
        self.make_decision = dspy.Predict(CatanDecisionSignature)
    
    def forward(self, game_state: GameState) -> Dict[str, Any]:
        """
        Main forward pass for decision making.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary containing best action, rationale, and analysis
        """
        # Convert game state to string format for LLM
        state_str = self._format_game_state(game_state)
        legal_actions_str = self._format_legal_actions(game_state.legal_actions)
        
        # First, analyze the strategic position
        analysis = self.analyze_position(
            game_state=state_str,
            player_perspective=game_state.current_player
        )
        
        # Then make the tactical decision
        decision = self.make_decision(
            game_state=state_str,
            legal_actions=legal_actions_str,
            player_name=game_state.current_player
        )
        
        return {
            "best_action": decision.best_action,
            "rationale": decision.rationale,
            "confidence": decision.confidence,
            "position_analysis": analysis.position_analysis,
            "strategic_priorities": analysis.strategic_priorities,
            "opponent_threats": analysis.opponent_threats
        }
    
    def _format_game_state(self, game_state: GameState) -> str:
        """Format game state for LLM consumption."""
        current_player = game_state.get_current_player_state()
        
        # Create a concise, readable representation
        formatted = f"""
Player: {game_state.current_player}
Turn: {game_state.turn}
Victory Points: {current_player.victory_points}
Resources: {current_player.resources}
Buildings: {current_player.settlements} settlements, {current_player.cities} cities, {current_player.roads} roads
Development Cards: {current_player.development_cards}

Other Players:
"""
        for name, state in game_state.player_states.items():
            if name != game_state.current_player:
                formatted += f"  {name}: {state.victory_points} VP, {state.settlements+state.cities} buildings\n"
        
        if game_state.dice_roll:
            formatted += f"\nLast Dice Roll: {game_state.dice_roll[0]} + {game_state.dice_roll[1]} = {sum(game_state.dice_roll)}"
        
        return formatted.strip()
    
    def _format_legal_actions(self, actions: List[GameAction]) -> str:
        """Format legal actions for LLM consumption."""
        formatted_actions = []
        for i, action in enumerate(actions):
            action_str = f"{i+1}. {action.type.value.replace('_', ' ').title()}"
            
            if action.position:
                action_str += f" at position {action.position}"
            
            if action.cost:
                cost_str = ", ".join([f"{amt} {res}" for res, amt in action.cost.items()])
                action_str += f" (Cost: {cost_str})"
            
            formatted_actions.append(action_str)
        
        return "\n".join(formatted_actions)

class CatanMemoryModule(dspy.Module):
    """
    Module for incorporating game memory and historical patterns.
    """
    
    def __init__(self, memory_retriever=None):
        super().__init__()
        self.memory_retriever = memory_retriever
        self.contextualize = dspy.Predict("context, current_state -> enhanced_state")
    
    def forward(self, game_state: GameState, similarity_threshold: float = 0.7):
        """
        Enhance current game state with relevant historical context.
        
        Args:
            game_state: Current game state
            similarity_threshold: Minimum similarity for memory retrieval
            
        Returns:
            Enhanced state with historical context
        """
        if not self.memory_retriever:
            return game_state
        
        # Retrieve similar game states from memory
        similar_states = self.memory_retriever.retrieve(
            query_state=game_state,
            threshold=similarity_threshold,
            top_k=3
        )
        
        # Contextualize with historical patterns
        context = self._format_similar_states(similar_states)
        current_state = self._format_game_state(game_state)
        
        enhanced = self.contextualize(
            context=context,
            current_state=current_state
        )
        
        return enhanced
    
    def _format_similar_states(self, similar_states: List[Dict]) -> str:
        """Format similar historical states for context."""
        if not similar_states:
            return "No similar historical patterns found."
        
        formatted = "Similar historical patterns:\n"
        for i, state_info in enumerate(similar_states, 1):
            formatted += f"{i}. {state_info.get('description', 'Unknown pattern')} "
            formatted += f"(Similarity: {state_info.get('similarity', 0):.2f})\n"
        
        return formatted