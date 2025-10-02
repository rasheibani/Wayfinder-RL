#!/usr/bin/env python3
"""
Realworld Environment PPO Evaluation Script

This script loads shapefiles from the realworld folder and runs PPO evaluation
similar to main_PPO.py, but specifically for realworld environments.
Runs 100 episodes and saves results to CSV for rigor of results.

Usage: python evaluate_realworld_ppo.py
"""

import random
import os
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re
from glob import glob
import warnings
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from tqdm import tqdm

# Try to import shapefile libraries for future use
try:
    import geopandas as gpd
    import fiona
    SHAPEFILE_SUPPORT = True
except ImportError:
    SHAPEFILE_SUPPORT = False
    print("Warning: geopandas/fiona not available. Shapefile processing will be limited.")

# Import existing functions from the main codebase
from z8file_to_dictionaries import z8file_to_dictionaries

# Define grammars and their corresponding directions (same as main_PPO)
GRAMMAR_DIRECTIONS = {
    4: ['north', 'south', 'east', 'west'],
    6: ['north', 'south', 'east', 'west', 'northeast', 'southwest'],
    8: ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
}

# Maximum distance for normalization
MAX_DISTANCE = 2000.0

def extract_coordinates(game_state):
    """Extract coordinates from game state feedback"""
    pattern = r"X:\s*([\d.]+)\s*\nY:\s*([\d.]+)"
    matches = re.search(pattern, game_state)
    if matches:
        x = float(matches.group(1))
        y = float(matches.group(2))
        return np.array([x, y])
    else:
        return np.array([0, 0])

def text_to_action(text, directions):
    """Converts a text instruction to an action index"""
    mapping = {direction: idx for idx, direction in enumerate(directions)}
    words = text.strip().lower().split()
    direction_word = None
    for word in words:
        if word in mapping:
            direction_word = word
            break
    
    if direction_word:
        return mapping[direction_word]
    else:
        return -1

def sentence_from_action(action, directions):
    """Converts an action index to a sentence"""
    if 0 <= action < len(directions):
        return f"go {directions[action]}"
    else:
        return "look"

def normalize(observation):
    """Normalize the observation to be in the correct range and shape"""
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    admissible_actions = observation[:max_directions]
    route_instructions = observation[max_directions:-2]
    instruction_indices = observation[-2:]

    # Normalize admissible actions (already in [0, 1])
    normalized_admissible_actions = admissible_actions

    # Normalize route instructions
    normalized_route_instructions = np.where(
        route_instructions != max_directions,
        route_instructions / (max_directions - 1),
        -1
    )

    # Normalize instruction indices
    normalized_instruction_indices = instruction_indices / max_directions

    return np.concatenate((
        normalized_admissible_actions,
        normalized_route_instructions,
        normalized_instruction_indices
    ))

class TextWorldEnv(gym.Env):
    """TextWorld Environment class - same as main_PPO"""
    def __init__(self, game_dict, room_positions, x_destination=None, y_destination=None, n_instructions=1, grammar=8,
                 reward_type='sparse', is_incomplete=False, route_instructions=None):
        super(TextWorldEnv, self).__init__()
        self.game_dict = game_dict
        self.room_positions = room_positions
        self.current_room_id = None
        self.n_instructions = n_instructions
        self.grammar = grammar
        self.reward_type = reward_type
        self.instruction_index = 0
        self.route_instructions = [] if route_instructions is None else route_instructions
        self.visited_states_actions = set()
        self.last_feedback_embedding = None
        self.counter = 0
        self.is_incomplete = is_incomplete
        self.x_destination = x_destination
        self.y_destination = y_destination
        self.x_origin = None
        self.y_origin = None
        self.exploration_threshold = 0

        # Define actions based on grammar
        if self.grammar in GRAMMAR_DIRECTIONS:
            self.directions = GRAMMAR_DIRECTIONS[self.grammar]
        else:
            raise ValueError("Invalid grammar. Choose from 4, 6, or 8.")

        self.max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
        self.action_space = spaces.Discrete(self.max_directions + 1)  # +1 for "look"
        self.observation_space = spaces.Box(low=0, high=8, shape=(self.max_directions + 15 + 2,), dtype=np.int32)

    def text_to_action_func(self, text):
        return text_to_action(text, self.directions)

    def sentence_from_action_func(self, action):
        return sentence_from_action(action, self.directions)

    def reset(self, **kwargs):
        self.counter = 0
        self.current_room_id = random.choice(list(self.game_dict.keys()))
        self.x, self.y = self.room_positions[self.current_room_id]
        self.x_origin, self.y_origin = self.x, self.y
        self.visited_states_actions.clear()
        self.instruction_index = 0

        if self.is_incomplete:
            if self.x_destination is None or self.y_destination is None:
                raise ValueError("x_destination and y_destination must be provided for incomplete instructions")
        else:
            self.route_instructions, self.x_destination, self.y_destination = self.generate_route_instructions()

        self.dist_from_origin_to_destination = np.sqrt(
            (self.x_destination - self.x_origin) ** 2 + (self.y_destination - self.y_origin) ** 2
        )

        admissible_actions = self.get_admissible_actions()
        observation = np.concatenate((
            admissible_actions_to_observation(admissible_actions, self.directions),
            self.pad_instructions(),
            np.array([self.instruction_index, self.route_instructions[self.instruction_index]]),
        ))
        
        assert observation.shape[0] == self.observation_space.shape[0], \
            f"Observation shape mismatch. Expected {self.observation_space.shape[0]}, got {observation.shape[0]}"

        observation = normalize(observation)
        return observation, {}

    def get_admissible_actions(self):
        return list(self.game_dict[self.current_room_id].keys())

    def generate_route_instructions(self):
        # Simple route generation - choose a random destination room
        dest_room_id = random.choice(list(self.room_positions.keys()))
        while dest_room_id == self.current_room_id:
            dest_room_id = random.choice(list(self.room_positions.keys()))
        
        dest_x, dest_y = self.room_positions[dest_room_id]
        
        # Generate simple instructions (this is a simplified version)
        instructions = [random.randint(0, len(self.directions)-1) for _ in range(self.n_instructions)]
        
        # # Print the generated instruction sequence
        # instruction_names = [self.directions[idx] for idx in instructions]
        # print(f"Generated route instructions ({len(instructions)} steps): {instruction_names}")
        # print(f"Instruction indices: {instructions}")
        
        return instructions, dest_x, dest_y

    def pad_instructions(self):
        # Pad instructions to fixed length (15)
        padded = self.route_instructions[:15]
        while len(padded) < 15:
            padded.append(self.max_directions)  # Use max_directions as padding
        return np.array(padded)

    def step(self, action):
        done = False
        truncated = False
        
        if action >= len(self.directions):  # "look" action
            reward = -1 if self.reward_type == 'step_cost' else 0
        else:
            # Execute action
            action_name = self.directions[action]
            if action_name in self.game_dict[self.current_room_id]:
                self.current_room_id = self.game_dict[self.current_room_id][action_name]
                self.x, self.y = self.room_positions[self.current_room_id]
                self.counter += 1
                self.instruction_index = min(self.instruction_index + 1, len(self.route_instructions) - 1)
                
                # Check if reached destination
                if abs(self.x - self.x_destination) < 1e-6 and abs(self.y - self.y_destination) < 1e-6:
                    reward = 25
                    done = True
                else:
                    reward = -1 if self.reward_type == 'step_cost' else 0
            else:
                reward = -1 if self.reward_type == 'step_cost' else 0
        
        # Episode truncation
        if self.counter >= 100:
            truncated = True
        
        # Create observation
        admissible_actions = self.get_admissible_actions()
        observation = np.concatenate((
            admissible_actions_to_observation(admissible_actions, self.directions),
            self.pad_instructions(),
            np.array([self.instruction_index, self.route_instructions[self.instruction_index]]),
        ))
        
        observation = normalize(observation)
        return observation, reward, done, truncated, {}

def admissible_actions_to_observation(admissible_actions, directions):
    """Convert admissible actions to binary observation"""
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    obs = np.zeros(max_directions)
    for action in admissible_actions:
        if action in directions:
            obs[directions.index(action)] = 1
    return obs

def get_admissible_actions_from_observation(observation, directions):
    """Extract admissible actions from observation"""
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    admissible_indices = np.where(observation[:max_directions] == 1)[0]
    return [i for i in admissible_indices if i < len(directions)]

def extract_distance_from_observation(observation):
    """Extract distance information from observation (placeholder)"""
    return 0  # Simplified for now

def load_realworld_environments():
    """
    Load realworld environments from shapefiles or .z8 files.
    
    Currently looks for:
    1. Any .z8 files that might correspond to realworld environments
    2. Shapefiles in the realworld folder for future conversion
    """
    realworld_envs = []
    
    # Check for any existing .z8 files that might be realworld environments
    # (This is a placeholder for when realworld shapefiles are converted)
    realworld_z8_pattern = "data/Environments/*realworld*.z8"
    realworld_z8_files = glob(realworld_z8_pattern)
    
    # Check for specific realworld environment names in existing .z8 files
    realworld_names = ['Emporia', 'Falcon', 'IVM', 'Chadstone']
    for env_name in realworld_names:
        pattern = f"data/Environments/*{env_name}*.z8"
        matching_files = glob(pattern)
        realworld_z8_files.extend(matching_files)
    
    # Process found .z8 files
    for z8_file in realworld_z8_files:
        env_name = os.path.basename(z8_file)
        env_name_parts = env_name.split('_')
        try:
            x_destination = float(env_name_parts[-2])
            y_destination = float(env_name_parts[-1].split('.')[0])
        except (IndexError, ValueError):
            x_destination = 0.0
            y_destination = 0.0
        
        realworld_envs.append({
            'env': env_name,
            'x_destination': x_destination,
            'y_destination': y_destination,
            'type': 'z8_file',
            'path': z8_file
        })
    
    # If no .z8 files found, check shapefiles for future use
    if not realworld_envs:
        print("No .z8 realworld environment files found.")
        print("Checking for shapefiles in realworld directory...")
        
        shapefile_dir = "data/realworld/shp"
        if os.path.exists(shapefile_dir):
            # Find unique environment groups based on prefixes
            shapefiles = glob(os.path.join(shapefile_dir, "*.shp"))
            env_prefixes = set()
            
            for shp_file in shapefiles:
                basename = os.path.basename(shp_file)
                # Extract environment prefix (e.g., 'Emporia', 'Falcon', 'IVM', 'Chadstone')
                for env_name in realworld_names:
                    if basename.startswith(env_name) or basename.startswith(env_name.lower()):
                        env_prefixes.add(env_name)
                        break
            
            for env_prefix in env_prefixes:
                # Create placeholder environment entries
                realworld_envs.append({
                    'env': f"{env_prefix}_realworld",
                    'x_destination': 0.0,  # Will need to be determined from shapefile
                    'y_destination': 0.0,  # Will need to be determined from shapefile
                    'type': 'shapefile',
                    'path': os.path.join(shapefile_dir, f"{env_prefix}*")
                })
                
        if realworld_envs:
            print(f"Found {len(realworld_envs)} realworld environment groups from shapefiles:")
            for env in realworld_envs:
                print(f"  - {env['env']} ({env['type']})")
        else:
            print("No realworld environments found in shapefiles either.")
    
    return realworld_envs

def load_shapefile_as_environment(shapefile_pattern):
    """
    Load shapefile and convert to environment format.
    
    Args:
        shapefile_pattern: Pattern for finding shapefiles (e.g., "data/realworld/shp/Emporia*")
    
    Returns:
        game_dict: Dictionary mapping room_id -> {direction: connected_room_id}
        room_positions: Dictionary mapping room_id -> (x, y) coordinates
    """
    if not SHAPEFILE_SUPPORT:
        raise ImportError("Shapefile support requires geopandas and fiona libraries")
    
    import numpy as np
    from collections import defaultdict
    
    # Extract environment name from pattern
    pattern_basename = shapefile_pattern.split('/')[-1].replace('*', '')
    base_env_name = pattern_basename.split('_')[0]  # Keep original case for display
    
    # Try multiple environment name cases and graph patterns
    env_name_variants = [base_env_name.lower(), base_env_name.capitalize(), base_env_name.upper()]
    
    # Find the graph shapefile (prefer UTM version)
    # Try multiple patterns to catch different naming conventions
    patterns_to_try = [
        f'*graph*utm*.shp',
        f'*Graph*utm*.shp', 
        f'*navigation*graph*.shp',
        f'*graph*.shp',
        f'*Graph*.shp'
    ]
    
    graph_files = []
    for env_variant in env_name_variants:
        for pattern in patterns_to_try:
            # Replace the environment name in the pattern with the current variant
            base_dir = '/'.join(shapefile_pattern.split('/')[:-1])
            test_pattern = f"{base_dir}/{env_variant}{pattern}"
            graph_files = glob(test_pattern)
            if graph_files:
                break
        if graph_files:
            break
    
    if not graph_files:
        print(f"No graph shapefile found for pattern: {shapefile_pattern}")
        return None, None
    
    # Use the first graph file found
    graph_file = graph_files[0]
    print(f"Processing graph file: {os.path.basename(graph_file)}")
    
    try:
        # Read the shapefile
        gdf = gpd.read_file(graph_file)
        print(f"Loaded {len(gdf)} edges from {os.path.basename(graph_file)}")
        
        # Extract all endpoints from LineStrings
        all_points = []
        edge_connections = []  # Store (point1_idx, point2_idx) for each edge
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if hasattr(geom, 'coords'):
                coords = list(geom.coords)
                if len(coords) >= 2:
                    # Get start and end points
                    start_point = coords[0]
                    end_point = coords[-1]
                    
                    all_points.append(start_point)
                    all_points.append(end_point)
                    
                    # Store connection (will resolve indices after deduplication)
                    edge_connections.append((len(all_points)-2, len(all_points)-1))
        
        if not all_points:
            print(f"No valid points found in {graph_file}")
            return None, None
        
        print(f"Extracted {len(all_points)} raw endpoints")
        
        # Convert to numpy array for efficient processing
        points_array = np.array(all_points)
        
        # De-duplicate nearby points (within 1 meter tolerance)
        tolerance = 1.0  # meters
        unique_points = []
        point_mapping = {}  # maps original index to unique index
        
        for i, point in enumerate(points_array):
            # Check if this point is close to any existing unique point
            is_duplicate = False
            for j, unique_point in enumerate(unique_points):
                distance = np.sqrt(np.sum((point - unique_point) ** 2))
                if distance < tolerance:
                    point_mapping[i] = j
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                point_mapping[i] = len(unique_points)
                unique_points.append(point)
        
        unique_points = np.array(unique_points)
        print(f"After deduplication: {len(unique_points)} unique nodes")
        
        # Build connectivity graph
        connections = defaultdict(set)
        for start_idx, end_idx in edge_connections:
            unique_start = point_mapping[start_idx]
            unique_end = point_mapping[end_idx]
            
            if unique_start != unique_end:  # Avoid self-loops
                connections[unique_start].add(unique_end)
                connections[unique_end].add(unique_start)  # Bidirectional
        
        # Create room IDs and positions
        room_positions = {}
        for i, point in enumerate(unique_points):
            room_id = f"{base_env_name}_room_{i}"
            room_positions[room_id] = (float(point[0]), float(point[1]))
        
        # Convert connections to game_dict format with directional actions
        game_dict = {}
        room_ids = list(room_positions.keys())
        
        for i, room_id in enumerate(room_ids):
            game_dict[room_id] = {}
            
            # Get connected rooms
            connected_indices = connections[i]
            
            for connected_idx in connected_indices:
                connected_room = room_ids[connected_idx]
                
                # Calculate direction from current room to connected room
                current_pos = unique_points[i]
                connected_pos = unique_points[connected_idx]
                
                # Calculate angle (in radians)
                dx = connected_pos[0] - current_pos[0]
                dy = connected_pos[1] - current_pos[1]
                angle = np.arctan2(dy, dx)
                
                # Convert angle to direction (8-sector)
                # 0° = East, 90° = North, 180° = West, 270° = South
                angle_degrees = np.degrees(angle) % 360
                
                if angle_degrees < 22.5 or angle_degrees >= 337.5:
                    direction = "east"
                elif angle_degrees < 67.5:
                    direction = "northeast" 
                elif angle_degrees < 112.5:
                    direction = "north"
                elif angle_degrees < 157.5:
                    direction = "northwest"
                elif angle_degrees < 202.5:
                    direction = "west"
                elif angle_degrees < 247.5:
                    direction = "southwest"
                elif angle_degrees < 292.5:
                    direction = "south"
                else:  # 292.5 <= angle_degrees < 337.5
                    direction = "southeast"
                
                game_dict[room_id][direction] = connected_room
        
        print(f"Created navigation graph with {len(room_positions)} rooms")
        print(f"Average connections per room: {np.mean([len(connections[i]) for i in range(len(unique_points))]):.1f}")
        
        return game_dict, room_positions
        
    except Exception as e:
        print(f"Error processing shapefile {graph_file}: {e}")
        return None, None

def evaluate_model(model, env, n_episodes=100):
    """Evaluate model on environment for specified number of episodes"""
    try:
        evaluation_result = evaluate_policy(
            model.policy, env, n_eval_episodes=n_episodes,
            deterministic=False, warn=False, return_episode_rewards=False
        )

        if not isinstance(evaluation_result, tuple) or len(evaluation_result) != 2:
            mean_success = np.nan
            std_success = np.nan
        else:
            mean_success, std_success = evaluation_result

        return mean_success, std_success

    except Exception as e:
        print(f"Exception in evaluate_model: {e}")
        return np.nan, np.nan

def evaluate_random_agent(env, n_eval_episodes=100, verbose=False):
    """Evaluate a random agent on the given environment"""
    successes = []

    for episode in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        episode_actions = []
        episode_distances = []
        reward = 0

        while not done and not truncated:
            # Extract admissible actions from the observation
            admissible_actions = get_admissible_actions_from_observation(observation, env.directions)

            if not admissible_actions:
                action = env.max_directions  # "look" action
            else:
                action = random.choice(admissible_actions)

            episode_actions.append(action)
            observation, reward, done, truncated, _ = env.step(action)
            
            distance = extract_distance_from_observation(observation)
            episode_distances.append(distance)

        successes.append(reward)

        if verbose:
            print(f"Episode {episode + 1}:")
            print(f"  Actions Taken: {episode_actions}")
            print(f"  Distances: {episode_distances}")
            print(f"  Success: {reward}")

    return successes

def determine_complexity(env_name):
    """Determine environment complexity (placeholder)"""
    # Simple heuristic based on name length for now
    if len(env_name) < 20:
        return "low"
    elif len(env_name) < 40:
        return "medium"
    else:
        return "high"

def create_result_entry(model_name, env_name, complexity, grammar, instruction_type, 
                       reward_type, average_success_rate, std_success_rate, 
                       random_agent_average_success_rate, random_agent_std_success_rate, is_seen):
    """Create a result entry for the CSV"""
    return {
        'model_name': model_name,
        'env_name': env_name,
        'complexity': complexity,
        'grammar': grammar,
        'instruction_type': instruction_type,
        'reward_type': reward_type,
        'average_success_rate': average_success_rate,
        'std_success_rate': std_success_rate,
        'random_agent_average_success_rate': random_agent_average_success_rate,
        'random_agent_std_success_rate': random_agent_std_success_rate,
        'is_seen': is_seen
    }

def main():
    """Main evaluation function"""
    print("=== Realworld Environment PPO Evaluation ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load realworld environments
    realworld_envs = load_realworld_environments()
    
    if not realworld_envs:
        print("No realworld environments found. Please ensure:")
        print("1. Realworld .z8 files exist in data/Environments/")
        print("2. Or shapefiles exist in data/realworld/shp/")
        print("3. If using shapefiles, they need to be converted to .z8 format first")
        return
    
    # Define evaluation parameters
    grammars = [8, 6, 4]
    reward_types = ['sparse', 'step_cost']
    results = []
    n_episodes = 100  # As requested by user
    n_instructions = 10
    
    print(f"Found {len(realworld_envs)} realworld environments")
    print(f"Will evaluate with {n_episodes} episodes per environment")
    
    for reward_type in reward_types:
        print(f"\n=== Evaluating {reward_type} reward model ===")
        
        # Load curriculum model
        model_path = f'data/trained/curriculum/PPO/{reward_type}/best_model.zip'
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping {reward_type}.")
            continue
        
        try:
            model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Loaded {reward_type} model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue
        
        # Evaluate across all grammars and realworld environments
        for grammar in grammars:
            print(f"\nEvaluating grammar {grammar}")
            
            for env_info in tqdm(realworld_envs, desc="Realworld Environments"):
                env_name = env_info['env']
                
                try:
                    # Load environment
                    if env_info['type'] == 'z8_file':
                        # Load from .z8 file
                        game_dict, room_positions = z8file_to_dictionaries(env_info['path'])
                    elif env_info['type'] == 'shapefile':
                        # Load from shapefile (not yet implemented)
                        game_dict, room_positions = load_shapefile_as_environment(env_info['path'])
                        if game_dict is None:
                            print(f"Skipping {env_name}: Shapefile processing not implemented")
                            continue
                    else:
                        print(f"Skipping {env_name}: Unknown environment type")
                        continue
                    
                except Exception as e:
                    print(f"Skipping {env_name}: {e}")
                    continue
                
                # Create environment instance
                env = TextWorldEnv(
                    game_dict=game_dict,
                    room_positions=room_positions,
                    n_instructions=n_instructions,
                    grammar=grammar,
                    reward_type=reward_type
                )
                env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
                
                # Evaluate PPO model
                print(f"Evaluating PPO model on {env_name} with {n_episodes} episodes...")
                success_trained, std_trained = evaluate_model(model, env, n_episodes=n_episodes)
                
                # Evaluate random agent
                print(f"Evaluating random agent on {env_name} with {n_episodes} episodes...")
                random_rewards = evaluate_random_agent(env, n_eval_episodes=n_episodes)
                random_successes = [1 if r >= 25 else 0 for r in random_rewards]
                success_random = np.mean(random_successes)
                std_random = np.std(random_successes)
                
                # Store results
                result_entry = create_result_entry(
                    model_name=f"curriculum_{reward_type}",
                    env_name=env_name,
                    complexity=determine_complexity(env_name),
                    grammar=grammar,
                    instruction_type='complete',
                    reward_type=reward_type,
                    average_success_rate=abs(success_trained)/100 if not np.isnan(success_trained) else np.nan,
                    std_success_rate=std_trained if not np.isnan(std_trained) else np.nan,
                    random_agent_average_success_rate=success_random,
                    random_agent_std_success_rate=std_random,
                    is_seen='realworld'  # Mark as realworld instead of seen/unseen
                )
                results.append(result_entry)
                
                print(f"PPO Success Rate: {success_trained:.3f} ± {std_trained:.3f}")
                print(f"Random Success Rate: {success_random:.3f} ± {std_random:.3f}")
    
    # Save results to CSV
    if results:
        output_filename = 'realworld_evaluation_results_PPO.csv'
        output_path = f'data/{output_filename}'
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved to: {output_path}")
        print(f"Total evaluations: {len(results)}")
        
        # Print summary statistics
        if len(results) > 0:
            print("\nSummary Statistics:")
            for reward_type in reward_types:
                reward_results = df[df['reward_type'] == reward_type]
                if len(reward_results) > 0:
                    mean_ppo = reward_results['average_success_rate'].mean()
                    mean_random = reward_results['random_agent_average_success_rate'].mean()
                    print(f"{reward_type}: PPO={mean_ppo:.3f}, Random={mean_random:.3f}")
    else:
        print("\nNo results to save. Please check environment loading.")

if __name__ == "__main__":
    # Load NLP model if available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spacy model 'en_core_web_sm' not found")
        nlp = None
    
    main() 