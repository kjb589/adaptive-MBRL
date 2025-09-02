# RL Training loop that uses neural network-based forward dynamics modeling
# and learns to find a path through the environment.
# Input:
#   - Latent representation of spatiotemporal info about the environment
#   - a maze (coordinate grid)
#   - Start point
#   - Goal point
# Output:
#   - a list of tuples as a path to next step
from math import sqrt
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp
import numpy as np
import heapq


# Is it ready?
# No

class Node:
    """A node class for path planning"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class PathPlanner:
    """RL Path Planner"""
    def __init__(self, model, state, maze, start, end):
        self.model = model
        self.state = state
        self.maze = maze
        self.start = start
        self.end = end
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def is_valid(self, position):
        x, y = position
        if x < self.start[0] or y < self.start[1]:
            return False
        if x >= self.start[0] + self.maze.shape[0] or y >= self.start[1] + self.maze.shape[1]:
            return False
        if self.maze[x, y] == 1:
            '''What is the purpose'''
            return False
        return True

    def dist_to_goal(self, position):
        return sqrt(pow(position[0] - self.end[0], 2) + pow(position[1] - self.end[1], 2))

    def predict_next_state(self, current_state, action):
        '''Use context models to predict next state'''
        input_tensor = self.convert_to_tensor(current_state, action)
        next_state = self.model.predict(input_tensor)
        return tuple(next_state)

    #def heuristic(self, pos, goal):
    #    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def convert_to_tensor(self, state, action):
        """
        Prepares the input tensor for the dynamics model.
        This will depend on how your model is trained.
        """
        x, y = state
        dx, dy = action
        input_array = np.array([*self.state.flatten(), x, y, dx, dy])
        return input_array.reshape(1, -1)

    def reconstruct_path(self, current_node):
        path = []
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1]  # reverse

    def plan(self, max_steps=1000):
        """
        Plans a path using the learned model to simulate forward dynamics.
        """
        start_node = Node(None, self.start)
        end_node = Node(None, self.end)

        open_list = []
        heapq.heappush(open_list, (start_node.f, start_node))
        closed_set = set()

        while open_list:
            _, current_node = heapq.heappop(open_list)

            if current_node.position == end_node.position:
                return self.reconstruct_path(current_node)

            closed_set.add(current_node.position)

            for action in self.actions:
                predicted_state = self.predict_next_state(current_node.position, action)

                if not self.is_valid(predicted_state) or predicted_state in closed_set:
                    continue

                child = Node(current_node, predicted_state)
                child.g = current_node.g + 1
                child.h = (pow((child.position[0] - end_node.position[0]), 2)) + (pow((child.position[1] - end_node.position[1]), 2))
                child.f = child.g + child.h

                heapq.heappush(open_list, (child.f, child))

        return None  # No path found


class Planner(Model):
    def __init__(self, attention_model, ensemble_dynamics, actor, critic, action_dim):
        super().__init__()
        self.attention = attention_model        # Pretrained attention mechanism
        self.ensemble = ensemble_dynamics        # EnsembleDynamics instance
        self.actor = actor                       # Actor network
        self.critic = critic                     # Critic network
        self.action_dim = action_dim

    def call(self, z, training=False):
        """
        z: current latent state, shape [batch, latent_dim]
        """

        # 1. Get attention weights for dynamics ensemble from attention mechanism
        # Assuming attention takes latent state and outputs weights [batch, ensemble_size]
        weights = self.attention(z, training=training)  # shape: [batch, ensemble_size]

        # 2. Actor outputs action probabilities for current state
        action_probs = self.actor(z)  # shape: [batch, action_dim]

        # 3. Sample action from actor policy (or use greedy argmax during evaluation)
        action_dist = tfp.distributions.Categorical(probs=action_probs)
        sampled_action = action_dist.sample()  # shape: [batch]

        # Convert to one-hot for ensemble input
        action_onehot = tf.one_hot(sampled_action, depth=self.action_dim)  # [batch, action_dim]

        # 4. Ensemble predicts next latent state weighted by attention weights
        z_next = self.ensemble.call(z, action_onehot, weights)  # [batch, latent_dim]

        # 5. Critic estimates value of current state-action pair
        q_value = self.critic(z, action_onehot)  # [batch, 1]

        return {
            "action_probs": action_probs,
            "sampled_action": sampled_action,
            "z_next": z_next,
            "q_value": q_value,
            "attention_weights": weights
        }

    def rollout(self, z_init, horizon=5):
        """
        Generate imagined rollout starting from initial latent state z_init.
        """
        rollout_states = [z_init]
        rollout_actions = []
        rollout_qvals = []

        z_t = z_init

        for t in range(horizon):
            weights = self.attention(z_t, training=False)

            action_probs = self.actor(z_t)
            action_dist = tfp.distributions.Categorical(probs=action_probs)
            action_t = action_dist.sample()
            action_onehot = tf.one_hot(action_t, depth=self.action_dim)

            z_next = self.ensemble.call(z_t, action_onehot, weights)

            q_val = self.critic(z_t, action_onehot)

            rollout_states.append(z_next)
            rollout_actions.append(action_t)
            rollout_qvals.append(q_val)

            # Update state for next step
            z_t = z_next

        return rollout_states, rollout_actions, rollout_qvals
