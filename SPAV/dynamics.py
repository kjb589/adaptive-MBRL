import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *


# Is it ready?
# Yes


class Dynamics(Model):
    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.input_layer = Dense(256, activation='relu')
        self.hidden1 = Dense(256, activation='relu')
        self.hidden2 = Dense(256, activation='relu')
        self.output_layer = Dense(latent_dim)

        self.use_skip = True

    def call(self, z, a):
        x = tf.concat([z, a], axis=-1)

        h = self.input_layer(x)
        h = self.hidden1(h)
        h = self.hidden2(h)
        pred = self.output_layer(h)

        if self.use_skip:
            z_next = z + pred
        else:
            z_next = pred

        return z_next


class EnsembleDynamics:
    def __init__(self, latent_dim, action_dim, ensemble_size=2):
        self.models = [Dynamics(latent_dim, action_dim) for _ in range(ensemble_size)]

    def predict(self, state, action):
        return [model(state, action) for model in self.models]

    def call(self, state, action, weights):
        """
        state: [batch, state_dim]
        action: [batch, action_dim]
        weights: [batch, num_models]
        """
        preds = tf.stack(self.predict(state, action), axis=1)
        weights = tf.expand_dims(weights, axis=-1)
        weighted_pred = tf.reduce_sum(preds * weights, axis=1)
        return weighted_pred


class Actor(Model):
    def __init__(self, state_dim, action_dim=8):
        super().__init__()
        self.normalizer = Normalization()
        self.planner = Sequential([
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(action_dim, activation='softmax')
        ])

    def adapt_normalizer(self, states):
        self.normalizer.adapt(states)

    def call(self, state):
        state = self.normalizer(state)
        return self.planner(state)


class Critic(Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_norm = Normalization()
        self.action_norm = Normalization()
        self.critic = Sequential([
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1)
        ])

    def adapt_normalizer(self, states, actions):
        self.state_norm.adapt(states)
        self.action_norm.adapt(actions)

    def call(self, state, action):
        state = self.state_norm(state)
        action = self.action_norm(action)
        x = tf.concat([state, action], axis=-1)
        return self.critic(x)